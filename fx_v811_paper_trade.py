import json
import math
import os
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# FX PAPER TRADE v811
# =========================================================
#
# 目的:
# - v810 daily signal の候補を paper trade として管理する
# - 候補は「次の営業日のOpenでエントリー予定」として PENDING 登録
# - 次回以降、該当日のOpenが取得できたら OPEN に変換
# - OPENポジションは毎日チェックして、決済条件に到達したら TRADES に記録
#
# 対象:
# - V700_RSI_PULLBACK
# - V800_ATR_TREND
#
# 実行:
#   python .\fx_v811_paper_trade.py
#
# 前提:
#   先に v810 を実行して fx_v810_daily_candidates.csv を作成しておく
#
# 入力:
# - fx_v810_daily_candidates.csv
#
# 出力:
# - fx_v811_positions.csv
# - fx_v811_trades.csv
# - fx_v811_equity.csv
# - fx_v811_daily_report.csv
#
# Discord通知:
# - DISCORD_WEBHOOK_URL があれば通知
#
# =========================================================


# =========================================================
# BASIC SETTINGS
# =========================================================

START = "2018-01-01"
END = None

INITIAL_EQUITY = 100_000

CANDIDATES_FILE = "fx_v810_daily_candidates.csv"

POSITIONS_FILE = "fx_v811_positions.csv"
TRADES_FILE = "fx_v811_trades.csv"
EQUITY_FILE = "fx_v811_equity.csv"
REPORT_FILE = "fx_v811_daily_report.csv"

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()


# =========================================================
# SPREAD SETTINGS
# =========================================================

SPREAD_PIPS = {
    "USDJPY=X": 0.3,
    "EURUSD=X": 0.2,
    "GBPUSD=X": 0.6,
    "AUDUSD=X": 0.5,
    "EURJPY=X": 0.5,
    "GBPJPY=X": 0.9,
}

DEFAULT_SPREAD_PIPS = 0.5


# =========================================================
# CSV SCHEMAS
# =========================================================

POSITION_COLUMNS = [
    "status",                 # PENDING / OPEN
    "strategy",
    "param_name",
    "pair",
    "side",
    "signal_date",
    "entry_date",
    "entry_price",
    "position_fraction",

    # V800 stop/trailing
    "initial_stop",
    "current_stop",
    "highest_high",
    "lowest_low",
    "signal_atr",

    # V700 TP/SL
    "tp_price",
    "sl_price",

    # management
    "last_checked_date",
    "opened_at",
    "updated_at",

    # params
    "pullback_pct",
    "tp_pct",
    "sl_pct",
    "hold_days",
    "rsi_long_max",
    "rsi_short_min",
    "breakout_days",
    "ma_days",
    "atr_days",
    "atr_mult",
    "max_hold_days",

    "note",
]

TRADE_COLUMNS = [
    "strategy",
    "param_name",
    "pair",
    "side",
    "signal_date",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "exit_reason",
    "bar_hold_days",
    "calendar_hold_days",
    "gross_return",
    "spread_cost_pct",
    "net_return",
    "position_fraction",
    "pnl",
    "equity_after",
    "initial_stop",
    "final_stop",
    "tp_price",
    "sl_price",
    "created_at",
]

EQUITY_COLUMNS = [
    "timestamp",
    "equity",
    "realized_pnl",
    "trade_count",
]

REPORT_COLUMNS = [
    "run_datetime",
    "candidate_rows",
    "new_pending_orders",
    "opened_positions",
    "closed_trades",
    "open_positions",
    "pending_positions",
    "equity",
]

# =========================================================
# COLUMN TYPE SETTINGS
# =========================================================
#
# GitHub Actions上のpandasでは、空文字だけの列がfloat64として読まれることがある。
# その状態で opened_at などへ文字列を代入すると、
# TypeError: Invalid value 'YYYY-MM-DD HH:MM:SS' for dtype 'float64'
# になるため、CSV読み込み時点で文字列列をobject型へ揃える。
#

STRING_COLUMNS = {
    "status",
    "strategy",
    "param_name",
    "pair",
    "side",
    "signal_date",
    "entry_date",
    "last_checked_date",
    "opened_at",
    "updated_at",
    "note",
    "exit_date",
    "exit_reason",
    "created_at",
    "timestamp",
    "run_datetime",
}



# =========================================================
# UTILS
# =========================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def to_date_str(value) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return ""

    return pd.Timestamp(value).strftime("%Y-%m-%d")


def get_float(row, col: str, default=np.nan) -> float:
    try:
        value = row.get(col, default)
    except Exception:
        return default

    if value is None:
        return default

    if isinstance(value, str) and value.strip() == "":
        return default

    if pd.isna(value):
        return default

    try:
        return float(value)
    except Exception:
        return default


def get_int(row, col: str, default=0) -> int:
    value = get_float(row, col, np.nan)

    if pd.isna(value):
        return default

    return int(value)


def normalize_csv_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    CSV読み込み/保存前に列とdtypeを安定化する。

    特にGitHub Actions上のpandasでは、空欄だけの opened_at などが
    float64として推定され、あとから日時文字列を代入すると落ちる。
    """
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            if col in STRING_COLUMNS:
                out[col] = ""
            else:
                out[col] = np.nan

    out = out[columns].copy()

    for col in columns:
        if col in STRING_COLUMNS:
            out[col] = out[col].replace({np.nan: ""}).astype("object")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def read_csv_or_empty(path: str, columns: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return normalize_csv_columns(pd.DataFrame(), columns)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return normalize_csv_columns(pd.DataFrame(), columns)

    return normalize_csv_columns(df, columns)


def save_csv(df: pd.DataFrame, path: str, columns: list[str] | None = None) -> None:
    if columns is not None:
        df = normalize_csv_columns(df, columns)

    df.to_csv(path, index=False)


def pip_size(pair: str) -> float:
    name = pair.replace("=X", "")

    if "JPY" in name:
        return 0.01

    return 0.0001


def spread_cost_pct(pair: str, price: float) -> float:
    spread_pips = SPREAD_PIPS.get(pair, DEFAULT_SPREAD_PIPS)
    spread_price = spread_pips * pip_size(pair)

    return spread_price / price


def calc_gross_return(side: str, entry_price: float, exit_price: float) -> float:
    if side == "LONG":
        return (exit_price - entry_price) / entry_price

    return (entry_price - exit_price) / entry_price


def current_equity_from_trades(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return INITIAL_EQUITY

    if "equity_after" in trades_df.columns and trades_df["equity_after"].notna().any():
        return float(trades_df["equity_after"].dropna().iloc[-1])

    equity = INITIAL_EQUITY

    for _, tr in trades_df.iterrows():
        net_return = get_float(tr, "net_return", 0.0)
        position_fraction = get_float(tr, "position_fraction", 1.0)
        equity += equity * position_fraction * net_return

    return float(equity)


# =========================================================
# DATA FETCH / INDICATORS
# =========================================================

def normalize_price_df(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    for col in required:
        if col not in df.columns:
            print(f"[WARN] missing column {col}: {pair}")
            return pd.DataFrame()

    out = df[required].copy()
    out = out.dropna()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    return out


def fetch_pair(pair: str) -> pd.DataFrame:
    print(f"[FETCH] {pair}")

    # yfinanceは start 指定の長期取得で、最新1本が遅れて返ることがある。
    # そのため、長期データ + 直近10日データを別々に取得してマージする。
    long_df = yf.download(
        pair,
        start=START,
        end=END,
        auto_adjust=False,
        progress=False,
    )

    recent_df = yf.download(
        pair,
        period="10d",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    long_df = normalize_price_df(long_df, pair)
    recent_df = normalize_price_df(recent_df, pair)

    if long_df.empty and recent_df.empty:
        print(f"[WARN] no data: {pair}")
        return pd.DataFrame()

    if long_df.empty:
        df = recent_df
    elif recent_df.empty:
        df = long_df
    else:
        df = pd.concat([long_df, recent_df])
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

    if not df.empty:
        print(f"[LATEST] {pair}: {df.index[-1].date()} open={float(df['Open'].iloc[-1]):.6f}")

    return df


def calc_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()

    return atr


def make_v800_features(df: pd.DataFrame, ma_days: int, atr_days: int, breakout_days: int) -> pd.DataFrame:
    out = df.copy()

    out["ma"] = out["Close"].rolling(ma_days).mean()
    out["atr"] = calc_atr(out, atr_days)

    out["prev_high"] = out["High"].rolling(breakout_days).max().shift(1)
    out["prev_low"] = out["Low"].rolling(breakout_days).min().shift(1)

    out = out.dropna().copy()

    return out


# =========================================================
# LOAD CANDIDATES
# =========================================================

def load_candidates() -> pd.DataFrame:
    if not os.path.exists(CANDIDATES_FILE):
        print(f"[WARN] candidates file not found: {CANDIDATES_FILE}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(CANDIDATES_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if df.empty:
        return df

    if "signal_date" in df.columns:
        df["signal_date"] = df["signal_date"].apply(to_date_str)

    return df


# =========================================================
# POSITION MANAGEMENT
# =========================================================

def has_existing_position_or_order(
    positions_df: pd.DataFrame,
    strategy: str,
    pair: str,
    side: str,
) -> bool:
    if positions_df.empty:
        return False

    active = positions_df[positions_df["status"].isin(["PENDING", "OPEN"])].copy()

    if active.empty:
        return False

    mask = (
        (active["strategy"] == strategy)
        & (active["pair"] == pair)
        & (active["side"] == side)
    )

    return bool(mask.any())


def has_same_signal(
    positions_df: pd.DataFrame,
    strategy: str,
    pair: str,
    side: str,
    signal_date: str,
) -> bool:
    if positions_df.empty:
        return False

    mask = (
        (positions_df["strategy"] == strategy)
        & (positions_df["pair"] == pair)
        & (positions_df["side"] == side)
        & (positions_df["signal_date"].astype(str) == signal_date)
        & (positions_df["status"].isin(["PENDING", "OPEN"]))
    )

    return bool(mask.any())


def add_pending_orders(
    positions_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    new_orders = []

    if candidates_df.empty:
        return positions_df, new_orders

    for _, cand in candidates_df.iterrows():
        strategy = str(cand.get("strategy", "")).strip()
        pair = str(cand.get("pair", "")).strip()
        side = str(cand.get("side", "")).strip()
        signal_date = to_date_str(cand.get("signal_date"))

        if not strategy or not pair or not side or not signal_date:
            continue

        if has_same_signal(positions_df, strategy, pair, side, signal_date):
            print(f"[SKIP] same signal already exists: {strategy} {pair} {side} {signal_date}")
            continue

        if has_existing_position_or_order(positions_df, strategy, pair, side):
            print(f"[SKIP] active position/order already exists: {strategy} {pair} {side}")
            continue

        position_fraction = get_float(cand, "position_fraction", 1.0)

        order = {
            "status": "PENDING",
            "strategy": strategy,
            "param_name": str(cand.get("param_name", "")),
            "pair": pair,
            "side": side,
            "signal_date": signal_date,
            "entry_date": "",
            "entry_price": np.nan,
            "position_fraction": position_fraction,

            "initial_stop": get_float(cand, "stop_ref", np.nan),
            "current_stop": get_float(cand, "stop_ref", np.nan),
            "highest_high": get_float(cand, "close", np.nan) if side == "LONG" else np.nan,
            "lowest_low": get_float(cand, "close", np.nan) if side == "SHORT" else np.nan,
            "signal_atr": get_float(cand, "atr", np.nan),

            "tp_price": np.nan,
            "sl_price": np.nan,

            "last_checked_date": "",
            "opened_at": "",
            "updated_at": now_str(),

            "pullback_pct": get_float(cand, "pullback_pct", np.nan),
            "tp_pct": get_float(cand, "tp_pct", np.nan),
            "sl_pct": get_float(cand, "sl_pct", np.nan),
            "hold_days": get_int(cand, "hold_days", 0),
            "rsi_long_max": get_int(cand, "rsi_long_max", 0),
            "rsi_short_min": get_int(cand, "rsi_short_min", 0),

            "breakout_days": get_int(cand, "breakout_days", 0),
            "ma_days": get_int(cand, "ma_days", 0),
            "atr_days": get_int(cand, "atr_days", 0),
            "atr_mult": get_float(cand, "atr_mult", np.nan),
            "max_hold_days": get_int(cand, "max_hold_days", 0),

            "note": "created from v810 candidate",
        }

        new_orders.append(order)
        positions_df = pd.concat([positions_df, pd.DataFrame([order])], ignore_index=True)

        print(f"[NEW PENDING] {strategy} {pair} {side} signal={signal_date}")

    return positions_df, new_orders


def process_pending_entries(
    positions_df: pd.DataFrame,
    raw_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[dict]]:
    opened = []

    if positions_df.empty:
        return positions_df, opened

    for idx, pos in positions_df.iterrows():
        if str(pos.get("status", "")) != "PENDING":
            continue

        pair = str(pos.get("pair", ""))
        side = str(pos.get("side", ""))
        strategy = str(pos.get("strategy", ""))
        signal_date = to_date_str(pos.get("signal_date"))

        if not pair or pair not in raw_data or not signal_date:
            continue

        df = raw_data[pair]

        if df.empty:
            continue

        future = df[df.index > pd.Timestamp(signal_date)]

        if future.empty:
            # まだ次足がないのでPENDINGのまま
            continue

        entry_date = future.index[0]
        entry_row = future.iloc[0]
        entry_price = float(entry_row["Open"])

        positions_df.at[idx, "status"] = "OPEN"
        positions_df.at[idx, "entry_date"] = entry_date.strftime("%Y-%m-%d")
        positions_df.at[idx, "entry_price"] = entry_price
        positions_df.at[idx, "opened_at"] = now_str()
        positions_df.at[idx, "updated_at"] = now_str()
        positions_df.at[idx, "last_checked_date"] = ""

        if strategy == "V700_RSI_PULLBACK":
            tp_pct = get_float(pos, "tp_pct", np.nan)
            sl_pct = get_float(pos, "sl_pct", np.nan)

            if side == "LONG":
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)

            positions_df.at[idx, "tp_price"] = tp_price
            positions_df.at[idx, "sl_price"] = sl_price

        elif strategy == "V800_ATR_TREND":
            atr_mult = get_float(pos, "atr_mult", np.nan)
            signal_atr = get_float(pos, "signal_atr", np.nan)

            if not math.isfinite(signal_atr) or signal_atr <= 0:
                signal_atr = np.nan

            if math.isfinite(signal_atr) and math.isfinite(atr_mult):
                if side == "LONG":
                    stop_price = entry_price - signal_atr * atr_mult
                    positions_df.at[idx, "highest_high"] = entry_price
                    positions_df.at[idx, "lowest_low"] = np.nan
                else:
                    stop_price = entry_price + signal_atr * atr_mult
                    positions_df.at[idx, "highest_high"] = np.nan
                    positions_df.at[idx, "lowest_low"] = entry_price

                positions_df.at[idx, "initial_stop"] = stop_price
                positions_df.at[idx, "current_stop"] = stop_price

        opened_row = positions_df.loc[idx].to_dict()
        opened.append(opened_row)

        print(f"[OPEN] {strategy} {pair} {side} entry_date={entry_date.date()} entry_price={entry_price}")

    return positions_df, opened


def append_trade(
    trades_df: pd.DataFrame,
    trade: dict,
) -> pd.DataFrame:
    return pd.concat([trades_df, pd.DataFrame([trade])], ignore_index=True)


def make_trade_row(
    pos,
    exit_date,
    exit_price: float,
    exit_reason: str,
    bar_hold_days: int,
    current_equity: float,
    final_stop=np.nan,
) -> dict:
    strategy = str(pos.get("strategy", ""))
    pair = str(pos.get("pair", ""))
    side = str(pos.get("side", ""))
    entry_price = get_float(pos, "entry_price", np.nan)
    position_fraction = get_float(pos, "position_fraction", 1.0)

    gross_return = calc_gross_return(side, entry_price, exit_price)
    cost_pct = spread_cost_pct(pair, entry_price) * 2
    net_return = gross_return - cost_pct

    pnl = current_equity * position_fraction * net_return
    equity_after = current_equity + pnl

    entry_date = pd.Timestamp(pos.get("entry_date"))
    exit_ts = pd.Timestamp(exit_date)

    return {
        "strategy": strategy,
        "param_name": str(pos.get("param_name", "")),
        "pair": pair,
        "side": side,
        "signal_date": to_date_str(pos.get("signal_date")),
        "entry_date": to_date_str(pos.get("entry_date")),
        "exit_date": exit_ts.strftime("%Y-%m-%d"),
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "bar_hold_days": int(bar_hold_days),
        "calendar_hold_days": int((exit_ts - entry_date).days),
        "gross_return": float(gross_return),
        "spread_cost_pct": float(cost_pct),
        "net_return": float(net_return),
        "position_fraction": float(position_fraction),
        "pnl": float(pnl),
        "equity_after": float(equity_after),
        "initial_stop": get_float(pos, "initial_stop", np.nan),
        "final_stop": float(final_stop) if math.isfinite(float(final_stop)) else np.nan,
        "tp_price": get_float(pos, "tp_price", np.nan),
        "sl_price": get_float(pos, "sl_price", np.nan),
        "created_at": now_str(),
    }


def process_open_v700_position(
    pos,
    raw_df: pd.DataFrame,
    current_equity: float,
) -> tuple[dict | None, dict]:
    pair = str(pos.get("pair", ""))
    side = str(pos.get("side", ""))
    entry_date = to_date_str(pos.get("entry_date"))
    last_checked_date = to_date_str(pos.get("last_checked_date"))

    if not entry_date:
        return None, pos.to_dict()

    df = raw_df.copy()
    post_entry = df[df.index >= pd.Timestamp(entry_date)]

    if post_entry.empty:
        return None, pos.to_dict()

    if last_checked_date:
        check_df = post_entry[post_entry.index > pd.Timestamp(last_checked_date)]
    else:
        check_df = post_entry

    if check_df.empty:
        return None, pos.to_dict()

    tp_price = get_float(pos, "tp_price", np.nan)
    sl_price = get_float(pos, "sl_price", np.nan)
    hold_days = get_int(pos, "hold_days", 0)

    if not math.isfinite(tp_price) or not math.isfinite(sl_price) or hold_days <= 0:
        return None, pos.to_dict()

    for date, row in check_df.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])

        bar_hold_days = int(len(post_entry[post_entry.index <= date]))

        if side == "LONG":
            if low <= sl_price:
                trade = make_trade_row(
                    pos,
                    exit_date=date,
                    exit_price=sl_price,
                    exit_reason="SL",
                    bar_hold_days=bar_hold_days,
                    current_equity=current_equity,
                    final_stop=sl_price,
                )
                return trade, pos.to_dict()

            if high >= tp_price:
                trade = make_trade_row(
                    pos,
                    exit_date=date,
                    exit_price=tp_price,
                    exit_reason="TP",
                    bar_hold_days=bar_hold_days,
                    current_equity=current_equity,
                    final_stop=sl_price,
                )
                return trade, pos.to_dict()

        else:
            if high >= sl_price:
                trade = make_trade_row(
                    pos,
                    exit_date=date,
                    exit_price=sl_price,
                    exit_reason="SL",
                    bar_hold_days=bar_hold_days,
                    current_equity=current_equity,
                    final_stop=sl_price,
                )
                return trade, pos.to_dict()

            if low <= tp_price:
                trade = make_trade_row(
                    pos,
                    exit_date=date,
                    exit_price=tp_price,
                    exit_reason="TP",
                    bar_hold_days=bar_hold_days,
                    current_equity=current_equity,
                    final_stop=sl_price,
                )
                return trade, pos.to_dict()

        if bar_hold_days >= hold_days:
            trade = make_trade_row(
                pos,
                exit_date=date,
                exit_price=close,
                exit_reason="TIME",
                bar_hold_days=bar_hold_days,
                current_equity=current_equity,
                final_stop=sl_price,
            )
            return trade, pos.to_dict()

    updated = pos.to_dict()
    updated["last_checked_date"] = check_df.index[-1].strftime("%Y-%m-%d")
    updated["updated_at"] = now_str()

    return None, updated


def process_open_v800_position(
    pos,
    raw_df: pd.DataFrame,
    current_equity: float,
) -> tuple[dict | None, dict]:
    side = str(pos.get("side", ""))
    entry_date = to_date_str(pos.get("entry_date"))
    last_checked_date = to_date_str(pos.get("last_checked_date"))

    if not entry_date:
        return None, pos.to_dict()

    ma_days = get_int(pos, "ma_days", 200)
    atr_days = get_int(pos, "atr_days", 14)
    breakout_days = get_int(pos, "breakout_days", 60)
    atr_mult = get_float(pos, "atr_mult", np.nan)
    max_hold_days = get_int(pos, "max_hold_days", 20)

    if not math.isfinite(atr_mult) or atr_mult <= 0:
        return None, pos.to_dict()

    feat = make_v800_features(raw_df, ma_days=ma_days, atr_days=atr_days, breakout_days=breakout_days)

    post_entry = feat[feat.index >= pd.Timestamp(entry_date)]

    if post_entry.empty:
        return None, pos.to_dict()

    if last_checked_date:
        check_df = post_entry[post_entry.index > pd.Timestamp(last_checked_date)]
    else:
        check_df = post_entry

    if check_df.empty:
        return None, pos.to_dict()

    current_stop = get_float(pos, "current_stop", np.nan)
    entry_price = get_float(pos, "entry_price", np.nan)

    if not math.isfinite(current_stop):
        signal_atr = get_float(pos, "signal_atr", np.nan)

        if math.isfinite(signal_atr):
            if side == "LONG":
                current_stop = entry_price - signal_atr * atr_mult
            else:
                current_stop = entry_price + signal_atr * atr_mult
        else:
            return None, pos.to_dict()

    highest_high = get_float(pos, "highest_high", entry_price)
    lowest_low = get_float(pos, "lowest_low", entry_price)

    if not math.isfinite(highest_high):
        highest_high = entry_price

    if not math.isfinite(lowest_low):
        lowest_low = entry_price

    for date, row in check_df.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        ma = float(row["ma"])
        atr = float(row["atr"])

        bar_hold_days = int(len(post_entry[post_entry.index <= date]))

        if side == "LONG":
            if low <= current_stop:
                trade = make_trade_row(
                    pos,
                    exit_date=date,
                    exit_price=current_stop,
                    exit_reason="ATR_STOP",
                    bar_hold_days=bar_hold_days,
                    current_equity=current_equity,
                    final_stop=current_stop,
                )
                return trade, pos.to_dict()

            if close < ma:
                trade = make_trade_row(
                    pos,
                    exit_date=date,
                    exit_price=close,
                    exit_reason="MA_EXIT",
                    bar_hold_days=bar_hold_days,
                    current_equity=current_equity,
                    final_stop=current_stop,
                )
                return trade, pos.to_dict()

            if high > highest_high:
                highest_high = high

            if math.isfinite(atr) and atr > 0:
                trailing_stop = highest_high - atr * atr_mult
                if trailing_stop > current_stop:
                    current_stop = trailing_stop

        else:
            if high >= current_stop:
                trade = make_trade_row(
                    pos,
                    exit_date=date,
                    exit_price=current_stop,
                    exit_reason="ATR_STOP",
                    bar_hold_days=bar_hold_days,
                    current_equity=current_equity,
                    final_stop=current_stop,
                )
                return trade, pos.to_dict()

            if close > ma:
                trade = make_trade_row(
                    pos,
                    exit_date=date,
                    exit_price=close,
                    exit_reason="MA_EXIT",
                    bar_hold_days=bar_hold_days,
                    current_equity=current_equity,
                    final_stop=current_stop,
                )
                return trade, pos.to_dict()

            if low < lowest_low:
                lowest_low = low

            if math.isfinite(atr) and atr > 0:
                trailing_stop = lowest_low + atr * atr_mult
                if trailing_stop < current_stop:
                    current_stop = trailing_stop

        if bar_hold_days >= max_hold_days:
            trade = make_trade_row(
                pos,
                exit_date=date,
                exit_price=close,
                exit_reason="TIME",
                bar_hold_days=bar_hold_days,
                current_equity=current_equity,
                final_stop=current_stop,
            )
            return trade, pos.to_dict()

    updated = pos.to_dict()
    updated["current_stop"] = current_stop
    updated["highest_high"] = highest_high
    updated["lowest_low"] = lowest_low
    updated["last_checked_date"] = check_df.index[-1].strftime("%Y-%m-%d")
    updated["updated_at"] = now_str()

    return None, updated


def process_open_positions(
    positions_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    raw_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    closed_trades = []

    if positions_df.empty:
        return positions_df, trades_df, closed_trades

    keep_positions = []
    current_equity = current_equity_from_trades(trades_df)

    for _, pos in positions_df.iterrows():
        status = str(pos.get("status", ""))

        if status != "OPEN":
            keep_positions.append(pos.to_dict())
            continue

        pair = str(pos.get("pair", ""))

        if pair not in raw_data:
            keep_positions.append(pos.to_dict())
            continue

        strategy = str(pos.get("strategy", ""))

        if strategy == "V700_RSI_PULLBACK":
            trade, updated_pos = process_open_v700_position(
                pos,
                raw_df=raw_data[pair],
                current_equity=current_equity,
            )
        elif strategy == "V800_ATR_TREND":
            trade, updated_pos = process_open_v800_position(
                pos,
                raw_df=raw_data[pair],
                current_equity=current_equity,
            )
        else:
            trade, updated_pos = None, pos.to_dict()

        if trade is None:
            keep_positions.append(updated_pos)
            continue

        trades_df = append_trade(trades_df, trade)
        closed_trades.append(trade)
        current_equity = float(trade["equity_after"])

        print(
            f"[CLOSE] {trade['strategy']} {trade['pair']} {trade['side']} "
            f"exit={trade['exit_date']} reason={trade['exit_reason']} "
            f"net_return={trade['net_return']:.6f} pnl={trade['pnl']:.2f}"
        )

    if keep_positions:
        new_positions_df = pd.DataFrame(keep_positions)
    else:
        new_positions_df = pd.DataFrame(columns=POSITION_COLUMNS)

    for col in POSITION_COLUMNS:
        if col not in new_positions_df.columns:
            new_positions_df[col] = np.nan

    return new_positions_df[POSITION_COLUMNS].copy(), trades_df, closed_trades


# =========================================================
# EQUITY / REPORT
# =========================================================

def make_equity_log(trades_df: pd.DataFrame) -> pd.DataFrame:
    equity = current_equity_from_trades(trades_df)
    realized_pnl = equity - INITIAL_EQUITY
    trade_count = 0 if trades_df.empty else len(trades_df)

    return pd.DataFrame([{
        "timestamp": now_str(),
        "equity": round(float(equity), 2),
        "realized_pnl": round(float(realized_pnl), 2),
        "trade_count": int(trade_count),
    }])


def append_equity_log(equity_file: str, new_log_df: pd.DataFrame) -> pd.DataFrame:
    old = read_csv_or_empty(equity_file, EQUITY_COLUMNS)
    out = pd.concat([old, new_log_df], ignore_index=True)
    save_csv(out, equity_file, EQUITY_COLUMNS)
    return out


def make_report(
    candidates_df: pd.DataFrame,
    new_orders: list[dict],
    opened_positions: list[dict],
    closed_trades: list[dict],
    positions_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    open_positions = 0
    pending_positions = 0

    if not positions_df.empty:
        open_positions = int((positions_df["status"] == "OPEN").sum())
        pending_positions = int((positions_df["status"] == "PENDING").sum())

    equity = current_equity_from_trades(trades_df)

    return pd.DataFrame([{
        "run_datetime": now_str(),
        "candidate_rows": 0 if candidates_df.empty else len(candidates_df),
        "new_pending_orders": len(new_orders),
        "opened_positions": len(opened_positions),
        "closed_trades": len(closed_trades),
        "open_positions": open_positions,
        "pending_positions": pending_positions,
        "equity": round(float(equity), 2),
    }])


# =========================================================
# DISCORD
# =========================================================

def format_discord_message(
    report_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    new_orders: list[dict],
    opened_positions: list[dict],
    closed_trades: list[dict],
) -> str:
    report = report_df.iloc[0].to_dict()

    lines = []
    lines.append("📊 FX Paper Trade v811")
    lines.append("")
    lines.append(f"run_datetime: {report['run_datetime']}")
    lines.append(f"equity: {report['equity']}")
    lines.append(f"new_pending_orders: {report['new_pending_orders']}")
    lines.append(f"opened_positions: {report['opened_positions']}")
    lines.append(f"closed_trades: {report['closed_trades']}")
    lines.append(f"open_positions: {report['open_positions']}")
    lines.append(f"pending_positions: {report['pending_positions']}")
    lines.append("")

    if new_orders:
        lines.append("NEW PENDING:")
        for x in new_orders:
            lines.append(f"- {x['strategy']} {x['pair']} {x['side']} signal={x['signal_date']}")
        lines.append("")

    if opened_positions:
        lines.append("OPENED:")
        for x in opened_positions:
            lines.append(f"- {x['strategy']} {x['pair']} {x['side']} entry={x['entry_date']} price={x['entry_price']}")
        lines.append("")

    if closed_trades:
        lines.append("CLOSED:")
        for x in closed_trades:
            lines.append(
                f"- {x['strategy']} {x['pair']} {x['side']} "
                f"exit={x['exit_date']} {x['exit_reason']} pnl={x['pnl']:.2f}"
            )
        lines.append("")

    if not positions_df.empty:
        active = positions_df[positions_df["status"].isin(["PENDING", "OPEN"])].copy()
    else:
        active = pd.DataFrame()

    if not active.empty:
        lines.append("ACTIVE:")
        for _, x in active.iterrows():
            status = x.get("status", "")
            strategy = x.get("strategy", "")
            pair = x.get("pair", "")
            side = x.get("side", "")
            entry_date = x.get("entry_date", "")
            entry_price = x.get("entry_price", "")
            current_stop = x.get("current_stop", "")
            tp_price = x.get("tp_price", "")
            sl_price = x.get("sl_price", "")

            if status == "PENDING":
                lines.append(f"- PENDING {strategy} {pair} {side} signal={x.get('signal_date', '')}")
            else:
                lines.append(
                    f"- OPEN {strategy} {pair} {side} "
                    f"entry={entry_date} price={entry_price} "
                    f"stop={current_stop} tp={tp_price} sl={sl_price}"
                )
    else:
        lines.append("ACTIVE: none")

    message = "\n".join(lines)

    if len(message) > 1900:
        message = message[:1900] + "\n...(truncated)"

    return message


def send_discord_message(message: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        print("[DISCORD] skipped: DISCORD_WEBHOOK_URL is not set")
        return

    payload = json.dumps({"content": message}).encode("utf-8")

    req = urllib.request.Request(
        DISCORD_WEBHOOK_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "fx-v811-paper-trade",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as res:
            print(f"[DISCORD] status={res.status}")
    except Exception as e:
        print(f"[DISCORD] failed: {e}")


# =========================================================
# PRINT HELPERS
# =========================================================

def print_df(title: str, df: pd.DataFrame) -> None:
    print("")
    print("========================================")
    print(f" {title}")
    print("========================================")

    if df.empty:
        print("(none)")
    else:
        print(df.to_string(index=False))


# =========================================================
# MAIN
# =========================================================

def run():
    print("========================================")
    print(" FX PAPER TRADE v811")
    print("========================================")
    print(f"START           : {START}")
    print(f"END             : {END}")
    print(f"CANDIDATES_FILE : {CANDIDATES_FILE}")
    print(f"POSITIONS_FILE  : {POSITIONS_FILE}")
    print(f"TRADES_FILE     : {TRADES_FILE}")
    print("========================================")

    candidates_df = load_candidates()
    positions_df = read_csv_or_empty(POSITIONS_FILE, POSITION_COLUMNS)
    trades_df = read_csv_or_empty(TRADES_FILE, TRADE_COLUMNS)

    if not positions_df.empty:
        for col in ["signal_date", "entry_date", "last_checked_date"]:
            positions_df[col] = positions_df[col].apply(to_date_str)

    pairs = set()

    if not candidates_df.empty and "pair" in candidates_df.columns:
        pairs.update(candidates_df["pair"].dropna().astype(str).tolist())

    if not positions_df.empty:
        pairs.update(positions_df["pair"].dropna().astype(str).tolist())

    raw_data = {}

    for pair in sorted(pairs):
        df = fetch_pair(pair)

        if df.empty:
            continue

        raw_data[pair] = df
        print(f"[DATA] {pair}: rows={len(df)}, latest={df.index[-1].date()}")

    # 1. 既存PENDINGを、次足Openが取れたものからOPENへ
    positions_df, opened_positions = process_pending_entries(positions_df, raw_data)

    # 2. OPENポジションを日次チェック
    positions_df, trades_df, closed_trades = process_open_positions(
        positions_df,
        trades_df,
        raw_data,
    )

    # 3. v810候補を新規PENDING登録
    positions_df, new_orders = add_pending_orders(positions_df, candidates_df)

    # 4. 保存
    save_csv(positions_df, POSITIONS_FILE, POSITION_COLUMNS)
    save_csv(trades_df, TRADES_FILE, TRADE_COLUMNS)

    equity_log_df = make_equity_log(trades_df)
    append_equity_log(EQUITY_FILE, equity_log_df)

    report_df = make_report(
        candidates_df=candidates_df,
        new_orders=new_orders,
        opened_positions=opened_positions,
        closed_trades=closed_trades,
        positions_df=positions_df,
        trades_df=trades_df,
    )
    save_csv(report_df, REPORT_FILE, REPORT_COLUMNS)

    # 5. 表示
    print_df("DAILY REPORT", report_df)

    active_positions = positions_df[positions_df["status"].isin(["PENDING", "OPEN"])].copy()
    print_df("ACTIVE POSITIONS", active_positions)

    if closed_trades:
        closed_df = pd.DataFrame(closed_trades)
    else:
        closed_df = pd.DataFrame()
    print_df("CLOSED TRADES THIS RUN", closed_df)

    print("")
    print("========================================")
    print(" SAVED")
    print("========================================")
    print(f"- {POSITIONS_FILE}")
    print(f"- {TRADES_FILE}")
    print(f"- {EQUITY_FILE}")
    print(f"- {REPORT_FILE}")

    message = format_discord_message(
        report_df=report_df,
        positions_df=positions_df,
        new_orders=new_orders,
        opened_positions=opened_positions,
        closed_trades=closed_trades,
    )
    send_discord_message(message)

    print("")
    print("Done.")


if __name__ == "__main__":
    run()
