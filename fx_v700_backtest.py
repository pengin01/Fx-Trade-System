import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# PARAMETERS
# =========================================================

PAIRS = [
    "USDJPY=X",
    "EURUSD=X",
    "GBPUSD=X",
    "AUDUSD=X",
    "EURJPY=X",
    "GBPJPY=X",
]

START = "2018-01-01"
END = None  # Noneなら最新まで

INITIAL_EQUITY = 100_000

RSI_DAYS = 14
MA_DAYS = 25
LOOKBACK_HIGH_LOW = 20

# 株版のpullback_pctに相当
# FXは株より変動率が小さいので、最初は0.3%〜1.0%くらいで見る
PULLBACK_PCT = 0.005

# LONG条件: RSIが低め
RSI_LONG_MAX = 45

# SHORT条件: RSIが高め
RSI_SHORT_MIN = 55

TP_PCT = 0.008  # 利確 0.8%
SL_PCT = 0.006  # 損切 0.6%
HOLD_DAYS = 4

# 1トレードあたりの資金投入割合
# まずは控えめに30%
POSITION_FRACTION = 0.30

# 簡易スプレッド想定
# JPYペアは 1pip = 0.01
# それ以外は 1pip = 0.0001
SPREAD_PIPS = {
    "USDJPY=X": 0.3,
    "EURUSD=X": 0.2,
    "GBPUSD=X": 0.6,
    "AUDUSD=X": 0.5,
    "EURJPY=X": 0.5,
    "GBPJPY=X": 0.9,
}

DEFAULT_SPREAD_PIPS = 0.5

OUT_TRADES = "fx_trades_v700.csv"
OUT_SUMMARY = "fx_summary_v700.csv"
OUT_CANDIDATES = "fx_candidates_v700.csv"
OUT_EQUITY = "fx_equity_curve_v700.csv"


# =========================================================
# INDICATORS
# =========================================================


def calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()

    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def pip_size(pair: str) -> float:
    name = pair.replace("=X", "")
    if "JPY" in name:
        return 0.01
    return 0.0001


def spread_cost_pct(pair: str, price: float) -> float:
    spread_pips = SPREAD_PIPS.get(pair, DEFAULT_SPREAD_PIPS)
    spread_price = spread_pips * pip_size(pair)
    return spread_price / price


# =========================================================
# DATA
# =========================================================


def fetch_pair(pair: str) -> pd.DataFrame:
    print(f"[FETCH] {pair}")

    df = yf.download(
        pair,
        start=START,
        end=END,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        print(f"[WARN] no data: {pair}")
        return pd.DataFrame()

    # yfinanceの戻りがMultiIndexになる場合への対応
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    for col in required:
        if col not in df.columns:
            print(f"[WARN] missing column {col}: {pair}")
            return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna()

    # timezoneが付く場合の保険
    df.index = pd.to_datetime(df.index).tz_localize(None)

    return df


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ma"] = df["Close"].rolling(MA_DAYS).mean()
    df["rsi"] = calc_rsi(df["Close"], RSI_DAYS)

    df["prev_high"] = df["Close"].rolling(LOOKBACK_HIGH_LOW).max().shift(1)
    df["prev_low"] = df["Close"].rolling(LOOKBACK_HIGH_LOW).min().shift(1)

    # LONG:
    # 上昇基調: Close > MA
    # 押し目: 直近高値から一定以上下落
    # RSI: やや売られすぎ
    df["long_signal"] = (
        (df["Close"] > df["ma"])
        & (df["Close"] <= df["prev_high"] * (1 - PULLBACK_PCT))
        & (df["rsi"] <= RSI_LONG_MAX)
    )

    # SHORT:
    # 下降基調: Close < MA
    # 戻り売り: 直近安値から一定以上反発
    # RSI: やや買われすぎ
    df["short_signal"] = (
        (df["Close"] < df["ma"])
        & (df["Close"] >= df["prev_low"] * (1 + PULLBACK_PCT))
        & (df["rsi"] >= RSI_SHORT_MIN)
    )

    return df


# =========================================================
# BACKTEST
# =========================================================


def backtest_pair(pair: str, df: pd.DataFrame) -> list[dict]:
    trades = []

    if df.empty:
        return trades

    df = add_signals(df)
    df = df.dropna().copy()

    if len(df) < HOLD_DAYS + 10:
        return trades

    i = 1
    n = len(df)

    while i < n - 1:
        signal_row = df.iloc[i - 1]
        entry_row = df.iloc[i]

        side = None

        if bool(signal_row["long_signal"]):
            side = "LONG"
        elif bool(signal_row["short_signal"]):
            side = "SHORT"

        if side is None:
            i += 1
            continue

        entry_date = df.index[i]
        entry_price = float(entry_row["Open"])

        if not math.isfinite(entry_price) or entry_price <= 0:
            i += 1
            continue

        if side == "LONG":
            tp_price = entry_price * (1 + TP_PCT)
            sl_price = entry_price * (1 - SL_PCT)
        else:
            tp_price = entry_price * (1 - TP_PCT)
            sl_price = entry_price * (1 + SL_PCT)

        exit_date = None
        exit_price = None
        exit_reason = None
        exit_index = None

        last_j = min(i + HOLD_DAYS - 1, n - 1)

        for j in range(i, last_j + 1):
            row = df.iloc[j]
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])

            # 同日にTP/SL両方到達した場合は保守的にSL優先
            if side == "LONG":
                hit_sl = low <= sl_price
                hit_tp = high >= tp_price

                if hit_sl:
                    exit_date = df.index[j]
                    exit_price = sl_price
                    exit_reason = "SL"
                    exit_index = j
                    break

                if hit_tp:
                    exit_date = df.index[j]
                    exit_price = tp_price
                    exit_reason = "TP"
                    exit_index = j
                    break

            else:
                hit_sl = high >= sl_price
                hit_tp = low <= tp_price

                if hit_sl:
                    exit_date = df.index[j]
                    exit_price = sl_price
                    exit_reason = "SL"
                    exit_index = j
                    break

                if hit_tp:
                    exit_date = df.index[j]
                    exit_price = tp_price
                    exit_reason = "TP"
                    exit_index = j
                    break

            # 最終保有日にCloseで手仕舞い
            if j == last_j:
                exit_date = df.index[j]
                exit_price = close
                exit_reason = "TIME"
                exit_index = j
                break

        if exit_date is None:
            i += 1
            continue

        if side == "LONG":
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price

        # 往復スプレッド控除
        cost_pct = spread_cost_pct(pair, entry_price) * 2
        net_return = gross_return - cost_pct

        trades.append(
            {
                "pair": pair,
                "side": side,
                "signal_date": signal_row.name.date(),
                "entry_date": entry_date.date(),
                "exit_date": exit_date.date(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "exit_reason": exit_reason,
                "hold_days": (exit_date - entry_date).days,
                "gross_return": gross_return,
                "spread_cost_pct": cost_pct,
                "net_return": net_return,
                "win": net_return > 0,
                "signal_close": float(signal_row["Close"]),
                "signal_ma": float(signal_row["ma"]),
                "signal_rsi": float(signal_row["rsi"]),
            }
        )

        # 同一ペアはポジション重複なし
        i = exit_index + 1

    return trades


def make_equity_curve(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "equity",
                "peak",
                "drawdown",
            ]
        )

    trades_df = trades_df.sort_values(["exit_date", "pair", "side"]).copy()

    equity = INITIAL_EQUITY
    rows = []

    for _, tr in trades_df.iterrows():
        pnl = equity * POSITION_FRACTION * float(tr["net_return"])
        equity += pnl

        rows.append(
            {
                "date": tr["exit_date"],
                "pair": tr["pair"],
                "side": tr["side"],
                "net_return": tr["net_return"],
                "pnl": pnl,
                "equity": equity,
            }
        )

    eq = pd.DataFrame(rows)
    eq["peak"] = eq["equity"].cummax()
    eq["drawdown"] = eq["equity"] / eq["peak"] - 1

    return eq


def make_summary(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    summary = trades_df.groupby(["pair", "side"], as_index=False).agg(
        trades=("net_return", "count"),
        win_rate=("win", "mean"),
        avg_return=("net_return", "mean"),
        total_return_sum=("net_return", "sum"),
        best_trade=("net_return", "max"),
        worst_trade=("net_return", "min"),
    )

    summary["win_rate"] = summary["win_rate"].round(4)
    summary["avg_return"] = summary["avg_return"].round(6)
    summary["total_return_sum"] = summary["total_return_sum"].round(6)
    summary["best_trade"] = summary["best_trade"].round(6)
    summary["worst_trade"] = summary["worst_trade"].round(6)

    if not equity_df.empty:
        final_equity = float(equity_df["equity"].iloc[-1])
        max_dd = float(equity_df["drawdown"].min())
    else:
        final_equity = INITIAL_EQUITY
        max_dd = 0.0

    total_row = pd.DataFrame(
        [
            {
                "pair": "ALL",
                "side": "ALL",
                "trades": len(trades_df),
                "win_rate": round(float(trades_df["win"].mean()), 4),
                "avg_return": round(float(trades_df["net_return"].mean()), 6),
                "total_return_sum": round(float(trades_df["net_return"].sum()), 6),
                "best_trade": round(float(trades_df["net_return"].max()), 6),
                "worst_trade": round(float(trades_df["net_return"].min()), 6),
                "final_equity": round(final_equity, 2),
                "max_drawdown": round(max_dd, 6),
            }
        ]
    )

    summary["final_equity"] = ""
    summary["max_drawdown"] = ""

    summary = pd.concat([summary, total_row], ignore_index=True)

    return summary


def make_candidates(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for pair, df in all_data.items():
        if df.empty:
            continue

        sig = add_signals(df).dropna()

        if sig.empty:
            continue

        latest = sig.iloc[-1]
        latest_date = sig.index[-1]

        side = None

        if bool(latest["long_signal"]):
            side = "LONG"
        elif bool(latest["short_signal"]):
            side = "SHORT"

        if side is None:
            continue

        rows.append(
            {
                "signal_date": latest_date.date(),
                "pair": pair,
                "side": side,
                "close": float(latest["Close"]),
                "ma": float(latest["ma"]),
                "rsi": float(latest["rsi"]),
                "next_action": "next open entry candidate",
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================


def run():
    print("========================================")
    print(" FX BACKTEST v700")
    print("========================================")
    print(f"START             : {START}")
    print(f"END               : {END}")
    print(f"PAIRS             : {PAIRS}")
    print(f"INITIAL_EQUITY    : {INITIAL_EQUITY}")
    print(f"POSITION_FRACTION : {POSITION_FRACTION}")
    print(f"RSI_DAYS          : {RSI_DAYS}")
    print(f"MA_DAYS           : {MA_DAYS}")
    print(f"PULLBACK_PCT      : {PULLBACK_PCT}")
    print(f"TP_PCT            : {TP_PCT}")
    print(f"SL_PCT            : {SL_PCT}")
    print(f"HOLD_DAYS         : {HOLD_DAYS}")
    print("========================================")

    all_data = {}
    all_trades = []

    for pair in PAIRS:
        df = fetch_pair(pair)
        all_data[pair] = df

        trades = backtest_pair(pair, df)
        all_trades.extend(trades)

        print(f"[RESULT] {pair}: trades={len(trades)}")

    trades_df = pd.DataFrame(all_trades)

    if trades_df.empty:
        print("")
        print("No trades found.")
        trades_df.to_csv(OUT_TRADES, index=False)
        pd.DataFrame().to_csv(OUT_SUMMARY, index=False)
        pd.DataFrame().to_csv(OUT_EQUITY, index=False)

        candidates_df = make_candidates(all_data)
        candidates_df.to_csv(OUT_CANDIDATES, index=False)

        print(f"Saved: {OUT_TRADES}, {OUT_SUMMARY}, {OUT_EQUITY}, {OUT_CANDIDATES}")
        return

    trades_df = trades_df.sort_values(["entry_date", "pair", "side"]).copy()

    equity_df = make_equity_curve(trades_df)
    summary_df = make_summary(trades_df, equity_df)
    candidates_df = make_candidates(all_data)

    trades_df.to_csv(OUT_TRADES, index=False)
    equity_df.to_csv(OUT_EQUITY, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    candidates_df.to_csv(OUT_CANDIDATES, index=False)

    print("")
    print("========================================")
    print(" SUMMARY")
    print("========================================")

    total = summary_df[summary_df["pair"] == "ALL"]

    if not total.empty:
        row = total.iloc[0]
        print(f"trades       : {row['trades']}")
        print(f"win_rate     : {row['win_rate']}")
        print(f"avg_return   : {row['avg_return']}")
        print(f"final_equity : {row['final_equity']}")
        print(f"max_drawdown : {row['max_drawdown']}")

    print("")
    print("========================================")
    print(" CURRENT CANDIDATES")
    print("========================================")

    if candidates_df.empty:
        print("(none)")
    else:
        print(candidates_df.to_string(index=False))

    print("")
    print("Saved:")
    print(f"- {OUT_TRADES}")
    print(f"- {OUT_SUMMARY}")
    print(f"- {OUT_EQUITY}")
    print(f"- {OUT_CANDIDATES}")


if __name__ == "__main__":
    run()
