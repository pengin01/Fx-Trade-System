import math
import time
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# FX TREND ATR BACKTEST v800
# =========================================================
#
# 目的:
# - v700系のRSI押し目型から、トレンドフォロー型へ切り替える
# - 20日/40日などの高値・安値ブレイクでエントリー
# - MAでトレンド方向を判定
# - ATRストップ / ATRトレーリングで損切り・利益伸ばし
# - 固定TPは使わない
#
# 実行:
#   python .\fx_v800_trend_atr_backtest.py
#
# 出力:
# - fx_v800_param_results.csv
# - fx_v800_best_trades.csv
# - fx_v800_best_equity.csv
# - fx_v800_pair_side_summary.csv
# - fx_v800_candidates.csv
#
# =========================================================


# =========================================================
# BASIC SETTINGS
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
END = None

INITIAL_EQUITY = 100_000
POSITION_FRACTION = 0.30

MIN_TRADES = 30


# =========================================================
# OPTIONAL PAIR SIDE FILTER
# =========================================================
#
# まずv800では全方向を試します。
# 必要なら後で禁止方向を追加できます。
#
# 例:
# BLOCKED_PAIR_SIDES = {
#     ("GBPJPY=X", "LONG"),
#     ("GBPJPY=X", "SHORT"),
# }
#
# =========================================================

BLOCKED_PAIR_SIDES = {
    ("USDJPY=X", "SHORT"),
    ("EURUSD=X", "LONG"),
    ("EURUSD=X", "SHORT"),
    ("GBPUSD=X", "LONG"),
    ("AUDUSD=X", "LONG"),
    ("AUDUSD=X", "SHORT"),
    ("EURJPY=X", "LONG"),
    ("EURJPY=X", "SHORT"),
    ("GBPJPY=X", "LONG"),
    ("GBPJPY=X", "SHORT"),
}


# =========================================================
# OUTPUT FILES
# =========================================================

OUT_RESULTS = "fx_v800_param_results.csv"
OUT_BEST_TRADES = "fx_v800_best_trades.csv"
OUT_BEST_EQUITY = "fx_v800_best_equity.csv"
OUT_PAIR_SIDE_SUMMARY = "fx_v800_pair_side_summary.csv"
OUT_CANDIDATES = "fx_v800_candidates.csv"


# =========================================================
# PARAM GRID
# =========================================================
#
# breakout_days:
#   何日高値/安値ブレイクを見るか
#
# ma_days:
#   トレンド判定用MA
#
# atr_days:
#   ATR計算期間
#
# atr_mult:
#   初期ストップ / トレーリングストップ幅
#
# max_hold_days:
#   最大保有日数
#
# =========================================================

PARAM_GRID = {
    "breakout_days": [20, 40, 60],
    "ma_days": [50, 100, 150, 200],
    "atr_days": [14],
    "atr_mult": [1.5, 2.0, 2.5, 3.0],
    "max_hold_days": [20, 40, 60, 90],
}


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
# DATA STRUCTURES
# =========================================================


@dataclass(frozen=True)
class Params:
    breakout_days: int
    ma_days: int
    atr_days: int
    atr_mult: float
    max_hold_days: int


@dataclass
class PairData:
    pair: str
    dates: pd.DatetimeIndex
    date_values: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


@dataclass
class FeatureData:
    pair: str
    dates: pd.DatetimeIndex
    date_values: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    ma: np.ndarray
    atr: np.ndarray
    prev_high: np.ndarray
    prev_low: np.ndarray


# =========================================================
# UTILS
# =========================================================


def pip_size(pair: str) -> float:
    name = pair.replace("=X", "")

    if "JPY" in name:
        return 0.01

    return 0.0001


def spread_cost_pct(pair: str, price: float) -> float:
    spread_pips = SPREAD_PIPS.get(pair, DEFAULT_SPREAD_PIPS)
    spread_price = spread_pips * pip_size(pair)

    return spread_price / price


def is_blocked(pair: str, side: str) -> bool:
    return (pair, side) in BLOCKED_PAIR_SIDES


def to_float_array(series: pd.Series) -> np.ndarray:
    return series.astype(float).to_numpy(dtype=np.float64)


# =========================================================
# INDICATORS
# =========================================================


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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    for col in required:
        if col not in df.columns:
            print(f"[WARN] missing column {col}: {pair}")
            return pd.DataFrame()

    df = df[required].copy()
    df = df.dropna()

    df.index = pd.to_datetime(df.index).tz_localize(None)

    return df


def to_pair_data(pair: str, df: pd.DataFrame) -> PairData:
    dates = pd.DatetimeIndex(pd.to_datetime(df.index))
    date_values = dates.astype("int64").to_numpy()

    return PairData(
        pair=pair,
        dates=dates,
        date_values=date_values,
        open=to_float_array(df["Open"]),
        high=to_float_array(df["High"]),
        low=to_float_array(df["Low"]),
        close=to_float_array(df["Close"]),
    )


def fetch_all_data() -> dict[str, PairData]:
    all_data = {}

    for pair in PAIRS:
        df = fetch_pair(pair)

        if df.empty:
            continue

        data = to_pair_data(pair, df)
        all_data[pair] = data

        print(f"[DATA] {pair}: rows={len(data.close)}")

    return all_data


def pair_data_to_df(data: PairData) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": data.open,
            "High": data.high,
            "Low": data.low,
            "Close": data.close,
        },
        index=data.dates,
    )


def prepare_features(data: PairData, params: Params) -> FeatureData:
    df = pair_data_to_df(data)

    df["ma"] = df["Close"].rolling(params.ma_days).mean()
    df["atr"] = calc_atr(df, params.atr_days)

    # signal日に判定できるよう、過去N日高値/安値はshift(1)
    df["prev_high"] = df["High"].rolling(params.breakout_days).max().shift(1)
    df["prev_low"] = df["Low"].rolling(params.breakout_days).min().shift(1)

    df = df.dropna().copy()

    dates = pd.DatetimeIndex(pd.to_datetime(df.index))
    date_values = dates.astype("int64").to_numpy()

    return FeatureData(
        pair=data.pair,
        dates=dates,
        date_values=date_values,
        open=to_float_array(df["Open"]),
        high=to_float_array(df["High"]),
        low=to_float_array(df["Low"]),
        close=to_float_array(df["Close"]),
        ma=to_float_array(df["ma"]),
        atr=to_float_array(df["atr"]),
        prev_high=to_float_array(df["prev_high"]),
        prev_low=to_float_array(df["prev_low"]),
    )


# =========================================================
# PARAMS
# =========================================================


def iter_params() -> list[Params]:
    keys = [
        "breakout_days",
        "ma_days",
        "atr_days",
        "atr_mult",
        "max_hold_days",
    ]

    values = [PARAM_GRID[k] for k in keys]

    params_list = []

    for combo in product(*values):
        d = dict(zip(keys, combo))
        params_list.append(Params(**d))

    return params_list


# =========================================================
# SIGNAL
# =========================================================


def make_signal_arrays(data: FeatureData) -> tuple[np.ndarray, np.ndarray]:
    close = data.close
    ma = data.ma
    prev_high = data.prev_high
    prev_low = data.prev_low

    # 上昇トレンド中の高値ブレイク
    long_signal = (close > ma) & (close > prev_high)

    # 下降トレンド中の安値ブレイク
    short_signal = (close < ma) & (close < prev_low)

    return long_signal, short_signal


# =========================================================
# BACKTEST
# =========================================================


def backtest_pair_min(
    raw_data: PairData, params: Params
) -> list[tuple[int, float, int]]:
    data = prepare_features(raw_data, params)

    n = len(data.close)

    if n < params.max_hold_days + 10:
        return []

    long_signal, short_signal = make_signal_arrays(data)
    signal_indices = np.flatnonzero(long_signal | short_signal)

    if len(signal_indices) == 0:
        return []

    trades = []
    last_exit_index = -1

    o = data.open
    h = data.high
    l = data.low
    c = data.close
    ma = data.ma
    atr = data.atr

    for signal_index in signal_indices:
        entry_index = signal_index + 1

        if entry_index >= n:
            continue

        if entry_index <= last_exit_index:
            continue

        if bool(long_signal[signal_index]):
            side_name = "LONG"
            side_int = 1
        elif bool(short_signal[signal_index]):
            side_name = "SHORT"
            side_int = -1
        else:
            continue

        if is_blocked(data.pair, side_name):
            continue

        entry_price = o[entry_index]
        entry_atr = atr[signal_index]

        if not math.isfinite(entry_price) or entry_price <= 0:
            continue

        if not math.isfinite(entry_atr) or entry_atr <= 0:
            continue

        last_j = min(entry_index + params.max_hold_days - 1, n - 1)

        exit_index = -1
        exit_price = np.nan

        if side_int == 1:
            stop_price = entry_price - entry_atr * params.atr_mult
            highest_high = entry_price

            for j in range(entry_index, last_j + 1):
                # まず既存stopに触れたか確認
                if l[j] <= stop_price:
                    exit_index = j
                    exit_price = stop_price
                    break

                # MA割れで手仕舞い
                if c[j] < ma[j]:
                    exit_index = j
                    exit_price = c[j]
                    break

                # 当日の値動きを使って翌日以降のトレーリングstopを更新
                if h[j] > highest_high:
                    highest_high = h[j]

                if math.isfinite(atr[j]) and atr[j] > 0:
                    trailing_stop = highest_high - atr[j] * params.atr_mult
                    if trailing_stop > stop_price:
                        stop_price = trailing_stop

                # 最大保有日数で手仕舞い
                if j == last_j:
                    exit_index = j
                    exit_price = c[j]
                    break

        else:
            stop_price = entry_price + entry_atr * params.atr_mult
            lowest_low = entry_price

            for j in range(entry_index, last_j + 1):
                # まず既存stopに触れたか確認
                if h[j] >= stop_price:
                    exit_index = j
                    exit_price = stop_price
                    break

                # MA上抜けで手仕舞い
                if c[j] > ma[j]:
                    exit_index = j
                    exit_price = c[j]
                    break

                # 当日の値動きを使って翌日以降のトレーリングstopを更新
                if l[j] < lowest_low:
                    lowest_low = l[j]

                if math.isfinite(atr[j]) and atr[j] > 0:
                    trailing_stop = lowest_low + atr[j] * params.atr_mult
                    if trailing_stop < stop_price:
                        stop_price = trailing_stop

                # 最大保有日数で手仕舞い
                if j == last_j:
                    exit_index = j
                    exit_price = c[j]
                    break

        if exit_index < 0 or not math.isfinite(exit_price):
            continue

        if side_int == 1:
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price

        cost_pct = spread_cost_pct(data.pair, entry_price) * 2
        net_return = gross_return - cost_pct

        trades.append(
            (
                int(data.date_values[exit_index]),
                float(net_return),
                side_int,
            )
        )

        last_exit_index = exit_index

    return trades


def backtest_pair_full(raw_data: PairData, params: Params) -> list[dict]:
    data = prepare_features(raw_data, params)

    n = len(data.close)

    if n < params.max_hold_days + 10:
        return []

    long_signal, short_signal = make_signal_arrays(data)
    signal_indices = np.flatnonzero(long_signal | short_signal)

    if len(signal_indices) == 0:
        return []

    trades = []
    last_exit_index = -1

    o = data.open
    h = data.high
    l = data.low
    c = data.close
    ma = data.ma
    atr = data.atr

    for signal_index in signal_indices:
        entry_index = signal_index + 1

        if entry_index >= n:
            continue

        if entry_index <= last_exit_index:
            continue

        if bool(long_signal[signal_index]):
            side = "LONG"
            side_int = 1
        elif bool(short_signal[signal_index]):
            side = "SHORT"
            side_int = -1
        else:
            continue

        if is_blocked(data.pair, side):
            continue

        entry_price = o[entry_index]
        entry_atr = atr[signal_index]

        if not math.isfinite(entry_price) or entry_price <= 0:
            continue

        if not math.isfinite(entry_atr) or entry_atr <= 0:
            continue

        last_j = min(entry_index + params.max_hold_days - 1, n - 1)

        exit_index = -1
        exit_price = np.nan
        exit_reason = ""

        initial_stop = None
        final_stop = None

        if side_int == 1:
            stop_price = entry_price - entry_atr * params.atr_mult
            initial_stop = stop_price
            highest_high = entry_price

            for j in range(entry_index, last_j + 1):
                if l[j] <= stop_price:
                    exit_index = j
                    exit_price = stop_price
                    exit_reason = "ATR_STOP"
                    break

                if c[j] < ma[j]:
                    exit_index = j
                    exit_price = c[j]
                    exit_reason = "MA_EXIT"
                    break

                if h[j] > highest_high:
                    highest_high = h[j]

                if math.isfinite(atr[j]) and atr[j] > 0:
                    trailing_stop = highest_high - atr[j] * params.atr_mult
                    if trailing_stop > stop_price:
                        stop_price = trailing_stop

                if j == last_j:
                    exit_index = j
                    exit_price = c[j]
                    exit_reason = "TIME"
                    break

            final_stop = stop_price

        else:
            stop_price = entry_price + entry_atr * params.atr_mult
            initial_stop = stop_price
            lowest_low = entry_price

            for j in range(entry_index, last_j + 1):
                if h[j] >= stop_price:
                    exit_index = j
                    exit_price = stop_price
                    exit_reason = "ATR_STOP"
                    break

                if c[j] > ma[j]:
                    exit_index = j
                    exit_price = c[j]
                    exit_reason = "MA_EXIT"
                    break

                if l[j] < lowest_low:
                    lowest_low = l[j]

                if math.isfinite(atr[j]) and atr[j] > 0:
                    trailing_stop = lowest_low + atr[j] * params.atr_mult
                    if trailing_stop < stop_price:
                        stop_price = trailing_stop

                if j == last_j:
                    exit_index = j
                    exit_price = c[j]
                    exit_reason = "TIME"
                    break

            final_stop = stop_price

        if exit_index < 0 or not math.isfinite(exit_price):
            continue

        if side_int == 1:
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price

        cost_pct = spread_cost_pct(data.pair, entry_price) * 2
        net_return = gross_return - cost_pct

        row = {
            "pair": data.pair,
            "side": side,
            "signal_date": data.dates[signal_index].date(),
            "entry_date": data.dates[entry_index].date(),
            "exit_date": data.dates[exit_index].date(),
            "exit_time": int(data.date_values[exit_index]),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "initial_stop": float(initial_stop),
            "final_stop": float(final_stop),
            "exit_reason": exit_reason,
            "bar_hold_days": int(exit_index - entry_index + 1),
            "calendar_hold_days": int(
                (data.dates[exit_index] - data.dates[entry_index]).days
            ),
            "gross_return": float(gross_return),
            "spread_cost_pct": float(cost_pct),
            "net_return": float(net_return),
            "win": bool(net_return > 0),
            "signal_close": float(c[signal_index]),
            "signal_ma": float(ma[signal_index]),
            "signal_atr": float(atr[signal_index]),
            "signal_prev_high": float(data.prev_high[signal_index]),
            "signal_prev_low": float(data.prev_low[signal_index]),
            **asdict(params),
        }

        trades.append(row)
        last_exit_index = exit_index

    return trades


def backtest_all_min(
    all_data: dict[str, PairData], params: Params
) -> list[tuple[int, float, int]]:
    trades = []

    for _, data in all_data.items():
        trades.extend(backtest_pair_min(data, params))

    return trades


def backtest_all_full(all_data: dict[str, PairData], params: Params) -> pd.DataFrame:
    rows = []

    for _, data in all_data.items():
        rows.extend(backtest_pair_full(data, params))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["entry_date", "pair", "side"]).reset_index(drop=True)

    return df


# =========================================================
# METRICS
# =========================================================


def calc_profit_factor(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0

    wins = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()

    if losses == 0:
        if wins > 0:
            return np.inf
        return 0.0

    return float(wins / abs(losses))


def make_equity_curve_from_min(trades: list[tuple[int, float, int]]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "exit_time",
                "net_return",
                "side_int",
                "pnl",
                "equity",
                "peak",
                "drawdown",
            ]
        )

    arr = np.array(trades, dtype=np.float64)

    exit_times = arr[:, 0]
    returns = arr[:, 1]
    sides = arr[:, 2]

    order = np.argsort(exit_times)

    equity = INITIAL_EQUITY
    peak = INITIAL_EQUITY
    rows = []

    for idx in order:
        net_return = float(returns[idx])
        side_int = int(sides[idx])

        pnl = equity * POSITION_FRACTION * net_return
        equity += pnl

        if equity > peak:
            peak = equity

        drawdown = equity / peak - 1

        rows.append(
            {
                "exit_time": int(exit_times[idx]),
                "net_return": net_return,
                "side_int": side_int,
                "pnl": pnl,
                "equity": equity,
                "peak": peak,
                "drawdown": drawdown,
            }
        )

    return pd.DataFrame(rows)


def make_equity_curve_from_full(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "pair",
                "side",
                "net_return",
                "pnl",
                "equity",
                "peak",
                "drawdown",
            ]
        )

    df = trades_df.sort_values(["exit_time", "pair", "side"]).copy()

    equity = INITIAL_EQUITY
    peak = INITIAL_EQUITY
    rows = []

    for _, tr in df.iterrows():
        net_return = float(tr["net_return"])
        pnl = equity * POSITION_FRACTION * net_return
        equity += pnl

        if equity > peak:
            peak = equity

        drawdown = equity / peak - 1

        rows.append(
            {
                "date": tr["exit_date"],
                "pair": tr["pair"],
                "side": tr["side"],
                "net_return": net_return,
                "pnl": pnl,
                "equity": equity,
                "peak": peak,
                "drawdown": drawdown,
            }
        )

    return pd.DataFrame(rows)


def summarize_min(
    params: Params, trades: list[tuple[int, float, int]], run_id: int
) -> dict:
    base = {
        "run_id": run_id,
        **asdict(params),
        "trades": 0,
        "long_trades": 0,
        "short_trades": 0,
        "win_rate": np.nan,
        "avg_return": np.nan,
        "median_return": np.nan,
        "total_return_sum": 0.0,
        "profit_factor": 0.0,
        "best_trade": np.nan,
        "worst_trade": np.nan,
        "final_equity": INITIAL_EQUITY,
        "total_equity_return": 0.0,
        "max_drawdown": 0.0,
        "score": -999999.0,
    }

    if not trades:
        return base

    arr = np.array(trades, dtype=np.float64)

    returns = arr[:, 1]
    sides = arr[:, 2]

    trade_count = len(returns)

    eq = make_equity_curve_from_min(trades)

    if eq.empty:
        final_equity = INITIAL_EQUITY
        total_equity_return = 0.0
        max_drawdown = 0.0
    else:
        final_equity = float(eq["equity"].iloc[-1])
        total_equity_return = final_equity / INITIAL_EQUITY - 1
        max_drawdown = float(eq["drawdown"].min())

    win_rate = float((returns > 0).mean())
    avg_return = float(returns.mean())
    median_return = float(np.median(returns))
    total_return_sum = float(returns.sum())
    profit_factor = calc_profit_factor(returns)
    best_trade = float(returns.max())
    worst_trade = float(returns.min())

    # score:
    # - 収益率を重視
    # - DDをペナルティ
    # - 平均リターンも少し加点
    # - tradesが少なすぎるものは落とす
    if trade_count < MIN_TRADES:
        score = -999999.0 + trade_count
    else:
        score = total_equity_return - abs(max_drawdown) * 0.6 + avg_return * 8

    base.update(
        {
            "trades": int(trade_count),
            "long_trades": int((sides == 1).sum()),
            "short_trades": int((sides == -1).sum()),
            "win_rate": round(win_rate, 6),
            "avg_return": round(avg_return, 8),
            "median_return": round(median_return, 8),
            "total_return_sum": round(total_return_sum, 8),
            "profit_factor": (
                round(float(profit_factor), 6)
                if math.isfinite(profit_factor)
                else np.inf
            ),
            "best_trade": round(best_trade, 8),
            "worst_trade": round(worst_trade, 8),
            "final_equity": round(final_equity, 2),
            "total_equity_return": round(total_equity_return, 8),
            "max_drawdown": round(max_drawdown, 8),
            "score": round(score, 8),
        }
    )

    return base


def make_pair_side_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    summary = trades_df.groupby(["pair", "side"], as_index=False).agg(
        trades=("net_return", "count"),
        win_rate=("win", "mean"),
        avg_return=("net_return", "mean"),
        total_return=("net_return", "sum"),
        best=("net_return", "max"),
        worst=("net_return", "min"),
    )

    summary = summary.sort_values("total_return", ascending=False).reset_index(
        drop=True
    )

    return summary


# =========================================================
# CANDIDATES
# =========================================================


def make_candidates(all_data: dict[str, PairData], params: Params) -> pd.DataFrame:
    rows = []

    for pair, raw_data in all_data.items():
        data = prepare_features(raw_data, params)

        if len(data.close) == 0:
            continue

        long_signal, short_signal = make_signal_arrays(data)

        idx = len(data.close) - 1

        side = None

        if bool(long_signal[idx]):
            side = "LONG"
        elif bool(short_signal[idx]):
            side = "SHORT"

        if side is None:
            continue

        if is_blocked(pair, side):
            continue

        close = float(data.close[idx])
        atr = float(data.atr[idx])

        if side == "LONG":
            stop_ref = close - atr * params.atr_mult
        else:
            stop_ref = close + atr * params.atr_mult

        rows.append(
            {
                "signal_date": data.dates[idx].date(),
                "pair": pair,
                "side": side,
                "close": close,
                "ma": float(data.ma[idx]),
                "atr": atr,
                "prev_high": float(data.prev_high[idx]),
                "prev_low": float(data.prev_low[idx]),
                "stop_ref": float(stop_ref),
                "next_action": "next open entry candidate",
                **asdict(params),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =========================================================
# PARAM SEARCH
# =========================================================


def run_param_search(all_data: dict[str, PairData]) -> pd.DataFrame:
    params_list = iter_params()

    print("")
    print("========================================")
    print(" PARAM SEARCH START")
    print("========================================")
    print(f"total combinations: {len(params_list)}")
    print(f"MIN_TRADES        : {MIN_TRADES}")
    print(f"BLOCKED_PAIR_SIDES: {sorted(BLOCKED_PAIR_SIDES)}")
    print("========================================")

    started = time.perf_counter()

    results = []
    best_summary = None

    for idx, params in enumerate(params_list, start=1):
        trades = backtest_all_min(all_data, params)
        summary = summarize_min(params, trades, run_id=idx)
        results.append(summary)

        if best_summary is None or float(summary["score"]) > float(
            best_summary["score"]
        ):
            best_summary = summary

        if idx == 1 or idx % 25 == 0 or idx == len(params_list):
            elapsed = time.perf_counter() - started
            print(
                f"[{idx}/{len(params_list)}] "
                f"elapsed={elapsed:.1f}s "
                f"best_score={best_summary['score']} "
                f"best_final_equity={best_summary['final_equity']} "
                f"best_trades={best_summary['trades']} "
                f"best_dd={best_summary['max_drawdown']}"
            )

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(
            ["score", "final_equity", "max_drawdown", "trades"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    return results_df


def params_from_best_row(results_df: pd.DataFrame) -> Params | None:
    if results_df.empty:
        return None

    best = results_df.iloc[0]

    return Params(
        breakout_days=int(best["breakout_days"]),
        ma_days=int(best["ma_days"]),
        atr_days=int(best["atr_days"]),
        atr_mult=float(best["atr_mult"]),
        max_hold_days=int(best["max_hold_days"]),
    )


# =========================================================
# REPORT
# =========================================================


def print_best_summary(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        print("No best summary.")
        return

    best = results_df.iloc[0]

    print("")
    print("========================================")
    print(" BEST PARAM")
    print("========================================")
    print(f"run_id              : {best['run_id']}")
    print(f"score               : {best['score']}")
    print(f"final_equity        : {best['final_equity']}")
    print(f"total_equity_return : {best['total_equity_return']}")
    print(f"max_drawdown        : {best['max_drawdown']}")
    print(f"trades              : {best['trades']}")
    print(f"long_trades         : {best['long_trades']}")
    print(f"short_trades        : {best['short_trades']}")
    print(f"win_rate            : {best['win_rate']}")
    print(f"avg_return          : {best['avg_return']}")
    print(f"median_return       : {best['median_return']}")
    print(f"profit_factor       : {best['profit_factor']}")
    print(f"best_trade          : {best['best_trade']}")
    print(f"worst_trade         : {best['worst_trade']}")
    print("")
    print("[PARAMS]")
    print(f"breakout_days       : {best['breakout_days']}")
    print(f"ma_days             : {best['ma_days']}")
    print(f"atr_days            : {best['atr_days']}")
    print(f"atr_mult            : {best['atr_mult']}")
    print(f"max_hold_days       : {best['max_hold_days']}")


def print_top_results(results_df: pd.DataFrame, top_n: int = 20) -> None:
    if results_df.empty:
        print("No parameter results.")
        return

    print("")
    print("========================================")
    print(f" TOP {top_n} PARAM RESULTS")
    print("========================================")

    cols = [
        "run_id",
        "score",
        "final_equity",
        "total_equity_return",
        "max_drawdown",
        "trades",
        "win_rate",
        "avg_return",
        "median_return",
        "profit_factor",
        "best_trade",
        "worst_trade",
        "breakout_days",
        "ma_days",
        "atr_days",
        "atr_mult",
        "max_hold_days",
    ]

    show_cols = [c for c in cols if c in results_df.columns]
    print(results_df.head(top_n)[show_cols].to_string(index=False))


def print_pair_side_summary(summary_df: pd.DataFrame) -> None:
    print("")
    print("========================================")
    print(" PAIR SIDE SUMMARY")
    print("========================================")

    if summary_df.empty:
        print("(none)")
        return

    print(summary_df.to_string(index=False))


# =========================================================
# MAIN
# =========================================================


def run():
    print("========================================")
    print(" FX TREND ATR BACKTEST v800")
    print("========================================")
    print(f"START             : {START}")
    print(f"END               : {END}")
    print(f"PAIRS             : {PAIRS}")
    print(f"INITIAL_EQUITY    : {INITIAL_EQUITY}")
    print(f"POSITION_FRACTION : {POSITION_FRACTION}")
    print(f"MIN_TRADES        : {MIN_TRADES}")
    print(f"BLOCKED           : {sorted(BLOCKED_PAIR_SIDES)}")
    print("========================================")

    all_data = fetch_all_data()

    available_pairs = list(all_data.keys())

    print("")
    print("========================================")
    print(" DATA SUMMARY")
    print("========================================")
    print(f"available_pairs: {available_pairs}")

    if not available_pairs:
        print("No data available.")
        return

    results_df = run_param_search(all_data)
    results_df.to_csv(OUT_RESULTS, index=False)

    best_params = params_from_best_row(results_df)

    if best_params is None:
        pd.DataFrame().to_csv(OUT_BEST_TRADES, index=False)
        pd.DataFrame().to_csv(OUT_BEST_EQUITY, index=False)
        pd.DataFrame().to_csv(OUT_PAIR_SIDE_SUMMARY, index=False)
        pd.DataFrame().to_csv(OUT_CANDIDATES, index=False)
        print("No best params.")
        return

    print("")
    print("========================================")
    print(" REBUILD BEST TRADES")
    print("========================================")

    best_trades_df = backtest_all_full(all_data, best_params)
    best_equity_df = make_equity_curve_from_full(best_trades_df)
    pair_side_summary_df = make_pair_side_summary(best_trades_df)
    candidates_df = make_candidates(all_data, best_params)

    best_trades_df.to_csv(OUT_BEST_TRADES, index=False)
    best_equity_df.to_csv(OUT_BEST_EQUITY, index=False)
    pair_side_summary_df.to_csv(OUT_PAIR_SIDE_SUMMARY, index=False)
    candidates_df.to_csv(OUT_CANDIDATES, index=False)

    print_best_summary(results_df)
    print_top_results(results_df, top_n=20)
    print_pair_side_summary(pair_side_summary_df)

    print("")
    print("========================================")
    print(" CURRENT CANDIDATES BY BEST PARAM")
    print("========================================")

    if candidates_df.empty:
        print("(none)")
    else:
        print(candidates_df.to_string(index=False))

    print("")
    print("========================================")
    print(" SAVED")
    print("========================================")
    print(f"- {OUT_RESULTS}")
    print(f"- {OUT_BEST_TRADES}")
    print(f"- {OUT_BEST_EQUITY}")
    print(f"- {OUT_PAIR_SIDE_SUMMARY}")
    print(f"- {OUT_CANDIDATES}")

    print("")
    print("Done.")


if __name__ == "__main__":
    run()
