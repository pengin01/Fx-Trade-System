import math
import time
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# FX TREND ATR WALK FORWARD v801
# =========================================================
#
# 目的:
# - fx_v800_trend_atr_backtest.py のWalk Forward版
# - train期間でATRトレンドフォロー戦略のパラメータを最適化
# - test期間では最適化済みパラメータを固定して評価
#
# 実行:
#   python .\fx_v801_trend_atr_walkforward.py
#
# 出力:
# - fx_v801_walkforward_summary.csv
# - fx_v801_walkforward_params.csv
# - fx_v801_walkforward_trades.csv
# - fx_v801_walkforward_equity.csv
# - fx_v801_walkforward_pair_side_summary.csv
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

MIN_TRAIN_TRADES = 20


# =========================================================
# PAIR SIDE FILTER
# =========================================================
#
# v800で全期間バックテストが改善した方向だけ残す。
#
# 残す方向:
# - USDJPY=X LONG
# - GBPJPY=X SHORT
# - EURJPY=X LONG
# - GBPUSD=X SHORT
# - EURUSD=X LONG
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
# WALK FORWARD SPLITS
# =========================================================

WALK_FORWARD_SPLITS = [
    {
        "wf_id": 1,
        "train_start": "2018-01-01",
        "train_end": "2020-12-31",
        "test_start": "2021-01-01",
        "test_end": "2021-12-31",
    },
    {
        "wf_id": 2,
        "train_start": "2018-01-01",
        "train_end": "2021-12-31",
        "test_start": "2022-01-01",
        "test_end": "2022-12-31",
    },
    {
        "wf_id": 3,
        "train_start": "2018-01-01",
        "train_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
    },
    {
        "wf_id": 4,
        "train_start": "2018-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
    {
        "wf_id": 5,
        "train_start": "2018-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
    {
        "wf_id": 6,
        "train_start": "2018-01-01",
        "train_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-12-31",
    },
]


# =========================================================
# OUTPUT FILES
# =========================================================

OUT_SUMMARY = "fx_v801_walkforward_summary.csv"
OUT_PARAMS = "fx_v801_walkforward_params.csv"
OUT_TRADES = "fx_v801_walkforward_trades.csv"
OUT_EQUITY = "fx_v801_walkforward_equity.csv"
OUT_PAIR_SIDE_SUMMARY = "fx_v801_walkforward_pair_side_summary.csv"


# =========================================================
# PARAM GRID
# =========================================================
#
# v800で良かった周辺を中心にする。
#
# v800ベスト:
# breakout_days : 20
# ma_days       : 200
# atr_days      : 14
# atr_mult      : 1.5
# max_hold_days : 20
#
# =========================================================

PARAM_GRID = {
    "breakout_days": [20, 40, 60],
    "ma_days": [100, 150, 200],
    "atr_days": [14],
    "atr_mult": [1.2, 1.5, 2.0, 2.5],
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


def parse_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


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


def slice_feature_data(data: FeatureData, start: str, end: str) -> FeatureData:
    start_ts = parse_date(start)
    end_ts = parse_date(end)

    mask = (data.dates >= start_ts) & (data.dates <= end_ts)

    if mask.sum() == 0:
        return FeatureData(
            pair=data.pair,
            dates=pd.DatetimeIndex([]),
            date_values=np.array([], dtype=np.int64),
            open=np.array([], dtype=np.float64),
            high=np.array([], dtype=np.float64),
            low=np.array([], dtype=np.float64),
            close=np.array([], dtype=np.float64),
            ma=np.array([], dtype=np.float64),
            atr=np.array([], dtype=np.float64),
            prev_high=np.array([], dtype=np.float64),
            prev_low=np.array([], dtype=np.float64),
        )

    return FeatureData(
        pair=data.pair,
        dates=data.dates[mask],
        date_values=data.date_values[mask],
        open=data.open[mask],
        high=data.high[mask],
        low=data.low[mask],
        close=data.close[mask],
        ma=data.ma[mask],
        atr=data.atr[mask],
        prev_high=data.prev_high[mask],
        prev_low=data.prev_low[mask],
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


def make_feature_cache(
    all_data: dict[str, PairData], params_list: list[Params]
) -> dict[Params, dict[str, FeatureData]]:
    """
    同じパラメータでtrain/testのたびに特徴量を作り直さないようにキャッシュする。
    """
    cache = {}

    for idx, params in enumerate(params_list, start=1):
        feature_map = {}

        for pair, raw_data in all_data.items():
            feature_map[pair] = prepare_features(raw_data, params)

        cache[params] = feature_map

        if idx == 1 or idx % 25 == 0 or idx == len(params_list):
            print(f"[FEATURE CACHE] {idx}/{len(params_list)}")

    return cache


def slice_feature_map(
    feature_map: dict[str, FeatureData], start: str, end: str
) -> dict[str, FeatureData]:
    sliced = {}

    for pair, data in feature_map.items():
        d = slice_feature_data(data, start, end)

        if len(d.close) > 0:
            sliced[pair] = d

    return sliced


# =========================================================
# SIGNAL
# =========================================================


def make_signal_arrays(data: FeatureData) -> tuple[np.ndarray, np.ndarray]:
    close = data.close
    ma = data.ma
    prev_high = data.prev_high
    prev_low = data.prev_low

    long_signal = (close > ma) & (close > prev_high)

    short_signal = (close < ma) & (close < prev_low)

    return long_signal, short_signal


# =========================================================
# BACKTEST
# =========================================================


def backtest_pair_min(
    data: FeatureData, params: Params
) -> list[tuple[int, float, int]]:
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
                if l[j] <= stop_price:
                    exit_index = j
                    exit_price = stop_price
                    break

                if c[j] < ma[j]:
                    exit_index = j
                    exit_price = c[j]
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
                    break

        else:
            stop_price = entry_price + entry_atr * params.atr_mult
            lowest_low = entry_price

            for j in range(entry_index, last_j + 1):
                if h[j] >= stop_price:
                    exit_index = j
                    exit_price = stop_price
                    break

                if c[j] > ma[j]:
                    exit_index = j
                    exit_price = c[j]
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


def backtest_pair_full(
    data: FeatureData, params: Params, wf_id: int | None = None
) -> list[dict]:
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

        initial_stop = np.nan
        final_stop = np.nan

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
            "wf_id": wf_id,
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
    feature_map: dict[str, FeatureData], params: Params
) -> list[tuple[int, float, int]]:
    trades = []

    for _, data in feature_map.items():
        trades.extend(backtest_pair_min(data, params))

    return trades


def backtest_all_full(
    feature_map: dict[str, FeatureData], params: Params, wf_id: int | None = None
) -> pd.DataFrame:
    rows = []

    for _, data in feature_map.items():
        rows.extend(backtest_pair_full(data, params, wf_id=wf_id))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["entry_date", "pair", "side"]).reset_index(drop=True)

    return df


# =========================================================
# METRICS
# =========================================================


def calc_profit_factor_from_returns(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0

    wins = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()

    if losses == 0:
        if wins > 0:
            return np.inf
        return 0.0

    return float(wins / abs(losses))


def make_equity_curve_from_trades_df(
    trades_df: pd.DataFrame,
    initial_equity: float = INITIAL_EQUITY,
) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "wf_id",
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

    equity = initial_equity
    peak = initial_equity
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
                "wf_id": tr.get("wf_id", None),
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


def summarize_trades_df(trades_df: pd.DataFrame, prefix: str = "") -> dict:
    p = f"{prefix}_" if prefix else ""

    base = {
        f"{p}trades": 0,
        f"{p}long_trades": 0,
        f"{p}short_trades": 0,
        f"{p}win_rate": np.nan,
        f"{p}avg_return": np.nan,
        f"{p}median_return": np.nan,
        f"{p}total_return_sum": 0.0,
        f"{p}profit_factor": 0.0,
        f"{p}best_trade": np.nan,
        f"{p}worst_trade": np.nan,
        f"{p}final_equity": INITIAL_EQUITY,
        f"{p}total_equity_return": 0.0,
        f"{p}max_drawdown": 0.0,
    }

    if trades_df.empty:
        return base

    returns = trades_df["net_return"].astype(float).to_numpy()

    eq = make_equity_curve_from_trades_df(
        trades_df,
        initial_equity=INITIAL_EQUITY,
    )

    if eq.empty:
        final_equity = INITIAL_EQUITY
        total_equity_return = 0.0
        max_drawdown = 0.0
    else:
        final_equity = float(eq["equity"].iloc[-1])
        total_equity_return = final_equity / INITIAL_EQUITY - 1
        max_drawdown = float(eq["drawdown"].min())

    base.update(
        {
            f"{p}trades": int(len(trades_df)),
            f"{p}long_trades": int((trades_df["side"] == "LONG").sum()),
            f"{p}short_trades": int((trades_df["side"] == "SHORT").sum()),
            f"{p}win_rate": round(float((returns > 0).mean()), 6),
            f"{p}avg_return": round(float(returns.mean()), 8),
            f"{p}median_return": round(float(np.median(returns)), 8),
            f"{p}total_return_sum": round(float(returns.sum()), 8),
            f"{p}profit_factor": round(calc_profit_factor_from_returns(returns), 6),
            f"{p}best_trade": round(float(returns.max()), 8),
            f"{p}worst_trade": round(float(returns.min()), 8),
            f"{p}final_equity": round(final_equity, 2),
            f"{p}total_equity_return": round(total_equity_return, 8),
            f"{p}max_drawdown": round(max_drawdown, 8),
        }
    )

    return base


def summarize_min_for_score(
    params: Params,
    trades: list[tuple[int, float, int]],
    run_id: int,
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
        "final_equity": INITIAL_EQUITY,
        "total_equity_return": 0.0,
        "max_drawdown": 0.0,
        "score": -999999.0,
    }

    if not trades:
        return base

    arr = np.array(trades, dtype=np.float64)

    exit_times = arr[:, 0]
    returns = arr[:, 1]
    sides = arr[:, 2]

    trade_count = len(returns)

    order = np.argsort(exit_times)

    equity = INITIAL_EQUITY
    peak = INITIAL_EQUITY
    max_drawdown = 0.0

    for idx in order:
        net_return = float(returns[idx])
        pnl = equity * POSITION_FRACTION * net_return
        equity += pnl

        if equity > peak:
            peak = equity

        drawdown = equity / peak - 1

        if drawdown < max_drawdown:
            max_drawdown = drawdown

    total_equity_return = equity / INITIAL_EQUITY - 1
    avg_return = float(returns.mean())

    if trade_count < MIN_TRAIN_TRADES:
        score = -999999.0 + trade_count
    else:
        score = total_equity_return - abs(max_drawdown) * 0.6 + avg_return * 8

    base.update(
        {
            "trades": int(trade_count),
            "long_trades": int((sides == 1).sum()),
            "short_trades": int((sides == -1).sum()),
            "win_rate": round(float((returns > 0).mean()), 6),
            "avg_return": round(avg_return, 8),
            "median_return": round(float(np.median(returns)), 8),
            "total_return_sum": round(float(returns.sum()), 8),
            "profit_factor": round(calc_profit_factor_from_returns(returns), 6),
            "final_equity": round(float(equity), 2),
            "total_equity_return": round(float(total_equity_return), 8),
            "max_drawdown": round(float(max_drawdown), 8),
            "score": round(float(score), 8),
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
# OPTIMIZATION
# =========================================================


def optimize_params(
    feature_cache: dict[Params, dict[str, FeatureData]],
    split: dict,
) -> tuple[Params | None, pd.DataFrame]:
    wf_id = int(split["wf_id"])
    train_start = split["train_start"]
    train_end = split["train_end"]

    params_list = list(feature_cache.keys())

    print("")
    print("========================================")
    print(f" OPTIMIZE WF {wf_id}")
    print("========================================")
    print(f"TRAIN: {train_start} -> {train_end}")
    print(f"total combinations: {len(params_list)}")
    print(f"MIN_TRAIN_TRADES  : {MIN_TRAIN_TRADES}")
    print("========================================")

    started = time.perf_counter()

    rows = []
    best_row = None
    best_params = None

    for idx, params in enumerate(params_list, start=1):
        train_feature_map = slice_feature_map(
            feature_cache[params],
            train_start,
            train_end,
        )

        trades = backtest_all_min(train_feature_map, params)
        row = summarize_min_for_score(params, trades, run_id=idx)
        row["wf_id"] = wf_id
        rows.append(row)

        if best_row is None or float(row["score"]) > float(best_row["score"]):
            best_row = row
            best_params = params

        if idx == 1 or idx % 25 == 0 or idx == len(params_list):
            elapsed = time.perf_counter() - started
            print(
                f"[WF {wf_id}] [{idx}/{len(params_list)}] "
                f"elapsed={elapsed:.1f}s "
                f"best_score={best_row['score']} "
                f"best_final_equity={best_row['final_equity']} "
                f"best_trades={best_row['trades']} "
                f"best_dd={best_row['max_drawdown']}"
            )

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        results_df = results_df.sort_values(
            ["score", "final_equity", "max_drawdown", "trades"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    return best_params, results_df


# =========================================================
# WALK FORWARD
# =========================================================


def run_one_walkforward(
    feature_cache: dict[Params, dict[str, FeatureData]],
    split: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    wf_id = int(split["wf_id"])

    train_start = split["train_start"]
    train_end = split["train_end"]
    test_start = split["test_start"]
    test_end = split["test_end"]

    print("")
    print("========================================")
    print(f" WALK FORWARD {wf_id}")
    print("========================================")
    print(f"TRAIN: {train_start} -> {train_end}")
    print(f"TEST : {test_start} -> {test_end}")
    print("========================================")

    best_params, train_param_results = optimize_params(
        feature_cache,
        split=split,
    )

    if best_params is None:
        summary_row = {
            "wf_id": wf_id,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "status": "NO_PARAMS",
        }

        return summary_row, train_param_results, pd.DataFrame()

    train_feature_map = slice_feature_map(
        feature_cache[best_params],
        train_start,
        train_end,
    )

    test_feature_map = slice_feature_map(
        feature_cache[best_params],
        test_start,
        test_end,
    )

    train_best_trades_df = backtest_all_full(
        train_feature_map,
        best_params,
        wf_id=wf_id,
    )

    test_trades_df = backtest_all_full(
        test_feature_map,
        best_params,
        wf_id=wf_id,
    )

    summary_row = {
        "wf_id": wf_id,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "status": "OK",
        **asdict(best_params),
    }

    summary_row.update(summarize_trades_df(train_best_trades_df, prefix="train"))
    summary_row.update(summarize_trades_df(test_trades_df, prefix="test"))

    print("")
    print("========================================")
    print(f" WF {wf_id} RESULT")
    print("========================================")
    print("[BEST PARAM]")
    print(f"breakout_days : {best_params.breakout_days}")
    print(f"ma_days       : {best_params.ma_days}")
    print(f"atr_days      : {best_params.atr_days}")
    print(f"atr_mult      : {best_params.atr_mult}")
    print(f"max_hold_days : {best_params.max_hold_days}")
    print("")
    print("[TRAIN]")
    print(f"trades        : {summary_row.get('train_trades')}")
    print(f"final_equity  : {summary_row.get('train_final_equity')}")
    print(f"max_dd        : {summary_row.get('train_max_drawdown')}")
    print(f"profit_factor : {summary_row.get('train_profit_factor')}")
    print("")
    print("[TEST]")
    print(f"trades        : {summary_row.get('test_trades')}")
    print(f"final_equity  : {summary_row.get('test_final_equity')}")
    print(f"max_dd        : {summary_row.get('test_max_drawdown')}")
    print(f"profit_factor : {summary_row.get('test_profit_factor')}")

    return summary_row, train_param_results, test_trades_df


def make_overall_test_equity(all_test_trades_df: pd.DataFrame) -> pd.DataFrame:
    if all_test_trades_df.empty:
        return pd.DataFrame()

    return make_equity_curve_from_trades_df(
        all_test_trades_df,
        initial_equity=INITIAL_EQUITY,
    )


def print_overall_summary(
    summary_df: pd.DataFrame,
    all_test_trades_df: pd.DataFrame,
    overall_equity_df: pd.DataFrame,
) -> None:
    print("")
    print("========================================")
    print(" WALK FORWARD OVERALL SUMMARY")
    print("========================================")

    if summary_df.empty:
        print("No summary.")
        return

    show_cols = [
        "wf_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "breakout_days",
        "ma_days",
        "atr_days",
        "atr_mult",
        "max_hold_days",
        "train_trades",
        "train_final_equity",
        "train_max_drawdown",
        "train_profit_factor",
        "test_trades",
        "test_final_equity",
        "test_max_drawdown",
        "test_profit_factor",
    ]

    show_cols = [c for c in show_cols if c in summary_df.columns]
    print(summary_df[show_cols].to_string(index=False))

    print("")
    print("========================================")
    print(" ALL TEST TRADES SUMMARY")
    print("========================================")

    if all_test_trades_df.empty:
        print("(no test trades)")
        return

    returns = all_test_trades_df["net_return"].astype(float).to_numpy()
    pf = calc_profit_factor_from_returns(returns)

    if overall_equity_df.empty:
        final_equity = INITIAL_EQUITY
        max_dd = 0.0
    else:
        final_equity = float(overall_equity_df["equity"].iloc[-1])
        max_dd = float(overall_equity_df["drawdown"].min())

    print(f"test_trades       : {len(all_test_trades_df)}")
    print(f"test_win_rate     : {float((returns > 0).mean()):.6f}")
    print(f"test_avg_return   : {float(returns.mean()):.8f}")
    print(f"test_median_return: {float(np.median(returns)):.8f}")
    print(f"test_profit_factor: {pf:.6f}")
    print(f"test_final_equity : {final_equity:.2f}")
    print(f"test_total_return : {final_equity / INITIAL_EQUITY - 1:.8f}")
    print(f"test_max_drawdown : {max_dd:.8f}")


# =========================================================
# MAIN
# =========================================================


def run():
    print("========================================")
    print(" FX TREND ATR WALK FORWARD v801")
    print("========================================")
    print(f"START             : {START}")
    print(f"END               : {END}")
    print(f"PAIRS             : {PAIRS}")
    print(f"INITIAL_EQUITY    : {INITIAL_EQUITY}")
    print(f"POSITION_FRACTION : {POSITION_FRACTION}")
    print(f"MIN_TRAIN_TRADES  : {MIN_TRAIN_TRADES}")
    print(f"BLOCKED           : {sorted(BLOCKED_PAIR_SIDES)}")
    print("========================================")

    all_data = fetch_all_data()

    if not all_data:
        print("No data available.")
        return

    params_list = iter_params()

    print("")
    print("========================================")
    print(" FEATURE CACHE BUILD")
    print("========================================")
    print(f"total combinations: {len(params_list)}")
    print("========================================")

    feature_cache = make_feature_cache(all_data, params_list)

    all_summary_rows = []
    all_param_rows = []
    all_test_trades = []

    for split in WALK_FORWARD_SPLITS:
        summary_row, param_results_df, test_trades_df = run_one_walkforward(
            feature_cache,
            split,
        )

        all_summary_rows.append(summary_row)

        if not param_results_df.empty:
            all_param_rows.append(param_results_df)

        if not test_trades_df.empty:
            all_test_trades.append(test_trades_df)

    summary_df = pd.DataFrame(all_summary_rows)

    if all_param_rows:
        params_df = pd.concat(all_param_rows, ignore_index=True)
    else:
        params_df = pd.DataFrame()

    if all_test_trades:
        all_test_trades_df = pd.concat(all_test_trades, ignore_index=True)
        all_test_trades_df = all_test_trades_df.sort_values(
            ["exit_time", "pair", "side"]
        ).reset_index(drop=True)
    else:
        all_test_trades_df = pd.DataFrame()

    overall_equity_df = make_overall_test_equity(all_test_trades_df)
    pair_side_summary_df = make_pair_side_summary(all_test_trades_df)

    summary_df.to_csv(OUT_SUMMARY, index=False)
    params_df.to_csv(OUT_PARAMS, index=False)
    all_test_trades_df.to_csv(OUT_TRADES, index=False)
    overall_equity_df.to_csv(OUT_EQUITY, index=False)
    pair_side_summary_df.to_csv(OUT_PAIR_SIDE_SUMMARY, index=False)

    print_overall_summary(
        summary_df,
        all_test_trades_df,
        overall_equity_df,
    )

    print("")
    print("========================================")
    print(" TEST PAIR SIDE SUMMARY")
    print("========================================")

    if pair_side_summary_df.empty:
        print("(none)")
    else:
        print(pair_side_summary_df.to_string(index=False))

    print("")
    print("========================================")
    print(" SAVED")
    print("========================================")
    print(f"- {OUT_SUMMARY}")
    print(f"- {OUT_PARAMS}")
    print(f"- {OUT_TRADES}")
    print(f"- {OUT_EQUITY}")
    print(f"- {OUT_PAIR_SIDE_SUMMARY}")

    print("")
    print("Done.")


if __name__ == "__main__":
    run()
