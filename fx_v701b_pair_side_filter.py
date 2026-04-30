import math
import time
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# FX PARAM SEARCH v701b - PAIR SIDE FILTER
# =========================================================
#
# 実行:
#   python .\fx_v701b_pair_side_filter.py
#
# 目的:
# - fx_v701_param_search.py の高速版をベースにする
# - 通貨ペア × 売買方向 の禁止フィルタを追加する
# - まずは EURJPY=X SHORT だけ禁止して検証する
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

RSI_DAYS = 14
MA_DAYS = 25
LOOKBACK_HIGH_LOW = 20

MIN_TRADES = 50


# =========================================================
# PAIR SIDE FILTER
# =========================================================
#
# v701 の結果で一番悪かった EURJPY=X SHORT だけ禁止。
#
# 必要なら後で以下のように増やせます。
#
# BLOCKED_PAIR_SIDES = {
#     ("EURJPY=X", "SHORT"),
#     ("USDJPY=X", "SHORT"),
#     ("GBPJPY=X", "SHORT"),
#     ("EURUSD=X", "LONG"),
#     ("GBPUSD=X", "LONG"),
# }
#
# =========================================================

BLOCKED_PAIR_SIDES = {
    ("EURJPY=X", "SHORT"),
    ("USDJPY=X", "SHORT"),
    ("GBPJPY=X", "SHORT"),
    ("GBPJPY=X", "LONG"),
    ("EURUSD=X", "LONG"),
    ("GBPUSD=X", "LONG"),
    ("AUDUSD=X", "SHORT"),
}


# =========================================================
# OUTPUT FILES
# =========================================================

OUT_RESULTS = "fx_param_results_v701b.csv"
OUT_BEST_TRADES = "fx_param_best_trades_v701b.csv"
OUT_BEST_EQUITY = "fx_param_best_equity_v701b.csv"
OUT_BEST_CANDIDATES = "fx_param_best_candidates_v701b.csv"
OUT_PAIR_SIDE_SUMMARY = "fx_pair_side_summary_v701b.csv"


# =========================================================
# PARAM GRID
# =========================================================

PARAM_GRID = {
    "pullback_pct": [0.003, 0.005, 0.007, 0.010],
    "tp_pct": [0.004, 0.006, 0.008, 0.010],
    "sl_pct": [0.004, 0.006, 0.008],
    "hold_days": [2, 3, 4, 5, 7],
    "rsi_long_max": [40, 45, 50],
    "rsi_short_min": [50, 55, 60],
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
    pullback_pct: float
    tp_pct: float
    sl_pct: float
    hold_days: int
    rsi_long_max: int
    rsi_short_min: int


@dataclass
class PairData:
    pair: str
    dates: np.ndarray
    date_values: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    ma: np.ndarray
    rsi: np.ndarray
    prev_high: np.ndarray
    prev_low: np.ndarray


# =========================================================
# UTILS
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


def to_float_array(series: pd.Series) -> np.ndarray:
    return series.astype(float).to_numpy(dtype=np.float64)


def is_blocked(pair: str, side: str) -> bool:
    return (pair, side) in BLOCKED_PAIR_SIDES


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


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ma"] = df["Close"].rolling(MA_DAYS).mean()
    df["rsi"] = calc_rsi(df["Close"], RSI_DAYS)

    df["prev_high"] = df["Close"].rolling(LOOKBACK_HIGH_LOW).max().shift(1)
    df["prev_low"] = df["Close"].rolling(LOOKBACK_HIGH_LOW).min().shift(1)

    df = df.dropna().copy()

    return df


def to_pair_data(pair: str, df: pd.DataFrame) -> PairData:
    dates = pd.to_datetime(df.index).to_pydatetime()
    date_values = pd.to_datetime(df.index).astype("int64").to_numpy()

    return PairData(
        pair=pair,
        dates=np.array(dates),
        date_values=date_values,
        open=to_float_array(df["Open"]),
        high=to_float_array(df["High"]),
        low=to_float_array(df["Low"]),
        close=to_float_array(df["Close"]),
        ma=to_float_array(df["ma"]),
        rsi=to_float_array(df["rsi"]),
        prev_high=to_float_array(df["prev_high"]),
        prev_low=to_float_array(df["prev_low"]),
    )


def fetch_all_data() -> dict[str, PairData]:
    all_data = {}

    for pair in PAIRS:
        raw = fetch_pair(pair)

        if raw.empty:
            continue

        featured = prepare_features(raw)

        if featured.empty:
            print(f"[WARN] no featured rows: {pair}")
            continue

        data = to_pair_data(pair, featured)
        all_data[pair] = data

        print(f"[DATA] {pair}: rows={len(data.close)}")

    return all_data


# =========================================================
# PARAMS
# =========================================================


def iter_params() -> list[Params]:
    keys = [
        "pullback_pct",
        "tp_pct",
        "sl_pct",
        "hold_days",
        "rsi_long_max",
        "rsi_short_min",
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


def make_signal_arrays(data: PairData, params: Params) -> tuple[np.ndarray, np.ndarray]:
    close = data.close
    ma = data.ma
    rsi = data.rsi
    prev_high = data.prev_high
    prev_low = data.prev_low

    long_signal = (
        (close > ma)
        & (close <= prev_high * (1 - params.pullback_pct))
        & (rsi <= params.rsi_long_max)
    )

    short_signal = (
        (close < ma)
        & (close >= prev_low * (1 + params.pullback_pct))
        & (rsi >= params.rsi_short_min)
    )

    return long_signal, short_signal


# =========================================================
# BACKTEST - MIN
# =========================================================


def backtest_pair_min(data: PairData, params: Params) -> list[tuple[int, float, int]]:
    """
    高速集計用。
    戻り値:
      (exit_time, net_return, side_int)
      side_int: 1=LONG, -1=SHORT
    """

    n = len(data.close)

    if n < params.hold_days + 10:
        return []

    long_signal, short_signal = make_signal_arrays(data, params)
    signal_indices = np.flatnonzero(long_signal | short_signal)

    if len(signal_indices) == 0:
        return []

    trades = []
    last_exit_index = -1

    o = data.open
    h = data.high
    l = data.low
    c = data.close

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

        if not math.isfinite(entry_price) or entry_price <= 0:
            continue

        if side_int == 1:
            tp_price = entry_price * (1 + params.tp_pct)
            sl_price = entry_price * (1 - params.sl_pct)
        else:
            tp_price = entry_price * (1 - params.tp_pct)
            sl_price = entry_price * (1 + params.sl_pct)

        last_j = min(entry_index + params.hold_days - 1, n - 1)

        exit_index = -1
        exit_price = np.nan

        for j in range(entry_index, last_j + 1):
            high = h[j]
            low = l[j]
            close = c[j]

            if side_int == 1:
                # LONG: 同日にTP/SL両方到達なら保守的にSL優先
                if low <= sl_price:
                    exit_index = j
                    exit_price = sl_price
                    break

                if high >= tp_price:
                    exit_index = j
                    exit_price = tp_price
                    break

            else:
                # SHORT: 同日にTP/SL両方到達なら保守的にSL優先
                if high >= sl_price:
                    exit_index = j
                    exit_price = sl_price
                    break

                if low <= tp_price:
                    exit_index = j
                    exit_price = tp_price
                    break

            if j == last_j:
                exit_index = j
                exit_price = close
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


def backtest_all_min(
    all_data: dict[str, PairData], params: Params
) -> list[tuple[int, float, int]]:
    trades = []

    for _, data in all_data.items():
        trades.extend(backtest_pair_min(data, params))

    return trades


# =========================================================
# BACKTEST - FULL
# =========================================================


def backtest_pair_full(data: PairData, params: Params) -> list[dict]:
    """
    最良パラメータ用の詳細トレード生成。
    """

    n = len(data.close)

    if n < params.hold_days + 10:
        return []

    long_signal, short_signal = make_signal_arrays(data, params)
    signal_indices = np.flatnonzero(long_signal | short_signal)

    if len(signal_indices) == 0:
        return []

    trades = []
    last_exit_index = -1

    o = data.open
    h = data.high
    l = data.low
    c = data.close

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

        if not math.isfinite(entry_price) or entry_price <= 0:
            continue

        if side_int == 1:
            tp_price = entry_price * (1 + params.tp_pct)
            sl_price = entry_price * (1 - params.sl_pct)
        else:
            tp_price = entry_price * (1 - params.tp_pct)
            sl_price = entry_price * (1 + params.sl_pct)

        last_j = min(entry_index + params.hold_days - 1, n - 1)

        exit_index = -1
        exit_price = np.nan
        exit_reason = ""

        for j in range(entry_index, last_j + 1):
            high = h[j]
            low = l[j]
            close = c[j]

            if side_int == 1:
                if low <= sl_price:
                    exit_index = j
                    exit_price = sl_price
                    exit_reason = "SL"
                    break

                if high >= tp_price:
                    exit_index = j
                    exit_price = tp_price
                    exit_reason = "TP"
                    break

            else:
                if high >= sl_price:
                    exit_index = j
                    exit_price = sl_price
                    exit_reason = "SL"
                    break

                if low <= tp_price:
                    exit_index = j
                    exit_price = tp_price
                    exit_reason = "TP"
                    break

            if j == last_j:
                exit_index = j
                exit_price = close
                exit_reason = "TIME"
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
            {
                "pair": data.pair,
                "side": side,
                "signal_date": data.dates[signal_index].date(),
                "entry_date": data.dates[entry_index].date(),
                "exit_date": data.dates[exit_index].date(),
                "exit_time": int(data.date_values[exit_index]),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "tp_price": float(tp_price),
                "sl_price": float(sl_price),
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
                "signal_ma": float(data.ma[signal_index]),
                "signal_rsi": float(data.rsi[signal_index]),
                "signal_prev_high": float(data.prev_high[signal_index]),
                "signal_prev_low": float(data.prev_low[signal_index]),
                **asdict(params),
            }
        )

        last_exit_index = exit_index

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
    long_trades = int((sides == 1).sum())
    short_trades = int((sides == -1).sum())

    win_rate = float((returns > 0).mean())
    avg_return = float(returns.mean())
    median_return = float(np.median(returns))
    total_return_sum = float(returns.sum())
    profit_factor = calc_profit_factor(returns)

    best_trade = float(returns.max())
    worst_trade = float(returns.min())

    eq = make_equity_curve_from_min(trades)

    if eq.empty:
        final_equity = INITIAL_EQUITY
        total_equity_return = 0.0
        max_drawdown = 0.0
    else:
        final_equity = float(eq["equity"].iloc[-1])
        total_equity_return = final_equity / INITIAL_EQUITY - 1
        max_drawdown = float(eq["drawdown"].min())

    if trade_count < MIN_TRADES:
        score = -999999.0 + trade_count
    else:
        score = total_equity_return - abs(max_drawdown) * 0.8 + avg_return * 10

    base.update(
        {
            "trades": trade_count,
            "long_trades": long_trades,
            "short_trades": short_trades,
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

    for pair, data in all_data.items():
        if len(data.close) == 0:
            continue

        long_signal, short_signal = make_signal_arrays(data, params)

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

        if side == "LONG":
            tp_price = close * (1 + params.tp_pct)
            sl_price = close * (1 - params.sl_pct)
        else:
            tp_price = close * (1 - params.tp_pct)
            sl_price = close * (1 + params.sl_pct)

        rows.append(
            {
                "signal_date": data.dates[idx].date(),
                "pair": pair,
                "side": side,
                "close": close,
                "ma": float(data.ma[idx]),
                "rsi": float(data.rsi[idx]),
                "prev_high": float(data.prev_high[idx]),
                "prev_low": float(data.prev_low[idx]),
                "tp_price_ref": float(tp_price),
                "sl_price_ref": float(sl_price),
                "next_action": "next open entry candidate",
                **asdict(params),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


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
    print(f"pullback_pct        : {best['pullback_pct']}")
    print(f"tp_pct              : {best['tp_pct']}")
    print(f"sl_pct              : {best['sl_pct']}")
    print(f"hold_days           : {best['hold_days']}")
    print(f"rsi_long_max        : {best['rsi_long_max']}")
    print(f"rsi_short_min       : {best['rsi_short_min']}")


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
        "profit_factor",
        "pullback_pct",
        "tp_pct",
        "sl_pct",
        "hold_days",
        "rsi_long_max",
        "rsi_short_min",
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


def params_from_best_row(results_df: pd.DataFrame) -> Params | None:
    if results_df.empty:
        return None

    best = results_df.iloc[0]

    return Params(
        pullback_pct=float(best["pullback_pct"]),
        tp_pct=float(best["tp_pct"]),
        sl_pct=float(best["sl_pct"]),
        hold_days=int(best["hold_days"]),
        rsi_long_max=int(best["rsi_long_max"]),
        rsi_short_min=int(best["rsi_short_min"]),
    )


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

        if idx == 1 or idx % 50 == 0 or idx == len(params_list):
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


# =========================================================
# MAIN
# =========================================================


def run():
    print("========================================")
    print(" FX PARAM SEARCH v701b PAIR SIDE FILTER")
    print("========================================")
    print(f"START             : {START}")
    print(f"END               : {END}")
    print(f"PAIRS             : {PAIRS}")
    print(f"INITIAL_EQUITY    : {INITIAL_EQUITY}")
    print(f"POSITION_FRACTION : {POSITION_FRACTION}")
    print(f"RSI_DAYS          : {RSI_DAYS}")
    print(f"MA_DAYS           : {MA_DAYS}")
    print(f"LOOKBACK          : {LOOKBACK_HIGH_LOW}")
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
        pd.DataFrame().to_csv(OUT_BEST_CANDIDATES, index=False)
        pd.DataFrame().to_csv(OUT_PAIR_SIDE_SUMMARY, index=False)
        print("No best params.")
        return

    print("")
    print("========================================")
    print(" REBUILD BEST TRADES")
    print("========================================")

    best_trades_df = backtest_all_full(all_data, best_params)
    best_equity_df = make_equity_curve_from_full(best_trades_df)
    candidates_df = make_candidates(all_data, best_params)
    pair_side_summary_df = make_pair_side_summary(best_trades_df)

    best_trades_df.to_csv(OUT_BEST_TRADES, index=False)
    best_equity_df.to_csv(OUT_BEST_EQUITY, index=False)
    candidates_df.to_csv(OUT_BEST_CANDIDATES, index=False)
    pair_side_summary_df.to_csv(OUT_PAIR_SIDE_SUMMARY, index=False)

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
    print(f"- {OUT_BEST_CANDIDATES}")
    print(f"- {OUT_PAIR_SIDE_SUMMARY}")

    print("")
    print("Done.")


if __name__ == "__main__":
    run()
