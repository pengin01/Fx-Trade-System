import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# FX POSITION SIZE TEST v803
# =========================================================
#
# 目的:
# - v802固定ロジックのまま、POSITION_FRACTIONだけ変えて比較する
# - リターンと最大ドローダウンのバランスを見る
#
# 実行:
#   python .\fx_v803_position_size_test.py
#
# 出力:
# - fx_v803_base_trades.csv
# - fx_v803_position_summary.csv
# - fx_v803_yearly_summary.csv
# - fx_v803_pair_side_summary.csv
# - fx_v803_equity_by_fraction.csv
# - fx_v803_candidates.csv
#
# =========================================================


# =========================================================
# BASIC SETTINGS
# =========================================================

PAIRS = [
    "USDJPY=X",
    "GBPUSD=X",
]

START = "2024-01-01"
END = None

INITIAL_EQUITY = 100_000

POSITION_FRACTIONS = [
    0.30,
    0.50,
    0.70,
    1.00,
    1.50,
    2.00,
]


# =========================================================
# FIXED STRATEGY PARAMS
# =========================================================

BREAKOUT_DAYS = 60
MA_DAYS = 200
ATR_DAYS = 14
ATR_MULT = 1.2
MAX_HOLD_DAYS = 20


# =========================================================
# ALLOWED PAIR SIDE
# =========================================================

ALLOWED_PAIR_SIDES = {
    ("USDJPY=X", "LONG"),
    ("GBPUSD=X", "SHORT"),
}


# =========================================================
# OUTPUT FILES
# =========================================================

OUT_BASE_TRADES = "fx_v803_base_trades.csv"
OUT_POSITION_SUMMARY = "fx_v803_position_summary.csv"
OUT_YEARLY_SUMMARY = "fx_v803_yearly_summary.csv"
OUT_PAIR_SIDE_SUMMARY = "fx_v803_pair_side_summary.csv"
OUT_EQUITY_BY_FRACTION = "fx_v803_equity_by_fraction.csv"
OUT_CANDIDATES = "fx_v803_candidates.csv"


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
    ma: np.ndarray
    atr: np.ndarray
    prev_high: np.ndarray
    prev_low: np.ndarray


# =========================================================
# UTILS
# =========================================================


def current_params() -> Params:
    return Params(
        breakout_days=BREAKOUT_DAYS,
        ma_days=MA_DAYS,
        atr_days=ATR_DAYS,
        atr_mult=ATR_MULT,
        max_hold_days=MAX_HOLD_DAYS,
    )


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


def is_allowed(pair: str, side: str) -> bool:
    return (pair, side) in ALLOWED_PAIR_SIDES


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


def prepare_features(pair: str, df: pd.DataFrame, params: Params) -> PairData:
    df = df.copy()

    df["ma"] = df["Close"].rolling(params.ma_days).mean()
    df["atr"] = calc_atr(df, params.atr_days)

    df["prev_high"] = df["High"].rolling(params.breakout_days).max().shift(1)
    df["prev_low"] = df["Low"].rolling(params.breakout_days).min().shift(1)

    df = df.dropna().copy()

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
        ma=to_float_array(df["ma"]),
        atr=to_float_array(df["atr"]),
        prev_high=to_float_array(df["prev_high"]),
        prev_low=to_float_array(df["prev_low"]),
    )


def fetch_all_data(params: Params) -> dict[str, PairData]:
    all_data = {}

    for pair in PAIRS:
        raw = fetch_pair(pair)

        if raw.empty:
            continue

        data = prepare_features(pair, raw, params)

        if len(data.close) == 0:
            print(f"[WARN] no featured rows: {pair}")
            continue

        all_data[pair] = data

        print(f"[DATA] {pair}: rows={len(data.close)}")

    return all_data


# =========================================================
# SIGNAL
# =========================================================


def make_signal_arrays(data: PairData) -> tuple[np.ndarray, np.ndarray]:
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


def backtest_pair(data: PairData, params: Params) -> list[dict]:
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

        if not is_allowed(data.pair, side):
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

        trades.append(
            {
                "pair": data.pair,
                "side": side,
                "signal_date": data.dates[signal_index].date(),
                "entry_date": data.dates[entry_index].date(),
                "exit_date": data.dates[exit_index].date(),
                "exit_year": int(data.dates[exit_index].year),
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
        )

        last_exit_index = exit_index

    return trades


def backtest_all(all_data: dict[str, PairData], params: Params) -> pd.DataFrame:
    rows = []

    for _, data in all_data.items():
        rows.extend(backtest_pair(data, params))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["exit_time", "pair", "side"]).reset_index(drop=True)

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


def make_equity_curve(
    trades_df: pd.DataFrame,
    position_fraction: float,
    initial_equity: float = INITIAL_EQUITY,
) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "position_fraction",
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
        pnl = equity * position_fraction * net_return
        equity += pnl

        if equity > peak:
            peak = equity

        drawdown = equity / peak - 1

        rows.append(
            {
                "position_fraction": position_fraction,
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


def summarize_trades(
    trades_df: pd.DataFrame,
    position_fraction: float,
    initial_equity: float = INITIAL_EQUITY,
) -> dict:
    base = {
        "position_fraction": position_fraction,
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
        "final_equity": initial_equity,
        "total_equity_return": 0.0,
        "max_drawdown": 0.0,
    }

    if trades_df.empty:
        return base

    returns = trades_df["net_return"].astype(float).to_numpy()
    eq = make_equity_curve(
        trades_df,
        position_fraction=position_fraction,
        initial_equity=initial_equity,
    )

    if eq.empty:
        final_equity = initial_equity
        total_equity_return = 0.0
        max_drawdown = 0.0
    else:
        final_equity = float(eq["equity"].iloc[-1])
        total_equity_return = final_equity / initial_equity - 1
        max_drawdown = float(eq["drawdown"].min())

    base.update(
        {
            "trades": int(len(trades_df)),
            "long_trades": int((trades_df["side"] == "LONG").sum()),
            "short_trades": int((trades_df["side"] == "SHORT").sum()),
            "win_rate": round(float((returns > 0).mean()), 6),
            "avg_return": round(float(returns.mean()), 8),
            "median_return": round(float(np.median(returns)), 8),
            "total_return_sum": round(float(returns.sum()), 8),
            "profit_factor": round(calc_profit_factor(returns), 6),
            "best_trade": round(float(returns.max()), 8),
            "worst_trade": round(float(returns.min()), 8),
            "final_equity": round(final_equity, 2),
            "total_equity_return": round(total_equity_return, 8),
            "max_drawdown": round(max_drawdown, 8),
        }
    )

    return base


def make_position_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for position_fraction in POSITION_FRACTIONS:
        rows.append(
            summarize_trades(
                trades_df,
                position_fraction=position_fraction,
                initial_equity=INITIAL_EQUITY,
            )
        )

    return pd.DataFrame(rows)


def make_yearly_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    rows = []

    for position_fraction in POSITION_FRACTIONS:
        for year, g in trades_df.groupby("exit_year"):
            summary = summarize_trades(
                g,
                position_fraction=position_fraction,
                initial_equity=INITIAL_EQUITY,
            )
            rows.append(
                {
                    "position_fraction": position_fraction,
                    "year": int(year),
                    **summary,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["position_fraction", "year"])
        .reset_index(drop=True)
    )


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

        long_signal, short_signal = make_signal_arrays(data)
        idx = len(data.close) - 1

        side = None

        if bool(long_signal[idx]):
            side = "LONG"
        elif bool(short_signal[idx]):
            side = "SHORT"

        if side is None:
            continue

        if not is_allowed(pair, side):
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
# REPORT
# =========================================================


def print_position_summary(position_summary_df: pd.DataFrame) -> None:
    print("")
    print("========================================")
    print(" POSITION SIZE SUMMARY")
    print("========================================")

    if position_summary_df.empty:
        print("(none)")
        return

    cols = [
        "position_fraction",
        "trades",
        "win_rate",
        "avg_return",
        "median_return",
        "profit_factor",
        "final_equity",
        "total_equity_return",
        "max_drawdown",
    ]

    print(position_summary_df[cols].to_string(index=False))


def print_yearly_summary(yearly_df: pd.DataFrame) -> None:
    print("")
    print("========================================")
    print(" YEARLY SUMMARY")
    print("========================================")

    if yearly_df.empty:
        print("(none)")
        return

    cols = [
        "position_fraction",
        "year",
        "trades",
        "win_rate",
        "profit_factor",
        "final_equity",
        "total_equity_return",
        "max_drawdown",
    ]

    print(yearly_df[cols].to_string(index=False))


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
    params = current_params()

    print("========================================")
    print(" FX POSITION SIZE TEST v803")
    print("========================================")
    print(f"START             : {START}")
    print(f"END               : {END}")
    print(f"PAIRS             : {PAIRS}")
    print(f"INITIAL_EQUITY    : {INITIAL_EQUITY}")
    print(f"POSITION_FRACTIONS: {POSITION_FRACTIONS}")
    print(f"ALLOWED           : {sorted(ALLOWED_PAIR_SIDES)}")
    print("")
    print("[FIXED PARAMS]")
    print(f"breakout_days     : {params.breakout_days}")
    print(f"ma_days           : {params.ma_days}")
    print(f"atr_days          : {params.atr_days}")
    print(f"atr_mult          : {params.atr_mult}")
    print(f"max_hold_days     : {params.max_hold_days}")
    print("========================================")

    all_data = fetch_all_data(params)

    if not all_data:
        print("No data available.")
        return

    trades_df = backtest_all(all_data, params)

    position_summary_df = make_position_summary(trades_df)
    yearly_summary_df = make_yearly_summary(trades_df)
    pair_side_summary_df = make_pair_side_summary(trades_df)
    candidates_df = make_candidates(all_data, params)

    equity_frames = []

    for position_fraction in POSITION_FRACTIONS:
        eq = make_equity_curve(
            trades_df,
            position_fraction=position_fraction,
            initial_equity=INITIAL_EQUITY,
        )
        equity_frames.append(eq)

    if equity_frames:
        equity_by_fraction_df = pd.concat(equity_frames, ignore_index=True)
    else:
        equity_by_fraction_df = pd.DataFrame()

    trades_df.to_csv(OUT_BASE_TRADES, index=False)
    position_summary_df.to_csv(OUT_POSITION_SUMMARY, index=False)
    yearly_summary_df.to_csv(OUT_YEARLY_SUMMARY, index=False)
    pair_side_summary_df.to_csv(OUT_PAIR_SIDE_SUMMARY, index=False)
    equity_by_fraction_df.to_csv(OUT_EQUITY_BY_FRACTION, index=False)
    candidates_df.to_csv(OUT_CANDIDATES, index=False)

    print_position_summary(position_summary_df)
    print_yearly_summary(yearly_summary_df)
    print_pair_side_summary(pair_side_summary_df)

    print("")
    print("========================================")
    print(" CURRENT CANDIDATES")
    print("========================================")

    if candidates_df.empty:
        print("(none)")
    else:
        print(candidates_df.to_string(index=False))

    print("")
    print("========================================")
    print(" SAVED")
    print("========================================")
    print(f"- {OUT_BASE_TRADES}")
    print(f"- {OUT_POSITION_SUMMARY}")
    print(f"- {OUT_YEARLY_SUMMARY}")
    print(f"- {OUT_PAIR_SIDE_SUMMARY}")
    print(f"- {OUT_EQUITY_BY_FRACTION}")
    print(f"- {OUT_CANDIDATES}")

    print("")
    print("Done.")


if __name__ == "__main__":
    run()
