import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# FX DAILY SIGNAL v810
# =========================================================
#
# 目的:
# - V700系 B_v703_more_trades と V800限定版を同時に日次判定する
# - 実売買ではなく、まずは paper signal 用
#
# 実行:
#   python .\fx_v810_dual_daily_signal.py
#
# 出力:
# - fx_v810_daily_candidates.csv
# - fx_v810_daily_summary.csv
#
# Discord通知:
# - 環境変数 DISCORD_WEBHOOK_URL があれば通知
# - なければCSV/コンソール出力のみ
#
# =========================================================


# =========================================================
# BASIC SETTINGS
# =========================================================

START = "2018-01-01"
END = None

OUT_CANDIDATES = "fx_v810_daily_candidates.csv"
OUT_SUMMARY = "fx_v810_daily_summary.csv"

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()


# =========================================================
# STRATEGY SETTINGS
# =========================================================

# -------------------------
# V700 B: RSI + MA + 押し目/戻り売り
# -------------------------

V700_PAIRS = [
    "USDJPY=X",
    "EURJPY=X",
    "EURUSD=X",
    "GBPUSD=X",
]

V700_ALLOWED_PAIR_SIDES = {
    ("USDJPY=X", "LONG"),
    ("EURJPY=X", "LONG"),
    ("EURUSD=X", "SHORT"),
    ("GBPUSD=X", "SHORT"),
}

V700_RSI_DAYS = 14
V700_MA_DAYS = 25
V700_LOOKBACK_HIGH_LOW = 20

V700_PARAMS = {
    "param_name": "B_v703_more_trades",
    "pullback_pct": 0.003,
    "tp_pct": 0.010,
    "sl_pct": 0.006,
    "hold_days": 9,
    "rsi_long_max": 40,
    "rsi_short_min": 50,
    "position_fraction": 1.0,
}


# -------------------------
# V800: USDJPY LONG / GBPUSD SHORT 限定 ATRトレンド
# -------------------------

V800_PAIRS = [
    "USDJPY=X",
    "GBPUSD=X",
]

V800_ALLOWED_PAIR_SIDES = {
    ("USDJPY=X", "LONG"),
    ("GBPUSD=X", "SHORT"),
}

V800_PARAMS = {
    "param_name": "V800_fixed_usdjpy_gbpusd",
    "breakout_days": 60,
    "ma_days": 200,
    "atr_days": 14,
    "atr_mult": 1.2,
    "max_hold_days": 20,
    "position_fraction": 1.0,
}


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class RunInfo:
    run_datetime: str
    start: str
    end: str | None
    strategies: str
    candidate_count: int


# =========================================================
# UTILS
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


def calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()

    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


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


def is_allowed(pair: str, side: str, allowed_pair_sides: set[tuple[str, str]]) -> bool:
    return (pair, side) in allowed_pair_sides


def round_float(value, digits: int = 6):
    if pd.isna(value):
        return np.nan

    return round(float(value), digits)


# =========================================================
# V700 SIGNAL
# =========================================================

def make_v700_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ma"] = out["Close"].rolling(V700_MA_DAYS).mean()
    out["rsi"] = calc_rsi(out["Close"], V700_RSI_DAYS)

    out["prev_high"] = out["Close"].rolling(V700_LOOKBACK_HIGH_LOW).max().shift(1)
    out["prev_low"] = out["Close"].rolling(V700_LOOKBACK_HIGH_LOW).min().shift(1)

    out = out.dropna().copy()

    return out


def detect_v700_candidate(pair: str, df: pd.DataFrame) -> dict | None:
    featured = make_v700_features(df)

    if featured.empty:
        return None

    row = featured.iloc[-1]

    signal_date = featured.index[-1].date()
    close = float(row["Close"])
    ma = float(row["ma"])
    rsi = float(row["rsi"])
    prev_high = float(row["prev_high"])
    prev_low = float(row["prev_low"])

    side = None

    long_signal = (
        (close > ma)
        and (close <= prev_high * (1 - V700_PARAMS["pullback_pct"]))
        and (rsi <= V700_PARAMS["rsi_long_max"])
    )

    short_signal = (
        (close < ma)
        and (close >= prev_low * (1 + V700_PARAMS["pullback_pct"]))
        and (rsi >= V700_PARAMS["rsi_short_min"])
    )

    if long_signal:
        side = "LONG"
    elif short_signal:
        side = "SHORT"

    if side is None:
        return None

    if not is_allowed(pair, side, V700_ALLOWED_PAIR_SIDES):
        return None

    if side == "LONG":
        tp_ref = close * (1 + V700_PARAMS["tp_pct"])
        sl_ref = close * (1 - V700_PARAMS["sl_pct"])
    else:
        tp_ref = close * (1 - V700_PARAMS["tp_pct"])
        sl_ref = close * (1 + V700_PARAMS["sl_pct"])

    return {
        "strategy": "V700_RSI_PULLBACK",
        "param_name": V700_PARAMS["param_name"],
        "signal_date": signal_date,
        "pair": pair,
        "side": side,
        "close": round_float(close),
        "ma": round_float(ma),
        "rsi": round_float(rsi),
        "prev_high": round_float(prev_high),
        "prev_low": round_float(prev_low),
        "tp_ref": round_float(tp_ref),
        "sl_ref": round_float(sl_ref),
        "stop_ref": np.nan,
        "entry_rule": "next open",
        "exit_rule": f"TP {V700_PARAMS['tp_pct']:.3f} / SL {V700_PARAMS['sl_pct']:.3f} / hold {V700_PARAMS['hold_days']} days",
        "position_fraction": V700_PARAMS["position_fraction"],
        "note": "paper signal only",
        **V700_PARAMS,
    }


# =========================================================
# V800 SIGNAL
# =========================================================

def make_v800_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ma"] = out["Close"].rolling(V800_PARAMS["ma_days"]).mean()
    out["atr"] = calc_atr(out, V800_PARAMS["atr_days"])

    out["prev_high"] = out["High"].rolling(V800_PARAMS["breakout_days"]).max().shift(1)
    out["prev_low"] = out["Low"].rolling(V800_PARAMS["breakout_days"]).min().shift(1)

    out = out.dropna().copy()

    return out


def detect_v800_candidate(pair: str, df: pd.DataFrame) -> dict | None:
    featured = make_v800_features(df)

    if featured.empty:
        return None

    row = featured.iloc[-1]

    signal_date = featured.index[-1].date()
    close = float(row["Close"])
    ma = float(row["ma"])
    atr = float(row["atr"])
    prev_high = float(row["prev_high"])
    prev_low = float(row["prev_low"])

    side = None

    long_signal = (
        (close > ma)
        and (close > prev_high)
    )

    short_signal = (
        (close < ma)
        and (close < prev_low)
    )

    if long_signal:
        side = "LONG"
    elif short_signal:
        side = "SHORT"

    if side is None:
        return None

    if not is_allowed(pair, side, V800_ALLOWED_PAIR_SIDES):
        return None

    if side == "LONG":
        stop_ref = close - atr * V800_PARAMS["atr_mult"]
    else:
        stop_ref = close + atr * V800_PARAMS["atr_mult"]

    return {
        "strategy": "V800_ATR_TREND",
        "param_name": V800_PARAMS["param_name"],
        "signal_date": signal_date,
        "pair": pair,
        "side": side,
        "close": round_float(close),
        "ma": round_float(ma),
        "rsi": np.nan,
        "atr": round_float(atr),
        "prev_high": round_float(prev_high),
        "prev_low": round_float(prev_low),
        "tp_ref": np.nan,
        "sl_ref": np.nan,
        "stop_ref": round_float(stop_ref),
        "entry_rule": "next open",
        "exit_rule": f"ATR trailing {V800_PARAMS['atr_mult']} / max hold {V800_PARAMS['max_hold_days']} days",
        "position_fraction": V800_PARAMS["position_fraction"],
        "note": "paper signal only",
        **V800_PARAMS,
    }


# =========================================================
# DISCORD
# =========================================================

def format_discord_message(candidates_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("📊 FX Daily Signal v810")
    lines.append("")
    lines.append(f"run_datetime: {today}")
    lines.append("strategies: V700_RSI_PULLBACK + V800_ATR_TREND")
    lines.append("mode: paper signal")
    lines.append("")

    if candidates_df.empty:
        lines.append("CURRENT CANDIDATES:")
        lines.append("(none)")
    else:
        lines.append("CURRENT CANDIDATES:")

        for _, row in candidates_df.iterrows():
            lines.append(
                f"- {row['strategy']} / {row['pair']} / {row['side']} "
                f"/ signal_date={row['signal_date']} / close={row['close']}"
            )

            if not pd.isna(row.get("tp_ref", np.nan)):
                lines.append(
                    f"  TP={row.get('tp_ref')} SL={row.get('sl_ref')}"
                )

            if not pd.isna(row.get("stop_ref", np.nan)):
                lines.append(
                    f"  STOP={row.get('stop_ref')}"
                )

    message = "\n".join(lines)

    # Discord 2000文字制限対策
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
            "User-Agent": "fx-v810-daily-signal",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as res:
            status = res.status
            print(f"[DISCORD] status={status}")
    except Exception as e:
        print(f"[DISCORD] failed: {e}")


# =========================================================
# MAIN
# =========================================================

def run():
    print("========================================")
    print(" FX DAILY SIGNAL v810")
    print("========================================")
    print(f"START             : {START}")
    print(f"END               : {END}")
    print("STRATEGIES        : V700_RSI_PULLBACK + V800_ATR_TREND")
    print("MODE              : paper signal")
    print("========================================")

    all_pairs = sorted(set(V700_PAIRS + V800_PAIRS))
    raw_data = {}

    for pair in all_pairs:
        df = fetch_pair(pair)

        if df.empty:
            continue

        raw_data[pair] = df
        print(f"[DATA] {pair}: rows={len(df)}, latest={df.index[-1].date()}")

    candidates = []

    print("")
    print("========================================")
    print(" CHECK V700")
    print("========================================")

    for pair in V700_PAIRS:
        df = raw_data.get(pair)

        if df is None or df.empty:
            continue

        candidate = detect_v700_candidate(pair, df)

        if candidate is not None:
            candidates.append(candidate)
            print(f"[V700 CANDIDATE] {pair} {candidate['side']}")
        else:
            print(f"[V700] {pair}: none")

    print("")
    print("========================================")
    print(" CHECK V800")
    print("========================================")

    for pair in V800_PAIRS:
        df = raw_data.get(pair)

        if df is None or df.empty:
            continue

        candidate = detect_v800_candidate(pair, df)

        if candidate is not None:
            candidates.append(candidate)
            print(f"[V800 CANDIDATE] {pair} {candidate['side']}")
        else:
            print(f"[V800] {pair}: none")

    if candidates:
        candidates_df = pd.DataFrame(candidates)
        candidates_df = candidates_df.sort_values(
            ["strategy", "pair", "side"]
        ).reset_index(drop=True)
    else:
        candidates_df = pd.DataFrame(columns=[
            "strategy",
            "param_name",
            "signal_date",
            "pair",
            "side",
            "close",
            "ma",
            "rsi",
            "atr",
            "prev_high",
            "prev_low",
            "tp_ref",
            "sl_ref",
            "stop_ref",
            "entry_rule",
            "exit_rule",
            "position_fraction",
            "note",
        ])

    run_info = RunInfo(
        run_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        start=START,
        end=END,
        strategies="V700_RSI_PULLBACK,V800_ATR_TREND",
        candidate_count=len(candidates_df),
    )

    summary_df = pd.DataFrame([asdict(run_info)])

    candidates_df.to_csv(OUT_CANDIDATES, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)

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
    print(f"- {OUT_CANDIDATES}")
    print(f"- {OUT_SUMMARY}")

    discord_message = format_discord_message(candidates_df, summary_df)
    send_discord_message(discord_message)

    print("")
    print("Done.")


if __name__ == "__main__":
    run()
