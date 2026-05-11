import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# FX SIGNAL DIAGNOSTICS v812
# =========================================================
#
# 目的:
# - v810で「候補なし」になった理由を可視化する
# - V700_RSI_PULLBACK / V800_ATR_TREND の各条件を分解してCSV出力する
#
# 実行:
#   python .\fx_v812_signal_diagnostics.py
#
# 出力:
# - fx_v812_signal_diagnostics.csv
# - fx_v812_signal_diagnostics_summary.csv
#
# =========================================================


START = "2018-01-01"
END = None

OUT_DIAGNOSTICS = "fx_v812_signal_diagnostics.csv"
OUT_SUMMARY = "fx_v812_signal_diagnostics_summary.csv"

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()


# =========================================================
# V700 SETTINGS
# =========================================================

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


# =========================================================
# V800 SETTINGS
# =========================================================

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


@dataclass
class RunSummary:
    run_datetime: str
    latest_data_date: str
    row_count: int
    pass_count: int
    near_count: int
    ng_count: int


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def round_float(value, digits: int = 6):
    if value is None or pd.isna(value):
        return np.nan

    try:
        return round(float(value), digits)
    except Exception:
        return np.nan


def join_reasons(reasons: list[str]) -> str:
    if not reasons:
        return "pass"

    return "|".join(reasons)


# =========================================================
# DATA FETCH
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

    # v811 hotfixと同じ思想:
    # start指定の長期取得だけだと最新1本が遅れることがあるため、
    # 直近10日も別取得してマージする。
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
        print(f"[DATA] {pair}: rows={len(df)}, latest={df.index[-1].date()}")

    return df


def fetch_all_data() -> dict[str, pd.DataFrame]:
    all_pairs = sorted(set(V700_PAIRS + V800_PAIRS))
    raw_data = {}

    for pair in all_pairs:
        df = fetch_pair(pair)

        if df.empty:
            continue

        raw_data[pair] = df

    return raw_data


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
# V700
# =========================================================

def make_v700_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ma"] = out["Close"].rolling(V700_MA_DAYS).mean()
    out["rsi"] = calc_rsi(out["Close"], V700_RSI_DAYS)

    out["prev_high"] = out["Close"].rolling(V700_LOOKBACK_HIGH_LOW).max().shift(1)
    out["prev_low"] = out["Close"].rolling(V700_LOOKBACK_HIGH_LOW).min().shift(1)

    out = out.dropna().copy()

    return out


def diagnose_v700_pair_side(pair: str, side: str, df: pd.DataFrame) -> dict:
    featured = make_v700_features(df)

    base = {
        "strategy": "V700_RSI_PULLBACK",
        "param_name": V700_PARAMS["param_name"],
        "pair": pair,
        "side": side,
        "allowed": (pair, side) in V700_ALLOWED_PAIR_SIDES,
        "latest_date": "",
        "close": np.nan,
        "ma": np.nan,
        "rsi": np.nan,
        "atr": np.nan,
        "prev_high": np.nan,
        "prev_low": np.nan,
        "target_price": np.nan,
        "price_margin": np.nan,
        "ma_margin": np.nan,
        "rsi_margin": np.nan,
        "trend_pass": False,
        "trigger_pass": False,
        "rsi_pass": False,
        "result": "NO_DATA",
        "reason": "no_feature_rows",
        "near_score": np.nan,
        **V700_PARAMS,
    }

    if featured.empty:
        return base

    row = featured.iloc[-1]

    latest_date = featured.index[-1].date()
    close = float(row["Close"])
    ma = float(row["ma"])
    rsi = float(row["rsi"])
    prev_high = float(row["prev_high"])
    prev_low = float(row["prev_low"])

    reasons = []

    if side == "LONG":
        target_price = prev_high * (1 - V700_PARAMS["pullback_pct"])
        trend_pass = close > ma
        trigger_pass = close <= target_price
        rsi_pass = rsi <= V700_PARAMS["rsi_long_max"]

        price_margin = close - target_price
        ma_margin = close - ma
        rsi_margin = rsi - V700_PARAMS["rsi_long_max"]

        if not trend_pass:
            reasons.append("ma_fail")
        if not trigger_pass:
            reasons.append("pullback_fail")
        if not rsi_pass:
            reasons.append("rsi_fail")

        # 小さいほど候補に近い。
        # ma_marginは、LONGでは close - ma がプラスならMA条件達成。
        near_score = (
            abs(price_margin / close)
            + max(-ma_margin / close, 0)
            + max(rsi_margin, 0) / 100
        )

    elif side == "SHORT":
        target_price = prev_low * (1 + V700_PARAMS["pullback_pct"])
        trend_pass = close < ma
        trigger_pass = close >= target_price
        rsi_pass = rsi >= V700_PARAMS["rsi_short_min"]

        price_margin = target_price - close
        ma_margin = ma - close
        rsi_margin = V700_PARAMS["rsi_short_min"] - rsi

        if not trend_pass:
            reasons.append("ma_fail")
        if not trigger_pass:
            reasons.append("pullback_fail")
        if not rsi_pass:
            reasons.append("rsi_fail")

        # 小さいほど候補に近い。
        # ma_marginは、SHORTでは ma - close がプラスならMA条件達成。
        near_score = (
            abs(price_margin / close)
            + max(-ma_margin / close, 0)
            + max(rsi_margin, 0) / 100
        )

    else:
        target_price = np.nan
        price_margin = np.nan
        ma_margin = np.nan
        rsi_margin = np.nan
        trend_pass = False
        trigger_pass = False
        rsi_pass = False
        near_score = np.nan
        reasons.append("invalid_side")

    allowed = (pair, side) in V700_ALLOWED_PAIR_SIDES

    if not allowed:
        reasons.append("not_allowed")

    passed = allowed and trend_pass and trigger_pass and rsi_pass

    if passed:
        result = "PASS"
        reason = "pass"
    else:
        result = "NG"
        reason = join_reasons(reasons)

    return {
        **base,
        "allowed": allowed,
        "latest_date": str(latest_date),
        "close": round_float(close),
        "ma": round_float(ma),
        "rsi": round_float(rsi),
        "prev_high": round_float(prev_high),
        "prev_low": round_float(prev_low),
        "target_price": round_float(target_price),
        "price_margin": round_float(price_margin),
        "ma_margin": round_float(ma_margin),
        "rsi_margin": round_float(rsi_margin),
        "trend_pass": trend_pass,
        "trigger_pass": trigger_pass,
        "rsi_pass": rsi_pass,
        "result": result,
        "reason": reason,
        "near_score": round_float(near_score, 8),
    }


def diagnose_v700(raw_data: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []

    # 実運用対象のみ診断
    for pair, side in sorted(V700_ALLOWED_PAIR_SIDES):
        df = raw_data.get(pair)

        if df is None or df.empty:
            rows.append({
                "strategy": "V700_RSI_PULLBACK",
                "param_name": V700_PARAMS["param_name"],
                "pair": pair,
                "side": side,
                "allowed": True,
                "latest_date": "",
                "result": "NO_DATA",
                "reason": "no_raw_data",
                **V700_PARAMS,
            })
            continue

        rows.append(diagnose_v700_pair_side(pair, side, df))

    return rows


# =========================================================
# V800
# =========================================================

def make_v800_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ma"] = out["Close"].rolling(V800_PARAMS["ma_days"]).mean()
    out["atr"] = calc_atr(out, V800_PARAMS["atr_days"])

    out["prev_high"] = out["High"].rolling(V800_PARAMS["breakout_days"]).max().shift(1)
    out["prev_low"] = out["Low"].rolling(V800_PARAMS["breakout_days"]).min().shift(1)

    out = out.dropna().copy()

    return out


def diagnose_v800_pair_side(pair: str, side: str, df: pd.DataFrame) -> dict:
    featured = make_v800_features(df)

    base = {
        "strategy": "V800_ATR_TREND",
        "param_name": V800_PARAMS["param_name"],
        "pair": pair,
        "side": side,
        "allowed": (pair, side) in V800_ALLOWED_PAIR_SIDES,
        "latest_date": "",
        "close": np.nan,
        "ma": np.nan,
        "rsi": np.nan,
        "atr": np.nan,
        "prev_high": np.nan,
        "prev_low": np.nan,
        "target_price": np.nan,
        "price_margin": np.nan,
        "ma_margin": np.nan,
        "rsi_margin": np.nan,
        "trend_pass": False,
        "trigger_pass": False,
        "rsi_pass": np.nan,
        "result": "NO_DATA",
        "reason": "no_feature_rows",
        "near_score": np.nan,
        **V800_PARAMS,
    }

    if featured.empty:
        return base

    row = featured.iloc[-1]

    latest_date = featured.index[-1].date()
    close = float(row["Close"])
    ma = float(row["ma"])
    atr = float(row["atr"])
    prev_high = float(row["prev_high"])
    prev_low = float(row["prev_low"])

    reasons = []

    if side == "LONG":
        target_price = prev_high
        trend_pass = close > ma
        trigger_pass = close > prev_high
        price_margin = close - prev_high
        ma_margin = close - ma

        if not trend_pass:
            reasons.append("ma_fail")
        if not trigger_pass:
            reasons.append("breakout_fail")

        # 小さいほど候補に近い。
        # ma_marginは、LONGでは close - ma がプラスならMA条件達成。
        near_score = abs(price_margin / close) + max(-ma_margin / close, 0)

    elif side == "SHORT":
        target_price = prev_low
        trend_pass = close < ma
        trigger_pass = close < prev_low
        price_margin = prev_low - close
        ma_margin = ma - close

        if not trend_pass:
            reasons.append("ma_fail")
        if not trigger_pass:
            reasons.append("breakout_fail")

        # 小さいほど候補に近い。
        # ma_marginは、SHORTでは ma - close がプラスならMA条件達成。
        near_score = abs(price_margin / close) + max(-ma_margin / close, 0)

    else:
        target_price = np.nan
        price_margin = np.nan
        ma_margin = np.nan
        trend_pass = False
        trigger_pass = False
        near_score = np.nan
        reasons.append("invalid_side")

    allowed = (pair, side) in V800_ALLOWED_PAIR_SIDES

    if not allowed:
        reasons.append("not_allowed")

    passed = allowed and trend_pass and trigger_pass

    if passed:
        result = "PASS"
        reason = "pass"
    else:
        result = "NG"
        reason = join_reasons(reasons)

    return {
        **base,
        "allowed": allowed,
        "latest_date": str(latest_date),
        "close": round_float(close),
        "ma": round_float(ma),
        "atr": round_float(atr),
        "prev_high": round_float(prev_high),
        "prev_low": round_float(prev_low),
        "target_price": round_float(target_price),
        "price_margin": round_float(price_margin),
        "ma_margin": round_float(ma_margin),
        "trend_pass": trend_pass,
        "trigger_pass": trigger_pass,
        "rsi_pass": np.nan,
        "result": result,
        "reason": reason,
        "near_score": round_float(near_score, 8),
    }


def diagnose_v800(raw_data: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []

    # 実運用対象のみ診断
    for pair, side in sorted(V800_ALLOWED_PAIR_SIDES):
        df = raw_data.get(pair)

        if df is None or df.empty:
            rows.append({
                "strategy": "V800_ATR_TREND",
                "param_name": V800_PARAMS["param_name"],
                "pair": pair,
                "side": side,
                "allowed": True,
                "latest_date": "",
                "result": "NO_DATA",
                "reason": "no_raw_data",
                **V800_PARAMS,
            })
            continue

        rows.append(diagnose_v800_pair_side(pair, side, df))

    return rows


# =========================================================
# SUMMARY / REPORT
# =========================================================

def make_summary(diagnostics_df: pd.DataFrame) -> pd.DataFrame:
    if diagnostics_df.empty:
        summary = RunSummary(
            run_datetime=now_str(),
            latest_data_date="",
            row_count=0,
            pass_count=0,
            near_count=0,
            ng_count=0,
        )
        return pd.DataFrame([asdict(summary)])

    latest_dates = diagnostics_df["latest_date"].dropna().astype(str)
    latest_dates = latest_dates[latest_dates != ""]

    if latest_dates.empty:
        latest_data_date = ""
    else:
        latest_data_date = str(latest_dates.max())

    pass_count = int((diagnostics_df["result"] == "PASS").sum())
    ng_count = int((diagnostics_df["result"] == "NG").sum())

    # near_scoreが小さいものを「近い」と見る。しきい値は雑に1%相当。
    near_mask = (
        (diagnostics_df["result"] == "NG")
        & (pd.to_numeric(diagnostics_df["near_score"], errors="coerce") <= 0.01)
    )
    near_count = int(near_mask.sum())

    summary = RunSummary(
        run_datetime=now_str(),
        latest_data_date=latest_data_date,
        row_count=int(len(diagnostics_df)),
        pass_count=pass_count,
        near_count=near_count,
        ng_count=ng_count,
    )

    return pd.DataFrame([asdict(summary)])


def print_report(diagnostics_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print("")
    print("========================================")
    print(" DIAGNOSTICS SUMMARY")
    print("========================================")
    print(summary_df.to_string(index=False))

    print("")
    print("========================================")
    print(" SIGNAL DIAGNOSTICS")
    print("========================================")

    if diagnostics_df.empty:
        print("(none)")
        return

    show_cols = [
        "strategy",
        "pair",
        "side",
        "latest_date",
        "close",
        "ma",
        "rsi",
        "atr",
        "target_price",
        "price_margin",
        "ma_margin",
        "rsi_margin",
        "trend_pass",
        "trigger_pass",
        "rsi_pass",
        "result",
        "reason",
        "near_score",
    ]

    show_cols = [c for c in show_cols if c in diagnostics_df.columns]

    out = diagnostics_df.copy()
    out = out.sort_values(
        ["result", "near_score", "strategy", "pair", "side"],
        ascending=[True, True, True, True, True],
    )

    print(out[show_cols].to_string(index=False))


# =========================================================
# DISCORD
# =========================================================

def format_discord_message(diagnostics_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "📊 FX Signal Diagnostics v812\n(no summary)"

    s = summary_df.iloc[0].to_dict()

    lines = []
    lines.append("📊 FX Signal Diagnostics v812")
    lines.append("")
    lines.append(f"run_datetime: {s.get('run_datetime')}")
    lines.append(f"latest_data_date: {s.get('latest_data_date')}")
    lines.append(f"pass_count: {s.get('pass_count')}")
    lines.append(f"near_count: {s.get('near_count')}")
    lines.append(f"ng_count: {s.get('ng_count')}")
    lines.append("")

    if diagnostics_df.empty:
        lines.append("DIAGNOSTICS: none")
    else:
        passed = diagnostics_df[diagnostics_df["result"] == "PASS"].copy()
        near = diagnostics_df[
            (diagnostics_df["result"] == "NG")
            & (pd.to_numeric(diagnostics_df["near_score"], errors="coerce") <= 0.01)
        ].copy()
        near = near.sort_values("near_score").head(5)

        if not passed.empty:
            lines.append("PASS:")
            for _, row in passed.iterrows():
                lines.append(
                    f"- {row['strategy']} {row['pair']} {row['side']} reason={row['reason']}"
                )
            lines.append("")

        if not near.empty:
            lines.append("NEAR:")
            for _, row in near.iterrows():
                lines.append(
                    f"- {row['strategy']} {row['pair']} {row['side']} "
                    f"reason={row['reason']} near={row['near_score']}"
                )
        else:
            lines.append("NEAR: none")

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
            "User-Agent": "fx-v812-signal-diagnostics",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as res:
            print(f"[DISCORD] status={res.status}")
    except Exception as e:
        print(f"[DISCORD] failed: {e}")


# =========================================================
# MAIN
# =========================================================

def run():
    print("========================================")
    print(" FX SIGNAL DIAGNOSTICS v812")
    print("========================================")
    print(f"START      : {START}")
    print(f"END        : {END}")
    print(f"OUT        : {OUT_DIAGNOSTICS}")
    print("========================================")

    raw_data = fetch_all_data()

    rows = []
    rows.extend(diagnose_v700(raw_data))
    rows.extend(diagnose_v800(raw_data))

    diagnostics_df = pd.DataFrame(rows)

    if not diagnostics_df.empty:
        diagnostics_df = diagnostics_df.sort_values(
            ["strategy", "pair", "side"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

    summary_df = make_summary(diagnostics_df)

    diagnostics_df.to_csv(OUT_DIAGNOSTICS, index=False)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print_report(diagnostics_df, summary_df)

    print("")
    print("========================================")
    print(" SAVED")
    print("========================================")
    print(f"- {OUT_DIAGNOSTICS}")
    print(f"- {OUT_SUMMARY}")

    discord_message = format_discord_message(diagnostics_df, summary_df)
    send_discord_message(discord_message)

    print("")
    print("Done.")


if __name__ == "__main__":
    run()
