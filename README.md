# Fx-Trade-System

FXの日足データを使って、売買シグナルの検証・日次判定・paper trade管理を行うためのPythonプロジェクトです。

現在は、以下の2系統を中心に検証しています。

- **V700系**: RSI + 移動平均 + 押し目/戻り売り
- **V800系**: 高値/安値ブレイク + MA200 + ATRトレーリング

実売買ではなく、まずは **paper signal / paper trade** として運用検証する前提です。

---

## 現在の運用対象

### V700_RSI_PULLBACK

RSIと移動平均を使った押し目買い・戻り売り型です。

採用候補は `B_v703_more_trades` です。

```text
対象方向:
- USDJPY=X LONG
- EURJPY=X LONG
- EURUSD=X SHORT
- GBPUSD=X SHORT

position_fraction: 1.0
```

主な固定パラメータ:

```text
pullback_pct  : 0.003
tp_pct        : 0.010
sl_pct        : 0.006
hold_days     : 9
rsi_long_max  : 40
rsi_short_min : 50
```

### V800_ATR_TREND

MA200方向へのブレイクアウトをATRトレーリングで追う型です。

```text
対象方向:
- USDJPY=X LONG
- GBPUSD=X SHORT

position_fraction: 1.0
```

主な固定パラメータ:

```text
breakout_days : 60
ma_days       : 200
atr_days      : 14
atr_mult      : 1.2
max_hold_days : 20
```

---

## 主要スクリプト

| ファイル | 役割 |
|---|---|
| `fx_v810_dual_daily_signal.py` | V700 / V800 の日次シグナル判定 |
| `fx_v811_paper_trade.py` | paper trade の注文・保有・決済・損益管理 |
| `fx_v704_v700_position_size_test.py` | V700系のポジションサイズ比較 |
| `fx_v803_position_size_test.py` | V800限定版のポジションサイズ比較 |
| `fx_v802_fixed_usdjpy_gbpusd.py` | V800限定版の固定パラメータ検証 |
| `fx_v801_trend_atr_walkforward.py` | V800系のWalk Forward検証 |
| `fx_v703_walkforward.py` | V700系のWalk Forward検証 |
| `fx_v800_trend_atr_backtest.py` | V800系の全方向ATRトレンド検証 |
| `fx_v701b_pair_side_filter.py` | V700系の通貨ペア・方向フィルタ検証 |
| `fx_v701_param_search.py` | V700系のパラメータ探索 |
| `fx_v700_backtest.py` | 初期FXバックテスト |

---

## 日次運用フロー

通常は以下の順で実行します。

```powershell
python .\fx_v810_dual_daily_signal.py
python .\fx_v811_paper_trade.py
```

### 1. `fx_v810_dual_daily_signal.py`

日足の最新データを取得し、V700 / V800 の候補を判定します。

出力:

```text
fx_v810_daily_candidates.csv
fx_v810_daily_summary.csv
```

### 2. `fx_v811_paper_trade.py`

`fx_v810_daily_candidates.csv` を読み込み、paper tradeとして管理します。

主な処理:

```text
- 新規候補を PENDING 登録
- 次足Openが取得できたら OPEN に変更
- OPENポジションを毎日チェック
- TP / SL / ATR_STOP / MA_EXIT / TIME で決済
- trades / equity / report をCSV保存
```

出力:

```text
fx_v811_positions.csv
fx_v811_trades.csv
fx_v811_equity.csv
fx_v811_daily_report.csv
```

---

## Discord通知の見方と取るべきアクション

GitHub Actionsまたはローカル実行時に、Discordへ以下のような通知が送られます。

通知は大きく分けて、`FX Daily Signal v810` と `FX Paper Trade v811` の2種類です。

---

### 1. `FX Daily Signal v810`

日次シグナル判定の結果です。

#### 候補なしの場合

例:

```text
📊 FX Daily Signal v810

CURRENT CANDIDATES:
(none)
```

取るべきアクション:

```text
何もしない。
新規のpaper注文候補はありません。
```

---

#### 候補ありの場合

例:

```text
📊 FX Daily Signal v810

CURRENT CANDIDATES:
- V800_ATR_TREND / USDJPY=X / LONG / signal_date=2026-04-30 / close=160.485992
  STOP=159.476875
```

意味:

```text
指定戦略の条件に合致したため、次の営業日のOpenでpaper entry候補になります。
```

取るべきアクション:

```text
1. 直後の v811 通知を見る。
2. v811 側で PENDING に登録されたか確認する。
3. 実売買はしない。まずはpaper上の記録だけ確認する。
```

補足:

```text
V700_RSI_PULLBACK の場合:
- TP / SL / 最大保有日数で管理されます。

V800_ATR_TREND の場合:
- ATRトレーリングストップ / MA_EXIT / 最大保有日数で管理されます。
```

---

### 2. `FX Paper Trade v811`

paper trade管理の結果です。

#### `new_pending_orders: 1` の場合

例:

```text
📊 FX Paper Trade v811

new_pending_orders: 1
opened_positions: 0
closed_trades: 0
open_positions: 0
pending_positions: 1

ACTIVE:
PENDING V800_ATR_TREND USDJPY=X LONG signal=2026-04-30
```

意味:

```text
新しいシグナルをPENDING注文として登録しました。
まだentryはしていません。
次の営業日のOpenデータ待ちです。
```

取るべきアクション:

```text
1. fx_v811_positions.csv に PENDING が登録されているか確認する。
2. 翌営業日以降の自動実行で OPEN に変わるか確認する。
3. 同じ日に再実行しても重複登録されないことを確認する。
```

---

#### `new_pending_orders: 0` かつ `pending_positions: 1` の場合

例:

```text
new_pending_orders: 0
opened_positions: 0
closed_trades: 0
open_positions: 0
pending_positions: 1

ACTIVE:
PENDING V800_ATR_TREND USDJPY=X LONG signal=2026-04-30
```

意味:

```text
既に同じPENDINGが登録済みです。
重複登録を防止しています。
```

取るべきアクション:

```text
正常です。
何もしなくてよいです。
次の営業日のOpenデータ取得を待ちます。
```

---

#### `opened_positions: 1` の場合

例:

```text
opened_positions: 1
open_positions: 1
pending_positions: 0

OPENED:
- V800_ATR_TREND USDJPY=X LONG entry=2026-05-01 price=160.70
```

意味:

```text
PENDINGだったpaper注文が、次の営業日のOpenでOPENポジションになりました。
```

取るべきアクション:

```text
1. fx_v811_positions.csv の status が OPEN になっているか確認する。
2. entry_date / entry_price / current_stop を確認する。
3. 実売買はせず、paper管理が想定通り動いているか見る。
```

---

#### `open_positions: 1` 以上の場合

例:

```text
open_positions: 1

ACTIVE:
OPEN V800_ATR_TREND USDJPY=X LONG entry=2026-05-01 price=160.70 stop=159.80
```

意味:

```text
paper上で保有中のポジションがあります。
```

取るべきアクション:

```text
1. stop / tp / sl の値を確認する。
2. V800の場合は current_stop が日々更新されるか確認する。
3. 決済通知が出るまで見守る。
```

見るポイント:

```text
V700:
- tp_price
- sl_price
- hold_days

V800:
- current_stop
- highest_high
- lowest_low
- max_hold_days
```

---

#### `closed_trades: 1` 以上の場合

例:

```text
closed_trades: 1

CLOSED:
- V800_ATR_TREND USDJPY=X LONG exit=2026-05-10 ATR_STOP pnl=1200.50
```

意味:

```text
paper上のポジションが決済されました。
```

取るべきアクション:

```text
1. fx_v811_trades.csv を確認する。
2. exit_reason を確認する。
3. net_return / pnl / equity_after を確認する。
4. fx_v811_equity.csv に資産推移が記録されているか確認する。
```

exit_reason の意味:

| exit_reason | 意味 | アクション |
|---|---|---|
| `TP` | V700で利確到達 | 正常。利益記録を確認 |
| `SL` | V700で損切り到達 | 正常。損失幅が想定内か確認 |
| `ATR_STOP` | V800でATRトレーリングストップ到達 | 正常。trend終了扱い |
| `MA_EXIT` | V800でMAを割り込んだ/上抜けた | 正常。trend崩れ扱い |
| `TIME` | 最大保有日数に到達 | 正常。時間切れ決済 |

---

#### `equity` が変化した場合

例:

```text
equity: 101200.5
```

意味:

```text
決済済みpaper tradeの損益が反映されています。
未決済ポジションの含み損益は基本的に反映していません。
```

取るべきアクション:

```text
1. fx_v811_trades.csv で直近の決済を確認する。
2. fx_v811_equity.csv で推移を確認する。
3. equity_after が連続して記録されているか確認する。
```

---

## 通知別チェックリスト

| 通知内容 | 状態 | やること |
|---|---|---|
| `CURRENT CANDIDATES: (none)` | 新規候補なし | 何もしない |
| `CURRENT CANDIDATES:` に候補あり | 新規シグナルあり | v811通知でPENDING登録を確認 |
| `new_pending_orders: 1` | 新規PENDING登録 | positions.csv確認 |
| `new_pending_orders: 0` + `pending_positions: 1` | 既存PENDINGあり | 正常。重複防止。待つ |
| `opened_positions: 1` | paper entry完了 | entry_price / stop を確認 |
| `open_positions: 1` | paper保有中 | stop更新・決済待ち |
| `closed_trades: 1` | paper決済あり | trades.csv / equity.csv確認 |
| `DISCORD skipped` | Webhook未設定 | 必要ならSecretを設定 |
| GitHub Actions失敗 | 自動実行失敗 | Actionsログを確認 |

---

## GitHub Actionsでの自動実行

`.github/workflows/` 配下のWorkflowで、日次実行を想定しています。

想定フロー:

```text
1. Pythonセットアップ
2. pandas / numpy / yfinance をインストール
3. fx_v810_dual_daily_signal.py を実行
4. fx_v811_paper_trade.py を実行
5. 生成CSVをコミット
6. Discord Webhookがあれば通知
```

Discord通知を使う場合は、GitHub Repository Secret に以下を設定します。

```text
DISCORD_WEBHOOK_URL
```

未設定でも処理自体は動きます。

---

## 検証結果メモ

### V800限定版 / `fx_v803_position_size_test.py`

`USDJPY=X LONG` と `GBPUSD=X SHORT` のみを対象にしたATRトレンド型です。

`position_fraction = 1.0` の結果:

```text
trades              : 43
profit_factor       : 1.893707
final_equity        : 113686.22
total_equity_return : +13.69%
max_drawdown        : -2.03%
```

特徴:

```text
- 取引回数は少なめ
- PFとDDは比較的良好
- 近年の成績はやや弱い
```

### V700 B / `fx_v704_v700_position_size_test.py`

`B_v703_more_trades` の `position_fraction = 1.0` の結果:

```text
trades              : 174
profit_factor       : 1.301939
final_equity        : 117722.80
total_equity_return : +17.72%
max_drawdown        : -4.75%
```

特徴:

```text
- 取引回数が多い
- リターンは大きめ
- DDはV800限定版より大きい
```

### V700 A / `fx_v704_v700_position_size_test.py`

`A_v703_stable` の `position_fraction = 1.0` の結果:

```text
trades              : 26
profit_factor       : 2.616993
final_equity        : 109465.43
total_equity_return : +9.47%
max_drawdown        : -1.21%
```

特徴:

```text
- 取引回数はかなり少ない
- PFとDDは優秀
- サブ戦略候補
```

---

## 現在の判断

```text
主力候補:
- V800_ATR_TREND 限定版
- V700_RSI_PULLBACK B_v703_more_trades

守り候補:
- V700_RSI_PULLBACK A_v703_stable

現在の方針:
- 実売買ではなくpaper tradeで挙動確認
- GitHub Actionsで日次自動実行
- Discordへ候補・保有・決済を通知
```

---

## セットアップ

### 必要パッケージ

```powershell
pip install pandas numpy yfinance
```

### 構文チェック

```powershell
python -m py_compile .\fx_v810_dual_daily_signal.py
python -m py_compile .\fx_v811_paper_trade.py
```

### 日次実行

```powershell
python .\fx_v810_dual_daily_signal.py
python .\fx_v811_paper_trade.py
```

---

## 注意事項

このプロジェクトは、FX売買ロジックの検証とpaper trade管理を目的としたものです。

```text
- 投資助言ではありません
- 実売買を推奨するものではありません
- バックテスト結果は将来の利益を保証しません
- yfinanceのデータ欠損・遅延・仕様変更の影響を受けます
- スプレッド、スリッページ、約定拒否、急変動リスクは実運用では別途考慮が必要です
```

まずはpaper運用で、候補・約定・決済・Discord通知・CSV更新が想定通り動くかを確認します。
