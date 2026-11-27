# Trading sentiment and next-bar prices

This project ranks five Hong Kong stocks and five Korean stocks by social mentions and sentiment. It estimates each stock's next observed 30-minute close from market bars, social posts, and news headlines collected over 10 calendar days.

## Run locally

Requires Python 3.11 or newer. From this directory, run the offline demo first.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m tzuroni demo
```

Open `outputs/demo/report.html`. The offline demo uses seeded synthetic data and runs with no provider accounts. Its metrics show that the pipeline runs end to end. The [GitHub Actions workflow](https://github.com/Shaunakrane914/TZURONI-LTD/actions/workflows/verify.yml) also tests the pipeline and packages the HTML and CSVs as the `synthetic-demo-report` artifact. On macOS or Linux, use `.venv/bin/python` for the commands below.

Across 600 synthetic holdout observations, mean absolute error was 34.38 bps for the flat-price forecast, 35.02 bps for the price-only model, and 35.14 bps for the model with sentiment. The synthetic run verifies the evaluation flow. Live data is required to measure whether sentiment improves the forecast.

For a live run, copy `.env.example` to `.env`. Set `APIFY_TOKEN` from [Apify Integrations](https://console.apify.com/settings/integrations) and `EXA_API_KEY` from the [Exa dashboard](https://dashboard.exa.ai/api-keys). Use the existing Apify actors listed below. A fresh run starts up to five Apify runs with the indicated USD cap on each and makes up to 10 Exa searches. At `$0.75` per Apify actor run, the Apify cap is `$3.75` before any explicitly retried failed runs. Exa bills separately. Provider limits can reduce coverage.

```powershell
.\.venv\Scripts\python -m tzuroni live --apify-budget 0.75
.\.venv\Scripts\python -m tzuroni replay outputs/live/snapshot
```

Resume an interrupted run with `python -m tzuroni live --resume --apify-budget 0.75`. It keeps the original time window, reuses completed stages and reconnects to known Apify runs. If Apify could not confirm startup, check the Apify Runs page before starting a fresh collection under a different `--out` directory. A hard process kill can leave `.live.lock` in the output directory. Remove that lock only after checking that the original process has stopped.

The report includes per-exchange mention and sentiment rankings, social-source coverage for the full window and at forecast time, the latest available close, rolling 20-bar support and resistance, and the forecast. `metrics.csv` shows per-exchange and pooled error, including MAE improvement versus a flat-price baseline. `backtest.csv` records the actual and predicted prices and returns, fold boundaries, and the latest training-label timestamp. The local snapshot contains normalized inputs and Apify run, actor, and build IDs. Keep `.env`, raw live snapshots, and collection checkpoints local.

## Sources

| Market | Forum | Telegram channel |
|---|---|---|
| HKEX | [LIHKG](https://apify.com/claude_code_reviewer/lihkg-scraper-en), actor ID `PJDFRxJdpPknN9nYn` | [StockIntelligence](https://t.me/s/StockIntelligence) |
| KRX | [Naver Stock Discussion](https://apify.com/hgservices/naver-stock-discussion-scraper), actor ID `xXFbG2oX6UpaPtcTC` | [FastStockNews](https://t.me/s/FastStockNews) |

Both Telegram collections use the [Telegram Channel Scraper](https://apify.com/nexuscortex/telegram-channel-scraper), actor ID `WS2JAAECSV1ajF5rR`. News comes from [Exa Search](https://exa.ai/docs/reference/search) and 30-minute bars come from [yfinance](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html). Search terms cover ticker symbols, cashtags, company names, and local names where applicable. The LIHKG collection fetches the original text for up to eight matching threads per ticker across different days and uses the title when a body is unavailable. Posts are deduplicated by source and ID. Missing engagement fields stay empty.

## Method and limits

- Normalize timestamps to UTC. At each completed bar, summarize the preceding 24 hours of text by source. A small English, Chinese, and Korean finance lexicon assigns polarity. Posts without a recognized term are counted as mentions and excluded from average sentiment. Engagement metadata stays in the snapshot because it may reflect counts observed after the forecast cutoff.
- Fit a ridge regression on next-bar log returns using price and text features. Compare it to ridge regression using price features and a flat-price forecast. Three expanding folds hold out whole UTC bar timestamps across all tickers. They drop training rows whose target price lands at or beyond a test fold's first cutoff. Fit scaling on each training partition and report MAE, RMSE, and directional accuracy.
- The target is the next bar available from the provider, so it can cross a session boundary. HKEX samples use continuous trading bars and exclude the closing auction. Yahoo may omit late KRX bars. The report displays the bar timestamp so stale closes are visible.
- Publication-time backtesting approximates historical availability. Search indexes can discover articles late, text can change, and query caps can miss posts. Telegram rows flagged as edited are excluded. A 10-day window is suitable for an internship exercise and offers limited evidence about performance across market conditions.
