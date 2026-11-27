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

Across 600 synthetic holdout observations, mean absolute error was 34.38 bps for the flat-price forecast, 35.02 bps for the price-only model, and 35.14 bps for the model with sentiment. The synthetic run verifies the evaluation flow. See the live results below for predictive performance on the collected market snapshot.

For a live run, copy `.env.example` to `.env`. Set `APIFY_TOKEN` from [Apify Integrations](https://console.apify.com/settings/integrations) and `EXA_API_KEY` from the [Exa dashboard](https://dashboard.exa.ai/api-keys). Use the existing Apify actors listed below. A fresh run starts up to five Apify runs with the indicated USD cap on each and makes up to 10 Exa searches. At `$0.75` per Apify actor run, the Apify cap is `$3.75` before any explicitly retried failed runs. Exa bills separately. Provider limits can reduce coverage.

```powershell
.\.venv\Scripts\python -m tzuroni live --apify-budget 0.75
.\.venv\Scripts\python -m tzuroni replay outputs/live/snapshot
```

Resume an interrupted run with `python -m tzuroni live --resume --apify-budget 0.75`. It keeps the original time window, reuses completed stages and reconnects to known Apify runs. If Apify could not confirm startup, check the Apify Runs page before starting a fresh collection under a different `--out` directory. A hard process kill can leave `.live.lock` in the output directory. Remove that lock only after checking that the original process has stopped.

To supplement a saved live snapshot, run `backfill-naver`. It collects Naver Blog posts in three-day shards within the same window and writes a new snapshot. The command accepts `--resume` and caps each shard separately. The optional semantic scorer runs locally using a pinned multilingual MiniLM encoder and fixed financial examples. Its first use downloads the tokenizer and model weights.

```powershell
.\.venv\Scripts\python -m pip install -r requirements-semantic.txt
.\.venv\Scripts\python -m tzuroni backfill-naver outputs/live/snapshot --apify-budget 0.35
.\.venv\Scripts\python -m tzuroni replay outputs/historical/snapshot --sentiment-model semantic --out outputs/submission
```

The report includes per-exchange mention and sentiment rankings, social-source coverage for the full window and at forecast time, the latest available close, rolling 20-bar support and resistance, and the forecast. `metrics.csv` shows per-exchange and pooled error, including MAE improvement versus a flat-price baseline. `backtest.csv` records actual and predicted prices and returns, fold boundaries, and the latest training-label timestamp. `news.csv` lists matched Exa articles, their source URLs, publication timestamps, excerpts, and headline scores for review. The local snapshot contains normalized inputs and Apify run, actor, and build IDs. Keep `.env`, raw live snapshots, and collection checkpoints local.

## Live results

The November 17–27, 2026 UTC snapshot includes ten stocks and 575 held-out observations. The Naver backfill added 167 timestamped posts across all five Korean tickers. Korean rows with both social feeds increased from 0 of 295 to 60 of 295 (20.3%). Hong Kong coverage remained 13 of 280 (4.6%). These counts require a matched forum post and a matched Telegram post in the preceding 24 hours. Apify reported $2.56 in total usage, including collection retries and backfill.

| Model | Finance lexicon MAE (bps) | Semantic sentiment MAE (bps) |
|---|---:|---:|
| Flat price | 69.14 | 69.14 |
| Price only | 70.84 | 70.84 |
| Price + social | 72.86 | 72.07 |
| Price + news | 70.58 | 72.00 |
| Price + social + news | 72.58 | 73.01 |

Both columns use the same enriched snapshot and holdout rows. The flat-price forecast had the lowest pooled MAE. Before the backfill, the original lexicon-based combined model had 72.08 bps MAE. Temporal coverage remains sparse, especially for Hong Kong, and the short window limits conclusions about predictive value.

The standalone report shows the semantic run, holdout coverage, and diagnostic charts. It contains aggregate results and charts and omits raw posts, URLs, run identifiers, and individual price levels. The real-data snapshot was replayed locally. GitHub CI runs the offline synthetic replay. [Download the sanitized live HTML report](docs/live-report.html?raw=1)

## Sources

| Market | Forum | Telegram channel |
|---|---|---|
| HKEX | [LIHKG](https://apify.com/claude_code_reviewer/lihkg-scraper-en), actor ID `PJDFRxJdpPknN9nYn` | [StockIntelligence](https://t.me/s/StockIntelligence) |
| KRX | [Naver Stock Discussion](https://apify.com/hgservices/naver-stock-discussion-scraper), actor ID `xXFbG2oX6UpaPtcTC` | [FastStockNews](https://t.me/s/FastStockNews) |

Both Telegram collections use the [Telegram Channel Scraper](https://apify.com/nexuscortex/telegram-channel-scraper), actor ID `WS2JAAECSV1ajF5rR`. News comes from [Exa Search](https://exa.ai/docs/reference/search) and 30-minute bars come from [yfinance](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html). Search terms cover ticker symbols, cashtags, company names, and local names where applicable. The LIHKG collection fetches the original text for up to eight matching threads per ticker across different days and uses the title when a body is unavailable. Posts are deduplicated by source and ID. Missing engagement fields stay empty.

The Exa Python SDK reads `EXA_API_KEY` from the environment after the live command loads `.env`. Each ticker search uses a company-specific query, a financial-news objective, article highlights, and the collection's 10-day publication window. Article highlights are for review because a page may have been updated after its published timestamp. The model uses the headline for its publication-time news feature.

The Naver Blog actor complements Naver Stock Discussion on KRX. The pipeline requires a time and an identifiable asset match for each retained post, and combines the two Naver feeds into one forum feature group. It keeps the blog provenance in the private snapshot. [Naver Blog actor documentation](https://apify.com/webdata_labs/naver-blog-cafe-scraper)

## Method and limits

The semantic option uses a pinned ONNX release of the multilingual MiniLM sentence encoder. [Model and revision](https://huggingface.co/onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX/tree/d4c06bf0d7680171ac30042a1387e1fdb7a90021)

- The report counts explicit first-person long and short disclosures in English, Chinese, and Korean social posts. Counts refer to posts and holdings are self-reported.
- Source counts in the rankings are for the whole collection window. The holdout coverage table shows how many evaluated bars actually had items available in the preceding 24 hours. Naver returns the newest posts first, so a capped collection can miss earlier bars.
- Normalize timestamps to UTC. At each completed bar, summarize the preceding 24 hours of text by source. The lexicon baseline recognizes English, Chinese, and Korean financial terms. The semantic option compares sentence embeddings with fixed bullish, bearish, and neutral examples in those three languages, using cosine similarity. Its scores have limited calibration on financial forums. Engagement metadata stays in the snapshot because it may reflect counts observed after the forecast cutoff.
- Fit ridge regressions using price only, price plus social sentiment, price plus news sentiment, and price plus both. Compare them with a flat-price forecast on the same next-bar returns. Three expanding folds hold out whole UTC bar timestamps across all tickers. They drop training rows whose target price lands at or beyond a test fold's first cutoff. Fit scaling on each training partition and report MAE, RMSE, and directional accuracy.
- The target is the next bar available from the provider, so it can cross a session boundary. HKEX samples use continuous trading bars and exclude the closing auction. Yahoo may omit late KRX bars. The report displays the bar timestamp so stale closes are visible.
- Publication-time backtesting approximates historical availability. Search indexes can discover articles late, text can change, and query caps can miss posts. Telegram rows flagged as edited are excluded. A 10-day window is suitable for an internship exercise and offers limited evidence about performance across market conditions.
