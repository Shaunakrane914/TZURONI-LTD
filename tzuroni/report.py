from html import escape

import pandas as pd


def render_report(ranking, metrics, backtest, start, end, mode):
    decimals = dict.fromkeys(['current_price', 'support', 'resistance', 'next_bar_price', 'next_bar_return_pct'], 2)
    display = ranking.round({**decimals, 'sentiment_score': 3}).copy()
    display['bar_end_utc'] = display['bar_end_utc'].dt.strftime('%Y-%m-%d %H:%M')
    display['sentiment_rank'] = display['sentiment_rank'].astype('Int64').astype('string').fillna('—')
    display = display.rename(columns={
        'ticker': 'Ticker', 'asset_name': 'Asset', 'currency': 'Currency', 'bar_end_utc': 'Bar end (UTC)',
        'current_price': 'Last close', 'social_mentions': 'Social mentions', 'scored_mentions': 'Scored',
        'social_sources': 'Social sources (10 days)', 'social_sources_at_bar': 'Social sources at forecast',
        'forum_source': 'Forum', 'forum_mentions': 'Forum posts', 'telegram_mentions': 'Telegram posts',
        'news_mentions': 'News', 'sentiment_score': 'Sentiment', 'sentiment_direction': 'Sentiment direction',
        'sentiment_rank': 'Sentiment rank',
        'support': 'Support', 'resistance': 'Resistance', 'next_bar_price': 'Next-bar price',
        'long_position_posts': 'Long disclosures', 'short_position_posts': 'Short disclosures',
        'next_bar_return_pct': 'Forecast change (%)',
    })
    sections = []
    for market, rows in display.groupby('exchange', sort=False):
        maximum = max(rows['Social mentions'].max(), 1)
        bars = []
        for _, row in rows.iterrows():
            count = row['Social mentions']
            sentiment = row['Sentiment']
            tone = row['Sentiment direction']
            score = '—' if pd.isna(sentiment) else f'{sentiment:+.2f}'
            bars.append(f'<li><span>{escape(row["Ticker"])}</span>'
                        f'<meter min="0" max="{maximum}" value="{count}">{count}</meter>'
                        f'<span>{count}</span><span class="{tone}">{score} {tone}</span></li>')
        table = rows.drop(columns=['exchange', 'Forum']).to_html(index=False, border=0, na_rep='—')
        forum = escape(rows['Forum'].iloc[0])
        two_sources = rows['Social sources (10 days)'].eq(2).sum()
        no_recent = rows['Social sources at forecast'].eq(0).sum()
        sections.append(f'<section><h2>{market}</h2><p>{two_sources} of {len(rows)} tickers had posts from both '
                        f'{forum} and Telegram. {no_recent} had no social posts in the 24 hours before their forecast.</p>'
                        f'<ul aria-label="Social mention counts and sentiment">{"".join(bars)}</ul>'
                        f'<div class="scroll">{table}</div></section>')

    metrics_table = metrics.assign(directional_accuracy_pct=metrics['directional_accuracy'] * 100)
    metrics_table = metrics_table.drop(columns='directional_accuracy').round(2).rename(columns={
        'exchange': 'Exchange', 'model': 'Model', 'observations': 'Observations',
        'mae_bps': 'MAE (bps)', 'rmse_bps': 'RMSE (bps)',
        'mae_improvement_vs_flat_pct': 'MAE improvement vs flat (%)',
        'mae_improvement_vs_price_only_pct': 'MAE improvement vs price-only (%)',
        'directional_accuracy_pct': 'Directional accuracy (%)',
    }).to_html(index=False, border=0, na_rep='—')
    coverage = backtest.groupby('exchange', sort=False).agg(**{
        'Held-out rows': ('ticker', 'size'),
        'Rows with forum posts': ('forum_available', 'sum'),
        'Rows with Telegram posts': ('telegram_available', 'sum'),
        'Rows with both social sources': ('both_social_available', 'sum'),
        'Rows with news': ('news_available', 'sum'),
    })
    coverage_table = coverage.sort_index().rename_axis('Exchange').reset_index().to_html(index=False, border=0)

    note = ('Synthetic demonstration. Prices and scores are generated for verification.' if mode == 'demo'
            else 'Live-source snapshot. Source coverage and the share of scored posts vary by asset.')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market sentiment and next-bar forecasts</title>
<style>
body {{ background: #f4f7fa; color: #172431; font: 16px/1.5 system-ui, sans-serif; margin: 0; }}
main {{ max-width: 1350px; margin: auto; padding: clamp(14px, 4vw, 50px); }}
h1 {{ font-size: clamp(1.6rem, 4vw, 2.75rem); line-height: 1.15; margin-bottom: 8px; }}
h2 {{ margin: 0; }} p {{ margin: 6px 0 14px; }}
section {{ background: white; border: 1px solid #dbe3ea; border-radius: 12px; margin-top: 22px; padding: clamp(12px, 2vw, 24px); }}
ul {{ padding: 0; max-width: 550px; list-style: none; }}
li {{ display: grid; grid-template-columns: 5.4em minmax(3em, 1fr) 2.9em 7.7em; align-items: center; gap: 7px; font-size: .9rem; }}
meter {{ width: 100%; accent-color: #287c89; }}
.bullish {{ color: #166b46; }} .bearish {{ color: #a12932; }} .unscored {{ color: #546376; }}
.scroll {{ overflow: auto; }} table {{ border-collapse: collapse; width: 100%; font-size: .86rem; }}
th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #e6ecf0; white-space: nowrap; }}
th {{ background: #eaf1f7; }}
</style></head>
<body><main><header><h1>Market sentiment and next-bar forecasts</h1>
<p>{note}</p><p>Window (UTC) {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}</p>
<p>Position counts reflect explicit first-person disclosures in matched social posts.</p></header>
{''.join(sections)}
<section><h2>Walk-forward validation</h2>
<p>Three chronological folds compare next-bar log returns from the price-only model, the model with text, and a flat-price forecast. Positive MAE improvement means lower error than the corresponding baseline.</p>
<div class="scroll">{metrics_table}</div>
<h3>Source coverage in the holdout</h3>
<p>Each count is the number of held-out rows with a matched item from the source in the 24 hours before that bar closed.</p>
<div class="scroll">{coverage_table}</div></section></main></body></html>'''
