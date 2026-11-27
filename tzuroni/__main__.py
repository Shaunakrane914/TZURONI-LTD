import argparse
import json
import math
import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from .checkpoint import open_live_checkpoint
from .demo import make_demo
from .model import build_panel, evaluate, summarize
from .providers import POST_COLUMNS, PRICE_COLUMNS, collect_live_posts, collect_prices
from .report import render_report


def analyze(prices, posts, metadata, output):
    start, end = pd.Timestamp(metadata['start']), pd.Timestamp(metadata['end'])
    prices['bar_end'] = pd.to_datetime(prices['bar_end'], utc=True)
    posts['published_at'] = pd.to_datetime(posts['published_at'], utc=True)
    posts = posts.loc[posts['published_at'].between(start, end) & posts['text'].notna()]
    posts = posts.drop_duplicates(subset=['ticker', 'source', 'post_id'])
    panel, scored = build_panel(prices, posts, start, end)
    latest, backtest, metrics = evaluate(panel)
    ranking = summarize(latest, scored)
    ranking.to_csv(output / 'rankings.csv', index=False)
    metrics.to_csv(output / 'metrics.csv', index=False)
    news_columns = ['ticker', 'published_at', 'url', 'text', 'article_highlights', 'score']
    news = scored.loc[scored['source'].eq('exa'), news_columns].copy()
    # Keep publisher-supplied text literal when the CSV is opened in a spreadsheet.
    for column in ('url', 'text', 'article_highlights'):
        news[column] = news[column].astype('string').str.replace(r'^(\s*[=+@-])', r"'\1", regex=True)
    news.sort_values(['ticker', 'published_at'], ascending=[True, False]).to_csv(output / 'news.csv', index=False)
    backtest_columns = ['fold', 'exchange', 'ticker', 'bar_end', 'target_at', 'trained_through',
                        'close', 'target_close', 'zero_return_close', 'price_only_close',
                        'price_and_text_close', 'target_return', 'zero_return', 'price_only', 'price_and_text']
    backtest[backtest_columns].to_csv(output / 'backtest.csv', index=False)
    (output / 'report.html').write_text(
        render_report(ranking, metrics, start, end, metadata['mode']), encoding='utf-8',
    )
    print(f'{metadata["mode"]} report: {output.resolve() / "report.html"}')


def save_snapshot(output, prices, posts, metadata):
    snapshot = output / 'snapshot'
    snapshot.mkdir(exist_ok=True)
    prices.to_json(snapshot / 'prices.json', orient='records', date_format='iso', double_precision=15)
    posts.to_json(snapshot / 'posts.json', orient='records', date_format='iso', force_ascii=False)
    (snapshot / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def run_live(args):
    load_dotenv(dotenv_path=Path('.env'))
    if not math.isfinite(args.apify_budget) or args.apify_budget <= 0:
        raise ValueError('--apify-budget must be a positive, finite USD amount.')
    if not os.getenv('APIFY_TOKEN') or not os.getenv('EXA_API_KEY'):
        raise ValueError('Set APIFY_TOKEN and EXA_API_KEY in the local .env file for a live run.')

    with open_live_checkpoint(args.out, args.resume, pd.Timestamp.now(tz='UTC')) as checkpoint:
        start, end = checkpoint.start, checkpoint.end
        cached = checkpoint.get('prices')
        if cached is None:
            prices = collect_prices(start, end)
            checkpoint.put('prices', json.loads(prices.to_json(orient='records', date_format='iso', double_precision=15)))
        else:
            prices = pd.DataFrame(cached, columns=PRICE_COLUMNS)
        posts, runs = collect_live_posts(start, end, args.apify_budget, checkpoint)
        metadata = {'mode': 'live', 'start': start.isoformat(), 'end': end.isoformat(), 'actor_runs': runs}
        save_snapshot(args.out, prices, posts, metadata)
        analyze(prices, posts, metadata, args.out)


def main():
    parser = argparse.ArgumentParser(description='Build the market sentiment and next-bar report.')
    modes = parser.add_subparsers(dest='mode', required=True)
    demo = modes.add_parser('demo', help='Run using deterministic synthetic input and no credentials.')
    demo.add_argument('--out', type=Path, default=Path('outputs/demo'))
    live = modes.add_parser('live', help='Fetch market bars, Apify social posts, and Exa news.')
    live.add_argument('--out', type=Path, default=Path('outputs/live'))
    live.add_argument('--apify-budget', type=float, required=True, help='Maximum USD per Apify actor run.')
    live.add_argument('--resume', action='store_true', help='Resume the live checkpoint in the output directory.')
    replay = modes.add_parser('replay', help='Rebuild a report from an existing input snapshot.')
    replay.add_argument('snapshot', type=Path)
    replay.add_argument('--out', type=Path, default=Path('outputs/replay'))
    args = parser.parse_args()

    try:
        if args.mode == 'live':
            run_live(args)
            return

        args.out.mkdir(parents=True, exist_ok=True)
        if args.mode == 'replay':
            metadata = json.loads((args.snapshot / 'metadata.json').read_text(encoding='utf-8'))
            prices = pd.read_json(args.snapshot / 'prices.json', orient='records', dtype=False).reindex(columns=PRICE_COLUMNS)
            posts = pd.read_json(args.snapshot / 'posts.json', orient='records', dtype=False).reindex(columns=POST_COLUMNS)
        else:
            end = pd.Timestamp('2026-11-27T10:45:00Z')
            start = end - timedelta(days=10)
            metadata = {'mode': 'demo', 'start': start.isoformat(), 'end': end.isoformat(), 'actor_runs': []}
            prices, posts = make_demo(start, end)
            save_snapshot(args.out, prices, posts, metadata)
        analyze(prices, posts, metadata, args.out)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
