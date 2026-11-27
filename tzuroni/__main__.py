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
from .model import BENCHMARKS, build_panel, evaluate, summarize
from .naver_backfill import collect_backfill
from .providers import POST_COLUMNS, PRICE_COLUMNS, collect_live_posts, collect_prices
from .report import render_report


def analyze(prices, posts, metadata, output):
    start, end = pd.Timestamp(metadata['start']), pd.Timestamp(metadata['end'])
    prices['bar_end'] = pd.to_datetime(prices['bar_end'], utc=True, format='mixed')
    posts['published_at'] = pd.to_datetime(posts['published_at'], utc=True, format='mixed')
    posts = posts.loc[posts['published_at'].between(start, end) & posts['text'].notna()]
    posts = posts.drop_duplicates(subset=['ticker', 'source', 'post_id'])
    panel, scored = build_panel(prices, posts, start, end, metadata.get('sentiment_model', 'lexicon'))
    latest, backtest, metrics = evaluate(panel)
    for source in ('forum', 'telegram', 'news'):
        backtest[f'{source}_available'] = backtest[f'{source}_count'].gt(0)
    backtest['both_social_available'] = backtest['forum_available'] & backtest['telegram_available']
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
                        'close', 'target_close', *(f'{model}_close' for model in BENCHMARKS),
                        'target_return', *BENCHMARKS,
                        'forum_available', 'telegram_available', 'both_social_available', 'news_available']
    backtest[backtest_columns].to_csv(output / 'backtest.csv', index=False)
    (output / 'report.html').write_text(
        render_report(ranking, metrics, backtest, start, end, metadata), encoding='utf-8',
    )
    if metadata['mode'] == 'live':
        (output / 'report-share.html').write_text(
            render_report(ranking, metrics, backtest, start, end, metadata, public=True), encoding='utf-8',
        )
    print(f'{metadata["mode"]} report: {output.resolve() / "report.html"}')


def save_snapshot(output, prices, posts, metadata):
    snapshot = output / 'snapshot'
    snapshot.mkdir(exist_ok=True)
    prices.to_json(snapshot / 'prices.json', orient='records', date_format='iso', double_precision=15)
    posts.to_json(snapshot / 'posts.json', orient='records', date_format='iso', force_ascii=False)
    (snapshot / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def read_snapshot(snapshot):
    metadata = json.loads((snapshot / 'metadata.json').read_text(encoding='utf-8'))
    prices = pd.read_json(snapshot / 'prices.json', orient='records', dtype=False).reindex(columns=PRICE_COLUMNS)
    posts = pd.read_json(snapshot / 'posts.json', orient='records', dtype=False).reindex(columns=POST_COLUMNS)
    return prices, posts, metadata


def run_backfill(args):
    load_dotenv(dotenv_path=Path('.env'))
    if not math.isfinite(args.apify_budget) or args.apify_budget <= 0 or not os.getenv('APIFY_TOKEN'):
        raise ValueError('Set APIFY_TOKEN and provide a positive finite per-run --apify-budget.')
    prices, posts, metadata = read_snapshot(args.snapshot)
    if metadata.get('mode') != 'live':
        raise ValueError('Historical backfill requires a live input snapshot.')
    start, end = pd.Timestamp(metadata['start']), pd.Timestamp(metadata['end'])
    with open_live_checkpoint(args.out, args.resume, end) as checkpoint:
        if checkpoint.end != end:
            raise ValueError('The input snapshot cutoff must match the backfill checkpoint.')
        historical, runs = collect_backfill(start, end, args.apify_budget, checkpoint)
        historical['published_at'] = pd.to_datetime(historical['published_at'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        posts = pd.concat([frame.dropna(axis=1, how='all') for frame in (posts, historical)], ignore_index=True)
        posts = posts.reindex(columns=POST_COLUMNS).drop_duplicates(['ticker', 'source', 'post_id'])
        metadata = {**metadata, 'naver_backfill': True, 'actor_runs': [*metadata['actor_runs'], *runs],
                    'sentiment_model': args.sentiment_model or metadata.get('sentiment_model', 'lexicon')}
        save_snapshot(args.out, prices, posts, metadata)
        analyze(prices, posts, metadata, args.out)


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
        metadata = {'mode': 'live', 'start': start.isoformat(), 'end': end.isoformat(), 'actor_runs': runs,
                    'sentiment_model': args.sentiment_model or 'lexicon'}
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
    backfill = modes.add_parser('backfill-naver', help='Supplement a live snapshot with dated Naver Blog posts.')
    backfill.add_argument('snapshot', type=Path)
    backfill.add_argument('--apify-budget', type=float, required=True, help='Maximum USD per three-day actor run.')
    backfill.add_argument('--resume', action='store_true')
    backfill.add_argument('--out', type=Path, default=Path('outputs/historical'))
    for command in (demo, live, replay, backfill):
        command.add_argument('--sentiment-model', choices=('lexicon', 'semantic'),
                             help='Use the keyword baseline or the optional local multilingual encoder.')
    args = parser.parse_args()

    try:
        if args.mode == 'live':
            run_live(args)
            return
        if args.mode == 'backfill-naver':
            run_backfill(args)
            return

        args.out.mkdir(parents=True, exist_ok=True)
        if args.mode == 'replay':
            prices, posts, metadata = read_snapshot(args.snapshot)
            if args.sentiment_model:
                metadata['sentiment_model'] = args.sentiment_model
        else:
            end = pd.Timestamp('2026-11-27T10:45:00Z')
            start = end - timedelta(days=10)
            metadata = {'mode': 'demo', 'start': start.isoformat(), 'end': end.isoformat(), 'actor_runs': [],
                        'sentiment_model': args.sentiment_model or 'lexicon'}
            prices, posts = make_demo(start, end)
            save_snapshot(args.out, prices, posts, metadata)
        analyze(prices, posts, metadata, args.out)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
