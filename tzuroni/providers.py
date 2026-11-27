import os
import time
from datetime import timedelta
from itertools import zip_longest
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from .assets import ASSETS
from .news import create_news_client, search_news


ACTORS = {
    'lihkg': 'claude_code_reviewer/lihkg-scraper-en',
    'naver': 'hgservices/naver-stock-discussion-scraper',
    'naver_blog': 'webdata_labs/naver-blog-cafe-scraper',
    'telegram': 'nexuscortex/telegram-channel-scraper',
}
CHANNELS = {'HKEX': ['StockIntelligence'], 'KRX': ['FastStockNews']}
BAR = timedelta(minutes=30)
PRICE_COLUMNS = ['ticker', 'bar_end', 'close', 'high', 'low', 'volume']
POST_COLUMNS = [
    'ticker', 'source', 'post_id', 'published_at', 'url', 'text',
    'likes', 'reposts', 'views', 'followers', 'reactions', 'article_highlights', 'source_detail',
]


class ProviderError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def request_json(session, method, url, context, **kwargs):
    attempts = 3 if method == 'GET' else 1
    for attempt in range(attempts):
        try:
            response = session.request(method, url, timeout=35, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as error:
            status = error.response.status_code
            raise ProviderError(f'{context}: HTTP {status}', status) from None
        except requests.exceptions.ChunkedEncodingError as error:
            if attempt == attempts - 1:
                raise ProviderError(f'{context}: {type(error).__name__}') from None
        except (requests.RequestException, ValueError) as error:
            raise ProviderError(f'{context}: {type(error).__name__}') from None


def run_actor(session, source, payload, token, budget, checkpoint, stage):
    actor = quote(ACTORS[source].replace('/', '~'), safe='~')
    base = 'https://api.apify.com/v2'
    headers = {'Authorization': f'Bearer {token}'}
    saved = checkpoint.get(stage) or {}
    if 'rows' in saved:
        return saved['rows'], saved['details']
    if saved.get('starting') and not saved.get('run_id'):
        raise RuntimeError(f'{stage}: Apify did not confirm startup. Check the Apify Runs page before starting a new run.')

    run_url = f'{base}/actor-runs'
    last_cap = saved.get('budget', budget)
    if saved.get('run_id'):
        run = request_json(session, 'GET', f'{run_url}/{saved["run_id"]}', stage, headers=headers)['data']
    else:
        checkpoint.put(stage, {'starting': True, 'budget': budget})
        try:
            run = request_json(
                session, 'POST', f'{base}/actors/{actor}/runs', stage,
                headers=headers, params={'maxTotalChargeUsd': budget, 'timeout': 240}, json=payload,
            )['data']
        except ProviderError as error:
            if error.status in {400, 401, 402, 403, 404, 422, 429}:
                checkpoint.put(stage, {})
            raise
        checkpoint.put(stage, {'run_id': run['id'], 'budget': budget})

    deadline = time.monotonic() + 280
    while run['status'] in {'READY', 'RUNNING', 'TIMING-OUT', 'ABORTING'}:
        if time.monotonic() >= deadline:
            raise RuntimeError(f'{stage}: Apify run {run["id"]} is still pending. Resume to check again.')
        run = request_json(
            session, 'GET', f'{run_url}/{run["id"]}', stage,
            headers=headers, params={'waitForFinish': 15},
        )['data']
    if run['status'] != 'SUCCEEDED':
        checkpoint.put(stage, {})
        raise RuntimeError(f'{stage}: Apify run {run["id"]} ended with {run["status"]}. Resume to retry this stage.')

    dataset_url = f'{base}/datasets/{run["defaultDatasetId"]}/items'
    results = []
    page_size = 25 if source == 'naver_blog' else 1000
    fields = 'source,url,publishedAt,title,text,snippet,keyword,likeCount,viewCount' if source == 'naver_blog' else None
    while True:
        page = request_json(
            session, 'GET', dataset_url, stage, headers=headers,
            params={'clean': 'true', 'limit': page_size, 'offset': len(results), 'fields': fields},
        )
        results.extend(page)
        if len(page) < page_size:
            details = dict(actor_id=run['actId'], run_id=run['id'], build_id=run.get('buildId'),
                           budget_usd=last_cap)
            checkpoint.put(stage, {'rows': results, 'details': details})
            return results, details


def utc_timestamp(value):
    if value is None:
        return pd.NaT
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        number = float(value)
        return pd.to_datetime(number, utc=True, unit='ms' if number >= 1e12 else 's', errors='coerce')
    if not isinstance(value, str) or len(value.strip()) <= 10:
        return pd.NaT
    timestamp = pd.to_datetime(value, errors='coerce')
    # Intraday alignment requires an explicit time zone.
    return timestamp.tz_convert('UTC') if not pd.isna(timestamp) and timestamp.tzinfo else pd.NaT


def first_value(row, *keys):
    return next((row[key] for key in keys if row.get(key) is not None), None)


def post_record(asset, source, row):
    article_highlights = None
    if source == 'naver':
        if row.get('type') != 'post' or row.get('postType') == 'itemNewsPrice':
            return None
        if str(row.get('stockCode', '')).zfill(6) != asset.code:
            return None
        text = ' '.join(filter(None, [row.get('title'), row.get('text')]))
        published = row.get('writtenAt')
        post_id = row.get('postId')
    elif source == 'telegram':
        text = row.get('text') or ''
        if row.get('type') != 'post' or row.get('isEdited') or not asset.mentioned_in(text):
            return None
        published = row.get('date')
        post_id = f'{row.get("channel")}:{row.get("id")}' if row.get('id') is not None else None
    elif source == 'lihkg':
        text = ' '.join(filter(None, [row.get('title'), first_value(row, 'own_text', 'content')]))
        if not asset.mentioned_in(text):
            return None
        published = first_value(row, 'createdAt', 'createTime', 'create_time', 'created_at', 'postTime')
        post_id = first_value(row, 'postId', 'threadId', 'thread_id', 'id')
    elif source == 'exa':
        text = row.get('title') or ''
        if not asset.mentioned_in(text):
            return None
        article_highlights = '\n'.join(row.get('highlights') or [])
        published = row.get('publishedDate')
        post_id = row.get('url') or row.get('id')
    else:
        raise ValueError(f'Unknown source: {source}')

    timestamp = utc_timestamp(published)
    url = row.get('url') or first_value(row, 'threadUrl', 'thread_url')
    if pd.isna(timestamp) or not text.strip() or not (post_id or url):
        return None
    return dict(
        ticker=asset.ticker, source=source, post_id=str(post_id or url), published_at=timestamp,
        url=url, text=text, likes=first_value(row, 'likes', 'likeCount', 'like_count'),
        reposts=first_value(row, 'reposts', 'repostCount', 'shares'),
        views=first_value(row, 'views', 'viewCount'),
        followers=first_value(row, 'authorFollowers', 'followerCount', 'followers'),
        reactions=row.get('reactionsTotal'), article_highlights=article_highlights,
    )


def select_lihkg_threads(rows, market, start, end):
    candidates = []
    for asset in market:
        matches = []
        for row in rows:
            exact = post_record(asset, 'lihkg', row)
            if not exact and row.get('source_query') not in asset.search_terms:
                continue
            date = utc_timestamp(row.get('created_at'))
            if row.get('thread_id') and not pd.isna(date) and start <= date <= end:
                matches.append((bool(exact), date, str(row['thread_id'])))
        by_day = {}
        for _, date, thread_id in sorted(matches, reverse=True):
            by_day.setdefault(date.date(), thread_id)
        candidates.append(list(by_day.values())[:8])
    return list(dict.fromkeys(thread for group in zip_longest(*candidates) for thread in group if thread))


def collect_live_posts(start, end, budget, checkpoint):
    apify_token = os.environ['APIFY_TOKEN']
    news_client = create_news_client()

    hk = [asset for asset in ASSETS if asset.exchange == 'HKEX']
    kr = [asset for asset in ASSETS if asset.exchange == 'KRX']
    requests_by_source = [
        ('lihkg', hk, {
            'mode': 'search', 'keywords': [term for asset in hk for term in asset.search_terms],
            'dateFrom': start.tz_convert('Asia/Hong_Kong').strftime('%m/%d/%y'),
            'order': 'desc_create_time', 'maxItems': 30,
        }),
        ('naver', kr, {
            'stocks': [asset.code for asset in kr],
            # Apify spends the cap across the batch. Reserve room for all five boards.
            'maxResults': 55,
            'dateFrom': start.tz_convert('Asia/Seoul').strftime('%Y-%m-%d'),
        }),
    ]
    for market in (hk, kr):
        requests_by_source.append(('telegram', market, {
            'channels': CHANNELS[market[0].exchange],
            'searchQueries': [term for asset in market for term in asset.search_terms],
            'postsNewerThan': start.strftime('%Y-%m-%d'), 'maxPostsPerChannel': 40, 'maxItems': 750,
        }))

    posts, runs = [], []
    with requests.Session() as session:
        session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=2, connect=0, read=2, status=2, other=0, allowed_methods={'GET'},
            status_forcelist=[429, 500, 502, 503, 504], backoff_factor=0.5,
            respect_retry_after_header=False,
        )))

        def collect_actor(source, exchange, payload, stage):
            rows, run_details = run_actor(session, source, payload, apify_token, budget, checkpoint, stage)
            runs.append(dict(source=source, exchange=exchange, actor=ACTORS[source], stage=stage,
                             provider_rows=len(rows), **run_details))
            return rows

        for source, market, payload in requests_by_source:
            exchange = market[0].exchange
            rows = collect_actor(source, exchange, payload, f'{source}-{exchange}')
            if source == 'lihkg':
                thread_ids = select_lihkg_threads(rows, market, start, end)
                if thread_ids:
                    details = collect_actor(source, exchange, {
                        'mode': 'thread', 'threadIds': thread_ids, 'maxItems': len(thread_ids), 'maxReplies': 1,
                    }, 'lihkg-HKEX-body')
                    bodies = {str(row['thread_id']): first_value(row, 'own_text', 'content')
                              for row in details if row.get('thread_id') and row.get('title') and not row.get('reply_id')}
                    rows = [{**row, 'own_text': bodies.get(str(row.get('thread_id')), row.get('own_text'))} for row in rows]
            for row in rows:
                for asset in market:
                    record = post_record(asset, source, row)
                    if record and start <= record['published_at'] <= end:
                        posts.append(record)

        for asset in ASSETS:
            stage = f'exa-{asset.ticker}'
            rows = checkpoint.get(stage)
            if rows is None:
                rows = search_news(news_client, asset, start, end)
                checkpoint.put(stage, rows)
            for row in rows:
                record = post_record(asset, 'exa', row)
                if record and start <= record['published_at'] <= end:
                    posts.append(record)

    frame = pd.DataFrame(posts, columns=POST_COLUMNS)
    frame = frame.sort_values('published_at').drop_duplicates(['ticker', 'source', 'post_id'])
    return frame, runs


def collect_prices(start, end):
    import yfinance as yf

    frames = []
    for asset in ASSETS:
        history = yf.Ticker(asset.ticker).history(
            start=(start - 3 * BAR).to_pydatetime(), end=end.to_pydatetime(), interval='30m',
            auto_adjust=False, actions=False,
        )
        if history.empty:
            raise RuntimeError(f'No market bars returned for {asset.ticker}.')
        frame = history[['Close', 'High', 'Low', 'Volume']].dropna().copy()
        frame.columns = ['close', 'high', 'low', 'volume']
        if asset.exchange == 'HKEX':
            local_time = frame.index.tz_convert('Asia/Hong_Kong')
            minutes = local_time.hour * 60 + local_time.minute
            frame = frame.loc[((570 <= minutes) & (minutes < 720)) | ((780 <= minutes) & (minutes < 960))].copy()
        frame['bar_end'] = frame.index.tz_convert('UTC') + BAR
        frame['ticker'] = asset.ticker
        frame = frame.loc[frame['bar_end'].le(end) & frame['close'].gt(0), PRICE_COLUMNS]
        if frame.empty:
            raise RuntimeError(f'No completed market bars returned for {asset.ticker}.')
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).drop_duplicates(['ticker', 'bar_end'])
