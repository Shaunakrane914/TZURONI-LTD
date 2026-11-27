import os
import re
from datetime import timedelta
from urllib.parse import urlsplit

import pandas as pd
import requests

from .assets import ASSETS
from .providers import ACTORS, POST_COLUMNS, run_actor, utc_timestamp


FINANCE_CONTEXT = re.compile(r'주가|실적|매출|투자|증시|배당|수익|목표가|매도|매수|상장|영업이익|공시')


def normalize_blog_post(row, assets_by_keyword):
    url = row.get('url') or ''
    if row.get('source') != 'blog' or urlsplit(url).hostname not in {'blog.naver.com', 'm.blog.naver.com'}:
        return None
    published = utc_timestamp(row.get('publishedAt'))
    if pd.isna(published):
        return None
    local = published.tz_convert('Asia/Seoul')
    if not (local.hour or local.minute or local.second or local.microsecond):
        # A date represented as midnight has insufficient precision for a 30-minute backtest.
        return None
    title, body = row.get('title') or '', row.get('text') or row.get('snippet') or ''
    asset = assets_by_keyword.get(row.get('keyword'))
    if asset is None:
        return None
    # Site references to Naver appear in other companies' posts and don't identify the issuer.
    if asset.ticker == '035420.KS':
        title = re.sub(r'네이버\s*(?:블로그|증권|금융|뉴스|카페)', '', title)
        body = re.sub(r'네이버\s*(?:블로그|증권|금융|뉴스|카페)', '', body)
    excerpt = body[:2000]
    if not asset.mentioned_in(title):
        excerpts = [part.strip() for part in re.split(r'[\n。!?]+|\.(?=\s)', excerpt)
                    if asset.mentioned_in(part) and FINANCE_CONTEXT.search(part)]
        if not excerpts:
            return None
        excerpt = ' '.join(excerpts[:3])
    return dict(ticker=asset.ticker, source='naver', source_detail='naver_blog',
                post_id=f'blog:{urlsplit(url).path.rstrip("/")}', published_at=published,
                url=url, text=' '.join(filter(None, (title, excerpt))), likes=row.get('likeCount'),
                views=row.get('viewCount'))


def collect_backfill(start, end, budget, checkpoint):
    market = {f'{asset.aliases[0]} 주가': asset for asset in ASSETS if asset.exchange == 'KRX'}
    beginning = start.tz_convert('Asia/Seoul').date()
    finish = end.tz_convert('Asia/Seoul').date()
    posts, runs = [], []
    with requests.Session() as session:
        for day in pd.date_range(beginning, finish, freq='3D').date:
            last_day = min(day + timedelta(days=2), finish)
            stage = f'naver-blog-{day}'
            payload = dict(searchKeywords=list(market), source='blog', sortBy='date',
                           dateFrom=day.isoformat(), dateTo=last_day.isoformat(), maxPostsPerKeyword=32,
                           includeContent=True, includeComments=False)
            rows, details = run_actor(session, 'naver_blog', payload, os.environ['APIFY_TOKEN'],
                                      budget, checkpoint, stage)
            matched = []
            for row in rows:
                post = normalize_blog_post(row, market)
                if post and start <= post['published_at'] <= end:
                    matched.append(post)
            posts.extend(matched)
            runs.append(dict(source='naver_blog', exchange='KRX', actor=ACTORS['naver_blog'], stage=stage,
                             provider_rows=len(rows), matched_rows=len(matched), **details))
    frame = pd.DataFrame(posts, columns=POST_COLUMNS).drop_duplicates(['ticker', 'source', 'post_id'])
    return frame, runs
