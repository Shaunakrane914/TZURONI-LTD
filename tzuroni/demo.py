from datetime import timedelta

import numpy as np
import pandas as pd

from .assets import ASSETS, SOURCES
from .providers import POST_COLUMNS


def make_demo(start, end):
    rng = np.random.default_rng(42)
    bars, posts = [], []
    initial_prices = (400, 105, 105, 45, 100, 80_000, 190_000, 230_000, 180_000, 40_000)
    for i, asset in enumerate(ASSETS):
        days = pd.date_range(start.normalize(), end.normalize(), freq='B')
        market = [(90, 240), (300, 480)] if asset.exchange == 'HKEX' else [(0, 390)]
        closes = [day + timedelta(minutes=minute + 30)
                  for day in days for opening, closing in market
                  for minute in range(opening, closing, 30)]
        price = initial_prices[i]
        previous_signal = 0.0
        for close in sorted(closes):
            if not start <= close <= end:
                continue
            price *= np.exp(0.001 * previous_signal + rng.normal(0, 0.004))
            bars.append(dict(ticker=asset.ticker, bar_end=close, close=price,
                             high=price * 1.002, low=price * 0.998, volume=int(rng.integers(1500, 50_000))))
            signal = int(rng.choice([-1, 0, 1]))
            previous_signal = signal
            phrases = {-1: 'bearish and weak demand', 0: 'posted its monthly update', 1: 'bullish after strong demand'}
            published = close - timedelta(minutes=7)
            if published < start:
                continue
            for source in (*SOURCES[asset.exchange], 'exa'):
                if rng.random() < (0.22 + 0.05 * (i % 5)):
                    posts.append(dict(ticker=asset.ticker, source=source, post_id=f'demo-{len(posts)}',
                                      published_at=published, text=f'{asset.name} {phrases[signal]}'))

    return pd.DataFrame(bars), pd.DataFrame(posts, columns=POST_COLUMNS)
