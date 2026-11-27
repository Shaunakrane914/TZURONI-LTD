import re
import unicodedata
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .assets import ASSETS, SOURCES
from .positions import disclosed_positions


POSITIVE = ('bullish', 'upgrade', 'upgraded', 'outperform', 'beat estimates', 'rally',
            'buy rating', 'strong demand', 'record profit', 'rebound', '看好', '利好', '上調',
            '上调', '反彈', '反弹', '突破', '大升', '上升', '급등', '상승', '호재', '매수', '실적 개선')
NEGATIVE = ('bearish', 'downgrade', 'downgraded', 'underperform', 'missed estimates', 'selloff',
            'sell rating', 'weak demand', 'profit warning', 'plunge', '利淡', '利空', '下調',
            '下调', '沽售', '下跌', '大跌', '回落', '급락', '하락', '악재', '매도', '실적 악화')
PRICE_FEATURES = ['last_return', 'momentum', 'volatility', 'bar_range', 'volume_change', 'is_krx']
SOCIAL_FEATURES = [f'{group}_{metric}' for group in ('forum', 'telegram')
                   for metric in ('score', 'count', 'scored_share')]
NEWS_FEATURES = ['news_score', 'news_count', 'news_scored_share']
FEATURES = PRICE_FEATURES + SOCIAL_FEATURES + NEWS_FEATURES
MODEL_FEATURES = {
    'price_only': PRICE_FEATURES,
    'price_and_social': PRICE_FEATURES + SOCIAL_FEATURES,
    'price_and_news': PRICE_FEATURES + NEWS_FEATURES,
    'price_and_text': FEATURES,
}
BENCHMARKS = ('zero_return', *MODEL_FEATURES)
LOOKBACK = timedelta(hours=24)


def score_text(text):
    text = unicodedata.normalize('NFKC', text).casefold()

    def hits(terms):
        return sum(bool(re.search(rf'\b{re.escape(term)}\b', text)) if term.isascii()
                   else term in text for term in terms)

    positive, negative = hits(POSITIVE), hits(NEGATIVE)
    if positive + negative == 0:
        return np.nan
    return (positive - negative) / (positive + negative)


def build_panel(prices, posts, start, end, sentiment_model='lexicon'):
    if sentiment_model == 'semantic':
        from .semantic import semantic_scores

        unique_text = posts['text'].unique()
        score_by_text = dict(zip(unique_text, semantic_scores(unique_text), strict=True))
        scores = posts['text'].map(score_by_text)
    else:
        scores = posts['text'].map(score_text)
    scored = posts.assign(score=scores)
    frames = []
    for asset in ASSETS:
        bars = prices.loc[prices['ticker'].eq(asset.ticker) & prices['bar_end'].le(end)].copy()
        bars = bars.sort_values('bar_end').reset_index(drop=True)
        bars['exchange'] = asset.exchange
        bars['is_krx'] = float(asset.exchange == 'KRX')
        log_close = np.log(bars['close'])
        bars['last_return'] = log_close.diff()
        bars['momentum'] = log_close.diff(3)
        bars['volatility'] = bars['last_return'].rolling(5, min_periods=3).std()
        bars['bar_range'] = (bars['high'] - bars['low']) / bars['close']
        bars['volume_change'] = np.log1p(bars['volume']).diff()
        bars['support'] = bars['low'].rolling(20, min_periods=5).min()
        bars['resistance'] = bars['high'].rolling(20, min_periods=5).max()
        bars['target_at'] = bars['bar_end'].shift(-1)
        bars['target_close'] = bars['close'].shift(-1)
        bars['target_return'] = log_close.shift(-1) - log_close

        asset_posts = scored.loc[scored['ticker'].eq(asset.ticker)]
        forum, telegram = SOURCES[asset.exchange]
        for group, source in [('forum', forum), ('telegram', telegram), ('news', 'exa')]:
            stream = asset_posts.loc[asset_posts['source'].eq(source)].sort_values('published_at')
            times = pd.DatetimeIndex(stream['published_at'])
            cutoffs = pd.DatetimeIndex(bars['bar_end'])
            left = times.searchsorted(cutoffs - LOOKBACK, side='right')
            right = times.searchsorted(cutoffs, side='right')
            scores = stream['score'].to_numpy(dtype=float)
            scored_prefix = np.r_[0, np.isfinite(scores).cumsum()]
            score_prefix = np.r_[0.0, np.nancumsum(scores)]
            count = right - left
            recognized = scored_prefix[right] - scored_prefix[left]
            bars[f'{group}_count'] = np.log1p(count)
            bars[f'{group}_score'] = (score_prefix[right] - score_prefix[left]) / np.maximum(recognized, 1)
            bars[f'{group}_scored_share'] = recognized / np.maximum(count, 1)

        bars = bars.loc[bars['bar_end'].ge(start + LOOKBACK)].dropna(subset=PRICE_FEATURES)
        if bars.empty:
            raise ValueError(f'Insufficient completed price bars for {asset.ticker} in the selected window.')
        frames.append(bars)

    return pd.concat(frames, ignore_index=True).sort_values(['bar_end', 'ticker']), scored


def predict(train, test, columns):
    model = make_pipeline(StandardScaler(), Ridge(alpha=15.0))
    model.fit(train[columns], train['target_return'])
    return model.predict(test[columns])


def evaluate(panel):
    labeled = panel.dropna(subset=['target_return'])
    dates = pd.DatetimeIndex(labeled['bar_end'].unique()).sort_values()
    if len(dates) < 20:
        raise ValueError(f'Need at least 20 labeled bar timestamps for validation, found {len(dates)}.')

    folds = []
    for fold, (train_dates, test_dates) in enumerate(TimeSeriesSplit(n_splits=3).split(dates), 1):
        first_test = dates[test_dates[0]]
        train = labeled.loc[labeled['bar_end'].isin(dates[train_dates]) & labeled['target_at'].lt(first_test)]
        test = labeled.loc[labeled['bar_end'].isin(dates[test_dates])].copy()
        test['fold'] = fold
        test['trained_through'] = train['target_at'].max()
        test['zero_return'] = 0.0
        for model, columns in MODEL_FEATURES.items():
            test[model] = predict(train, test, columns)
        folds.append(test)

    scores = pd.concat(folds, ignore_index=True)
    for model in BENCHMARKS:
        scores[f'{model}_close'] = scores['close'] * np.exp(scores[model])

    metrics = []
    for market in ('All', *SOURCES):
        rows = scores if market == 'All' else scores.loc[scores['exchange'].eq(market)]
        actual = rows['target_return'].to_numpy()
        flat_mae = np.abs(actual * 10_000).mean()
        price_mae = np.abs((rows['price_only'].to_numpy() - actual) * 10_000).mean()
        for model in BENCHMARKS:
            predicted = rows[model].to_numpy()
            error = (predicted - actual) * 10_000
            mae = np.abs(error).mean()
            metrics.append(dict(
                exchange=market, model=model, observations=len(rows),
                mae_bps=mae, rmse_bps=np.sqrt(np.mean(error ** 2)),
                mae_improvement_vs_flat_pct=100 * (1 - mae / flat_mae) if flat_mae > 0 else np.nan,
                mae_improvement_vs_price_only_pct=100 * (1 - mae / price_mae) if price_mae > 0 else np.nan,
                directional_accuracy=np.mean(np.sign(actual) == np.sign(predicted)),
            ))

    # Different markets can have different latest bar times. Refit at each cutoff.
    latest = panel.groupby('ticker', sort=False).tail(1).copy()
    latest['forecast_return'] = np.nan
    for cutoff, rows in latest.groupby('bar_end'):
        train = labeled.loc[labeled['target_at'].le(cutoff)]
        latest.loc[rows.index, 'forecast_return'] = predict(train, rows, FEATURES)
    latest['forecast_price'] = latest['close'] * np.exp(latest['forecast_return'])
    return latest, scores, pd.DataFrame(metrics)


def summarize(latest, scored):
    rows = []
    for asset in ASSETS:
        forecast = latest.loc[latest['ticker'].eq(asset.ticker)].iloc[0]
        posts = scored.loc[scored['ticker'].eq(asset.ticker)]
        forum, telegram = SOURCES[asset.exchange]
        social = posts.loc[posts['source'].isin((forum, telegram))]
        at_bar = social.loc[social['published_at'].le(forecast['bar_end'])
                            & social['published_at'].gt(forecast['bar_end'] - LOOKBACK)]
        disclosures = social['text'].map(lambda text: disclosed_positions(text, asset))
        score = social.groupby('source')['score'].mean().mean()
        direction = 'unscored' if pd.isna(score) else 'bullish' if score > 0.05 else 'bearish' if score < -0.05 else 'neutral'
        rows.append(dict(
            exchange=asset.exchange, ticker=asset.ticker, asset_name=asset.name, currency=asset.currency,
            bar_end_utc=forecast['bar_end'], current_price=forecast['close'],
            social_mentions=len(social), scored_mentions=int(social['score'].count()),
            social_sources=social['source'].nunique(), social_sources_at_bar=at_bar['source'].nunique(),
            forum_source=forum, forum_mentions=int(posts['source'].eq(forum).sum()),
            telegram_mentions=int(posts['source'].eq(telegram).sum()),
            news_mentions=int(posts['source'].eq('exa').sum()), sentiment_score=score,
            sentiment_direction=direction, support=forecast['support'], resistance=forecast['resistance'],
            long_position_posts=int(disclosures.map(lambda positions: 'long' in positions).sum()),
            short_position_posts=int(disclosures.map(lambda positions: 'short' in positions).sum()),
            next_bar_price=forecast['forecast_price'], next_bar_return_pct=100 * np.expm1(forecast['forecast_return']),
        ))
    ranking = pd.DataFrame(rows)
    ranking['sentiment_rank'] = ranking.groupby('exchange')['sentiment_score'].rank(method='min', ascending=False)
    return ranking.sort_values(
        ['exchange', 'social_mentions', 'sentiment_score', 'ticker'], ascending=[True, False, False, True],
    )
