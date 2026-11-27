import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from tzuroni.assets import ASSETS
from tzuroni.checkpoint import LiveCheckpoint, open_live_checkpoint
from tzuroni.demo import make_demo
from tzuroni.model import build_panel, evaluate, summarize
from tzuroni.news import search_news
from tzuroni.positions import disclosed_positions
from tzuroni.providers import POST_COLUMNS, collect_live_posts, post_record, run_actor, select_lihkg_threads


class ForecastTimingTests(unittest.TestCase):
    def setUp(self):
        self.end = pd.Timestamp('2026-11-27T10:45:00Z')
        self.start = self.end - timedelta(days=10)
        self.prices, self.posts = make_demo(self.start, self.end)

    def test_posts_change_only_features_at_or_after_publication(self):
        ticker = ASSETS[0].ticker
        bars = self.prices.loc[self.prices['ticker'].eq(ticker), 'bar_end'].sort_values()
        cutoff = bars.iloc[25]
        self.posts = self.posts.iloc[:0].copy()
        self.posts.loc[0, ['ticker', 'source', 'published_at', 'text']] = (
            ticker, 'lihkg', cutoff + timedelta(seconds=1), 'Tencent is bullish.',
        )
        panel, _ = build_panel(self.prices, self.posts, self.start, self.end)
        rows = panel.loc[panel['ticker'].eq(ticker)].set_index('bar_end')
        self.assertEqual(rows.loc[cutoff, 'forum_count'], 0)
        self.assertAlmostEqual(rows.loc[bars.iloc[26], 'forum_count'], np.log(2))
        self.assertEqual(rows.loc[bars.iloc[26], 'forum_score'], 1)

    def test_fold_never_uses_a_training_label_from_a_test_cutoff(self):
        panel, _ = build_panel(self.prices, self.posts, self.start, self.end)
        latest, backtest, metrics = evaluate(panel)
        self.assertEqual(len(latest), 10)
        self.assertEqual(len(metrics), 9)
        price_baseline = metrics.loc[metrics['model'].eq('price_only'), 'mae_improvement_vs_price_only_pct']
        np.testing.assert_allclose(price_baseline, 0)
        for _, fold in backtest.groupby('fold'):
            self.assertLess(fold['trained_through'].max(), fold['bar_end'].min())
            self.assertTrue(fold['target_at'].gt(fold['bar_end']).all())
            np.testing.assert_allclose(fold['close'] * np.exp(fold['target_return']), fold['target_close'])

    def test_report_distinguishes_missing_social_sources(self):
        posts = self.posts.loc[self.posts['source'].ne('telegram')]
        panel, scored = build_panel(self.prices, posts, self.start, self.end)
        latest, _, _ = evaluate(panel)
        ranking = summarize(latest, scored)
        self.assertTrue(ranking['social_sources'].eq(1).all())
        self.assertTrue(ranking['social_sources_at_bar'].le(1).all())
        self.assertTrue(ranking['sentiment_rank'].between(1, 5).all())
        self.assertTrue(ranking['long_position_posts'].eq(0).all())


class PositionDisclosureTests(unittest.TestCase):
    def test_matches_explicit_asset_specific_positions(self):
        tencent, byd, samsung = ASSETS[0], ASSETS[4], ASSETS[5]
        text = "I'm still long 0700.HK. I am short BYD."
        self.assertEqual(disclosed_positions(text, tencent), {'long'})
        self.assertEqual(disclosed_positions(text, byd), {'short'})
        self.assertEqual(disclosed_positions('我目前持有騰訊空倉', tencent), {'short'})
        self.assertEqual(disclosed_positions('저는 삼성전자 주식을 보유 중입니다.', samsung), {'long'})

    def test_skips_opinions_negation_and_ambiguous_assets(self):
        asset = ASSETS[0]
        for text in (
            'Tencent looks bullish.', 'I am not short Tencent.', 'If I am long Tencent, I would sell.',
            'Tencent and BYD look good. I am long.', 'Tencent: I am short of cash.',
        ):
            with self.subTest(text=text):
                self.assertEqual(disclosed_positions(text, asset), set())


class ProviderMappingTests(unittest.TestCase):
    def test_lihkg_search_term_can_select_a_thread_but_cannot_count_it(self):
        asset = ASSETS[2]
        end = pd.Timestamp('2026-11-27T10:45Z')
        start = end - timedelta(days=10)
        row = dict(thread_id='a', source_query=asset.aliases[0], title='Delivery sector roundup',
                   created_at='2026-11-26T03:00:00Z')
        self.assertEqual(select_lihkg_threads([row], [asset], start, end), ['a'])
        self.assertIsNone(post_record(asset, 'lihkg', row))
        row['own_text'] = f'{asset.aliases[0]} reported earnings.'
        self.assertIsNotNone(post_record(asset, 'lihkg', row))

    def test_normalizes_and_deduplicates_provider_records(self):
        when = '2026-11-26T12:00:00+09:00'
        actor_rows = [
            [{'thread_id': 'a', 'created_at': when, 'title': '騰訊利好', 'like_count': 3}] * 2,
            [{'thread_id': 'a', 'title': '騰訊利好', 'own_text': 'outperform', 'content': 'bearish quoted text'}],
            [{'type': 'post', 'stockCode': '005930', 'postId': 'a', 'writtenAt': when,
              'text': '삼성전자 호재', 'views': 13}],
            [{'type': 'post', 'channel': 'StockIntelligence', 'id': 1, 'date': when,
              'text': 'Tencent rally', 'reactionsTotal': 2}],
            [{'type': 'post', 'channel': 'FastStockNews', 'id': 1, 'date': when,
              'text': '삼성전자 상승', 'isEdited': True}],
        ]
        article = {'title': 'Tencent upgraded', 'publishedDate': when, 'url': 'https://example.com/news',
                   'highlights': ['This passage was updated after publication.']}

        def actor_result(_, source, payload, token, budget, checkpoint, stage):
            return actor_rows.pop(0), {'run_id': stage, 'actor_id': source, 'build_id': None}

        with TemporaryDirectory() as directory, \
             patch.dict('os.environ', {'APIFY_TOKEN': 'test', 'EXA_API_KEY': 'test'}), \
             patch('tzuroni.providers.run_actor', side_effect=actor_result), \
             patch('tzuroni.providers.search_news', return_value=[article]):
            checkpoint = LiveCheckpoint(Path(directory) / 'state.json', pd.Timestamp('2026-11-27T00:00Z'))
            posts, runs = collect_live_posts(checkpoint.start, checkpoint.end, 0.25, checkpoint)

        self.assertEqual(list(posts.columns), POST_COLUMNS)
        self.assertEqual(posts['source'].nunique(), 4)
        self.assertEqual(len(posts), 4)
        self.assertEqual(len(runs), 5)
        self.assertTrue(posts['published_at'].eq(pd.Timestamp('2026-11-26T03:00Z')).all())
        lihkg = posts.loc[posts['source'].eq('lihkg')].iloc[0]
        self.assertEqual(lihkg['likes'], 3)
        self.assertIn('outperform', lihkg['text'])
        self.assertNotIn('bearish', lihkg['text'])
        exa = posts.loc[posts['source'].eq('exa')].iloc[0]
        self.assertEqual(exa['text'], article['title'])
        self.assertEqual(exa['article_highlights'], article['highlights'][0])

    def test_exa_sdk_request_matches_pipeline_window(self):
        end = pd.Timestamp('2026-11-27T10:45Z')
        start = end - timedelta(days=10)
        client = Mock()
        client.search.return_value = SimpleNamespace(results=[SimpleNamespace(
            id='article', url='https://example.com/news', title='Tencent upgraded',
            published_date='2026-11-26T03:00:00Z', highlights=['An analyst raised guidance.'],
        )])
        result = search_news(client, ASSETS[0], start, end)[0]
        query, = client.search.call_args.args
        options = client.search.call_args.kwargs
        self.assertIn('0700.HK', query)
        self.assertIn('stock sentiment', options['objective'])
        self.assertEqual(options['start_published_date'], start.isoformat())
        self.assertEqual(options['end_published_date'], end.isoformat())
        self.assertEqual(options['contents'], {'highlights': True})
        self.assertEqual(set(options), {'objective', 'start_published_date', 'end_published_date', 'contents'})
        self.assertEqual(result['publishedDate'], '2026-11-26T03:00:00Z')
        self.assertEqual(result['highlights'], ['An analyst raised guidance.'])


class CheckpointTests(unittest.TestCase):
    def test_pending_actor_resumes_by_run_id_without_another_post(self):
        end = pd.Timestamp('2026-11-27T10:45Z')
        succeeded = dict(id='run1', actId='actor1', defaultDatasetId='dataset1', status='SUCCEEDED')
        with TemporaryDirectory() as directory:
            target = Path(directory)
            with open_live_checkpoint(target, False, end) as checkpoint:
                with patch('tzuroni.providers.request_json', side_effect=[
                    {'data': dict(id='run1', status='RUNNING')}, RuntimeError('interrupted'),
                ]):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        run_actor(None, 'lihkg', {}, 'test', 0.25, checkpoint, 'lihkg-HKEX')
                with self.assertRaisesRegex(ValueError, 'already holds'):
                    with open_live_checkpoint(target, True, end):
                        self.fail('A second writer acquired the live checkpoint')

            with open_live_checkpoint(target, True, end + timedelta(days=1)) as checkpoint:
                self.assertEqual(checkpoint.end, end)
                with patch('tzuroni.providers.request_json', side_effect=[{'data': succeeded}, [{'id': 1}]]) as api:
                    rows, details = run_actor(None, 'lihkg', {}, 'test', 0.50, checkpoint, 'lihkg-HKEX')
                self.assertEqual([call.args[1] for call in api.call_args_list], ['GET', 'GET'])
                self.assertEqual(rows, [{'id': 1}])
                self.assertEqual(details['budget_usd'], 0.25)
                with patch('tzuroni.providers.request_json') as api:
                    self.assertEqual(run_actor(None, 'lihkg', {}, 'test', 0.50, checkpoint, 'lihkg-HKEX'), (rows, details))
                    api.assert_not_called()


if __name__ == '__main__':
    unittest.main()
