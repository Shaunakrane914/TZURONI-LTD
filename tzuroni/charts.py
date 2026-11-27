from base64 import b64encode
from io import BytesIO

from matplotlib.figure import Figure
import pandas as pd


COLORS = {'HKEX': '#187e83', 'KRX': '#4f61b9'}


def png_element(figure, alt):
    output = BytesIO()
    figure.savefig(output, format='png', dpi=130, bbox_inches='tight', facecolor='white')
    encoded = b64encode(output.getvalue()).decode('ascii')
    return f'<img class="chart" alt="{alt}" src="data:image/png;base64,{encoded}">'


def prediction_chart(backtest):
    figure = Figure(figsize=(9, 3.8), layout='constrained')
    axes = figure.subplots(1, 2)
    limit = max(1, backtest[['target_return', 'price_and_text']].abs().max().max() * 10_000 * 1.04)
    for axis, market in zip(axes, sorted(COLORS), strict=True):
        rows = backtest.loc[backtest['exchange'].eq(market)]
        axis.plot([-limit, limit], [-limit, limit], linestyle='--', linewidth=1, color='#677685')
        axis.scatter(rows['target_return'] * 10_000, rows['price_and_text'] * 10_000,
                     color=COLORS[market], alpha=.34, s=12, linewidths=0)
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), title=market,
                 xlabel='Actual next-bar return (bps)', ylabel='Predicted next-bar return (bps)')
        axis.grid(alpha=.15)
    return png_element(figure, 'Scatter plots of actual versus combined-model next-bar returns for HKEX and KRX')


def sentiment_chart(ranking):
    figure = Figure(figsize=(9, 3.15), layout='constrained')
    axes = figure.subplots(1, 2)
    values = ranking['sentiment_score'].dropna().abs()
    limit = max(.15, 1.2 * values.max()) if len(values) else .15
    for axis, market in zip(axes, sorted(COLORS), strict=True):
        rows = ranking.loc[ranking['exchange'].eq(market)].sort_values('sentiment_score', na_position='first')
        for ticker, value in zip(rows['ticker'], rows['sentiment_score'], strict=True):
            if pd.isna(value):
                axis.text(0, ticker, 'unscored', va='center', fontsize=8, color='#677685')
                axis.barh(ticker, 0)
            else:
                axis.barh(ticker, value, color='#19774f' if value > 0 else '#ac4751')
        axis.axvline(0, linewidth=.7, color='#677685')
        axis.set(xlim=(-limit, limit), title=market, xlabel='Mean social sentiment')
        axis.grid(axis='x', alpha=.15)
    return png_element(figure, 'Signed social sentiment scores by stock and exchange')
