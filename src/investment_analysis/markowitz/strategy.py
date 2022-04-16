import logging
import pathlib
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas_datareader.data as web

from zipline import api
from zipline import protocol
from zipline import run_algorithm
from zipline.algorithm import TradingAlgorithm

from investment_analysis.markowitz import computation
from investment_analysis.zipline_ingester import VANGUARD_UNIVERSE
from investment_analysis.markowitz.symbols import TICKERS


logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def compute_moving_averages(context, data):
    short_term = {
        f'{asset.symbol}_{context.short_term}_mvavg': val for asset, val in \
                data.history(
                    assets=list(filter(data.can_trade, context.assets)),
                    bar_count=context.short_term,
                    fields=[context.price_col],
                    frequency='1d'
                ) \
                .unstack() \
                .mean()[context.price_col] \
                .to_dict() \
                .items()
    }

    long_term = {
        f'{asset.symbol}_{context.long_term}_mvavg': val for asset, val in \
                data.history(
                    assets=list(filter(data.can_trade, context.assets)),
                    bar_count=context.long_term,
                    fields=[context.price_col],
                    frequency='1d'
                ) \
                .unstack() \
                .mean()[context.price_col] \
                .to_dict() \
                .items()
    }

    api.record(**short_term, **long_term)


def initialize(context: TradingAlgorithm):
    context.tick = 0
    context.window_size = 30
    context.rebal_interval = 30
    context.price_col = 'close'
    context.assets = list(map(api.symbol, TICKERS))

    context.short_term = 20
    context.long_term = 100


def handle_data(context: TradingAlgorithm, data: protocol.BarData):
    # Allow history to accumulate 100 days of prices before trading
    # and rebalance every day thereafter.
    context.tick += 1
    #if context.tick >= context.long_term:
    #    compute_moving_averages(context, data)
    if context.tick < context.window_size:
        return
    if context.tick % context.rebal_interval != 0:
        # rebalance every 30 days
        return
    # Get rolling window of past prices and compute returns
    prices = data.history(
            assets=list(filter(data.can_trade, context.assets)),
            bar_count=context.window_size,
            fields=[context.price_col],
            frequency='1d')
    if prices.empty:
        return
    prices = prices[context.price_col]
    returns: pd.DataFrame = prices \
                .pct_change() \
                .unstack() \
                .dropna()
    # Perform Markowitz-style portfolio optimization

    try:
        weights, _, _ = computation.optimal_portfolio(returns.T)
    except ValueError as ex:
        logger.warning("failed to compute optimal portfolio %s", ex.args)
        return
    # Rebalance portfolio accordingly
    #weights = [[1] for _ in context.assets]
    out = {}
    weights = np.array([w[0] for w in weights])
    normalized_weights = weights/sum(weights)
    for stock, weight in zip(returns.columns, normalized_weights):
        out[stock.symbol] = weight
        api.order_target_percent(stock, weight)
    api.record(**out)


if __name__ == '__main__':

    start = pd.Timestamp('2021-04-10')
    end = pd.Timestamp('2022-04-08')
    logger.setLevel(logging.INFO)

    sp500 = web.DataReader('SP500', 'fred', start, end).SP500
    benchmark_returns = sp500.pct_change()

    result: pd.DataFrame = run_algorithm(
            start=start.tz_localize('UTC'),
            end=end.tz_localize('UTC'),
            initialize=initialize,
            handle_data=handle_data,
            #analyze=analyze,
            benchmark_returns=benchmark_returns,
            capital_base=27000,
            #bundle='yahoo-finance-universe',
            bundle='quandl',
            data_frequency='daily'
    )

    result.to_pickle('./backtest.pickle')
