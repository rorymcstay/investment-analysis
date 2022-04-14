""""""
from datetime import datetime
import logging
from typing import Dict

import pytz

import yfinance as yf
import pandas as pd
import numpy as np

from zipline.data.minute_bars import BcolzMinuteBarWriter
from zipline.data.bcolz_daily_bars import BcolzDailyBarWriter
from zipline.data.adjustments import SQLiteAdjustmentWriter
from zipline.utils.calendar_utils import TradingCalendar
from zipline.assets import AssetDBWriter, Asset


logger = logging.getLogger(__name__)


VANGUARD_UNIVERSE = {
    'VUKE.L': 1,
    'VMID.L': 2,
    'VUSA.L': 3,
    'VERX.L': 4,
    'VGER.L': 6,
    'VWRL.L': 8,
    'VHYL.L': 9,
    'VEVE.L': 10,
    'VJPN.L': 11,
    'VAPX.L': 12,
    'VNRT.L': 13,
    'VFEM.L': 14,
    'VECP.L': 15,
    'VETY.L': 16,
    'VAGP.L': 17,
    'VGOV.L': 18,
    'VUCP.L': 19,
    'VUTY.L': 20,
    'VEMT.L': 21,

    #'^FTSE': 22,
    #'^GSPC': 23,
    #'^FTMC': 24,
    #'^GDAXI': 25,

}

PERIOD_YEARS = 5

def df_cache_key(name, period, symbol):
    now = datetime.now()
    return f'{name}_{period}_{symbol}_{now.year}_{now.month}_{now.day}'


def _yield_yahoo_finance_data(
        equities: pd.DataFrame,
        universe: yf.Tickers,
        cache: Dict[str, pd.DataFrame],
        calendar: TradingCalendar,
    ):

    for asset in equities.itertuples():
        ticker = universe.tickers[asset.symbol]
        symbol = asset.symbol
        cache_key = df_cache_key('history', PERIOD_YEARS, symbol)
        data = ticker.history(f'{PERIOD_YEARS}y') if cache_key not in cache else cache[cache_key]
        data.columns = [c.lower().replace(' ', '_') for c in data.columns]
        ohlc: pd.DataFrame = data[['open', 'high', 'low', 'close', 'volume']]
        sessions = calendar.sessions_in_range(asset.start_date, asset.end_date)
        ohlc_clean = ohlc.dropna()
        ohlc_clean.index = ohlc_clean.index.tz_localize(pytz.utc)
        ohlc_clean = ohlc_clean.reindex(sessions)
        indexer = sessions.slice_indexer(asset.start_date, asset.end_date)

        yield (asset.Index,ohlc_clean)


def yahoo_finance(environ: Dict[str, str],
       asset_db_writer: AssetDBWriter,
       minute_bar_writer: BcolzMinuteBarWriter,
       daily_bar_writer: BcolzDailyBarWriter,
       adjustment_writer: SQLiteAdjustmentWriter,
       calendar: TradingCalendar,
       start_session: pd.Timestamp,
       end_session: pd.Timestamp,
       cache: Dict[str, pd.DataFrame],
       show_progress: bool,
       output_dir):
    """"""

    universe = yf.Tickers(list(VANGUARD_UNIVERSE.keys()))
    assets = []
    info_cache_key = f'{PERIOD_YEARS}y_{str(" ".join(tuple(k for k in universe.tickers.keys())))}'
    if info_cache_key not in cache:
        ticker: yf.Ticker
        for symbol, ticker in universe.tickers.items():
            info = ticker.get_info()
            key = df_cache_key('history', PERIOD_YEARS, symbol)
            history = cache[key] if key in cache else ticker.history(f'{PERIOD_YEARS}y')
            cache[key] = history
            assets.append({
                'sid': VANGUARD_UNIVERSE[symbol],
                'symbol': symbol,
                'asset_name': info['shortName'],
                'exchange': info['exchange'],
                'start_date': history.iloc[[0]].index[0],
                'first_traded': history.iloc[[0]].index[0],
                'end_date': history.iloc[[-1]].index[0]
            })
        equities = pd.DataFrame(data=assets)
    else:
        equities = cache[info_cache_key]
    cache[info_cache_key] = equities
    equities.index = equities.sid
    print(equities)
    asset_db_writer.write(equities=equities, chunk_size=100)

    daily_bar_writer.write(_yield_yahoo_finance_data(
            equities, universe, cache, calendar
        ), invalid_data_behavior='warn')

    for symbol, ticker in universe.tickers.items():
        key = df_cache_key('history', PERIOD_YEARS, symbol)
        data = cache[key] if key in cache else ticker.history(f'{PERIOD_YEARS}y')
        data.columns = [c.lower().replace(' ', '_') for c in data.columns]
        stock_splits: pd.DataFrame = data[data.stock_splits != 0.0][['stock_splits']].dropna()
        dividends: pd.Series = data[data['dividends'] > 0]['dividends'].squeeze().dropna()
        if not stock_splits.empty:
            stock_splits['sid'] = VANGUARD_UNIVERSE[ticker]
            stock_splits['ratio'] = stock_splits.stock_splits
            stock_splits['effective_date'] = stock_splits.index.view(np.int64)
            stock_splits = stock_splits[['ratio', 'sid', 'effective_date']]
            stock_splits = stock_splits.reset_index(drop=True)
            stock_splits.index = stock_splits.sid
        if not dividends.empty:
            dividends_ = pd.DataFrame()
            dividends_['pay_date'] = dividends.index
            dividends_['ex_date'] = dividends.index
            dividends_['declared_date'] = dividends.index
            dividends_['record_date'] = dividends.index
            dividends_['amount'] = dividends.values
            dividends_['sid'] = VANGUARD_UNIVERSE[ticker]
            dividends=dividends_
        adjustment_writer.write(splits=stock_splits if not stock_splits.empty else None,
                                dividends=dividends if not dividends.empty else None)


