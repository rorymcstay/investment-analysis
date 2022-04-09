from typing import Dict
from zipline.data.minute_bars import BcolzMinuteBarWriter
from zipline.data.bcolz_daily_bars import BcolzDailyBarWriter
from zipline.data.adjustments import SQLiteAdjustmentWriter
from zipline.utils.calendar_utils import TradingCalendar
import yfinance as yf
import pandas as pd


from zipline.assets import AssetDBWriter


def ingest(environ: Dict[str, str],
       asset_db_writer: AssetDBWriter,
       minute_bar_writer: minute_bars.BcolzMinuteBarWriter,
       daily_bar_writer: BcolzDailyBarWriter,
       adjustment_writer: SQLiteAdjustmentWriter,
       calendar: TradingCalendar,
       start_session: pd.Timestamp,
       end_session: pd.Timestamp,
       cache: Dict[str, pd.DataFrame],
       show_progress: bool,
       output_dir):
    """"""
    universe = yf.Tickers([
        'VMID.L',
        'VUKE.L',
    ])
    daily_bar_writer.write(
            ((ticker, asset.history('5y')) for ticker, asset in universe.tickers.items()),
            set(universe.tickers.keys()),
            show_progress=True,
            invalid_data_behavior='raise'
            )
