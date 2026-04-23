from enum import Enum, auto
from typing import Dict

import numpy as np
import sqlalchemy
from ib_insync import *
import pandas as pd
from sqlalchemy import create_engine, text
import pickle
import zstandard as zstd

class SQLSTATUS(Enum):
    OK = 0
    TABLE_NOT_EXIST = auto()
    DATE_NOT_EXIST = auto()

class Database:

    def __init__(self,
                 db_name='stock',
                 db_username='metis',
                 db_password='123456'):
        self.connection_str = f'mysql+mysqlconnector://{db_username}:{db_password}@localhost/{db_name}'
        self.engine = create_engine(self.connection_str)
        self.ib = IB()

    def connectIB(self):
        if not self.ib.isConnected():
            self.ib.connect('localhost', timeout=50, port=7497, clientId=1, readonly=True)

    def disconnectIB(self):
        if self.ib.isConnected():
            self.ib.disconnect()

    def get_bars_from_ib(self, root: str, start_date: int, barSizeSetting: str):
        try:
            start_date = pd.Timestamp(str(start_date))
            query = text(f"SELECT * FROM {root} WHERE date > '{start_date}' ORDER BY date")
            df = pd.read_sql_query(query, con=self.engine.connect())
            return df
        except:
            print(f'{root} does not exist.')

        start_date = pd.Timestamp(str(start_date), tz='America/New_York')
        end_date = pd.Timestamp.now(tz='America/New_York')
        dates = pd.date_range(start=start_date, end=end_date, freq='6MS')

        dfs = []
        
        if root.upper() == 'VIX':
            contract = Index('VIX', 'CBOE', 'USD')
        elif root.upper() == 'SPX':
            contract = Index('SPX', 'CBOE', 'USD')
        else:
            contract = Stock(root, 'SMART', 'USD')

        for end_date in dates[1:]:
            bars = self.ib.reqHistoricalData(contract,
                                             endDateTime=end_date,
                                             durationStr='6 M',
                                             barSizeSetting=barSizeSetting,
                                             whatToShow='TRADES',
                                             useRTH=True,
                                             formatDate=1,
                                             timeout=0)
            trade_bars = util.df(bars)
            dfs.append(trade_bars)
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates()
        self.save(root, start_date, df)
        print('Done')

    def check_if_root_exist(self, root: str):
        with self.engine.connect() as conn:
            query = f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{root}'"
            result = conn.execute(text(query)).scalar()
            table_exists = result == 1
            if not table_exists:
                return SQLSTATUS.TABLE_NOT_EXIST
        return SQLSTATUS.OK

    def check_if_exist(self, root: str, date: pd.Timestamp):
        with self.engine.connect() as conn:
            query = f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{root}'"
            result = conn.execute(text(query)).scalar()
            table_exists = result == 1
            if not table_exists:
                return SQLSTATUS.TABLE_NOT_EXIST
            query = f"SELECT COUNT(*) FROM {root} WHERE date = '{date}'"
            result = conn.execute(text(query)).scalar()
            date_exists = result > 0
            if not date_exists:
                return SQLSTATUS.DATE_NOT_EXIST
        return SQLSTATUS.OK

    def load(self, root: str, start_date: int, end_date: int):
        start_date = pd.Timestamp(str(start_date))
        end_date = pd.Timestamp(str(end_date))
        query = text(f"SELECT * FROM {root} WHERE date >= '{start_date}' and date <= '{end_date}' ORDER BY date ASC")
        df = pd.read_sql_query(query, con=self.engine.connect())
        return df


    def save(self, root: str, date: pd.Timestamp, df: pd.DataFrame):
        if_exist = self.check_if_exist(root, date)
        if if_exist == SQLSTATUS.OK:
            print(f'{root} @ {date} exists in the database. Skipped.')
            return
    
        df.to_sql(name=root, con=self.engine, if_exists='append', index=False,
                  dtype={'date': sqlalchemy.types.DATETIME(timezone=True)})

        if if_exist == SQLSTATUS.TABLE_NOT_EXIST:
            with self.engine.connect() as conn:
                create_idx_query = f"CREATE INDEX idx_date ON {root} (date)"
                conn.execute(text(create_idx_query))
                conn.commit()

    def get_day_range(self, root: str, start_date=-1, end_date=21000000):
        start_date = pd.Timestamp(str(start_date))
        end_date = pd.Timestamp(str(end_date))
        with self.engine.connect() as conn:
            query = f"SELECT DISTINCT date AS date FROM {root} WHERE date >= '{start_date}' and date <= '{end_date}' ORDER BY date ASC "
            days = conn.execute(text(query)).fetchall()
        if len(days) == 0:
            return None
        else:
            return days[0]


if __name__ == "__main__":
    db = Database(db_name='stock', db_username='metis', db_password='123456')
    db.connectIB()
    # df = db.get_bars_from_ib(root='QQQ', start_date=20200101, barSizeSetting='1 min')
    # df = db.get_bars_from_ib(root='SPY', start_date=20200101, barSizeSetting='1 min')
    # df = db.get_bars_from_ib(root='NVDA', start_date=20200101, barSizeSetting='1 min')
    # df = db.get_bars_from_ib(root='MSFT', start_date=20200101, barSizeSetting='1 min')
    df = db.get_bars_from_ib(root='TQQQ', start_date=20200101, barSizeSetting='1 min')
    df = db.get_bars_from_ib(root='SOXL', start_date=20200101, barSizeSetting='1 min')
    

        

    # df['open_ind'] = (df['date'].dt.time == pd.Timestamp('09:30').time()).astype(int)
    # df['close_ind']= (df.groupby(df['date'].dt.date)['date'].transform('max') == df['date']).astype(int)

    # print(';a')
