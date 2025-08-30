from sqlalchemy import text
from database.connection import Connection
import pandas as pd
from database.model import TMarket, TPrice
from datetime import datetime
from sqlalchemy.exc import IntegrityError
import requests
from sqlalchemy import and_
from enum import Enum

class TimeEnum(Enum):
    OneMinute = 60
    FiveMinutes = 300
    TenMinutes = 600
    FifteenMinutes = 900
    ThirtyMinutes = 1800
    OneHour = 3600
    FourHours = 14400
    OneDay = 86400
    OneWeek = 604800

class DataManager:
    def __init__(self):
        self.session = None
        self.db_connection = Connection()

    def start_session(self):
        self.session = self.db_connection.get_session()

    def close_session(self):
        self.db_connection.close_session()

    def create_tables(self):
        try:
            self.db_connection.create_tables()
            print("Tables created successfully.")
        except Exception as e:
            print("Error creating tables:", e)
    
    def drop_all_tables(self):
        query = "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
        try:
            self.session.execute(text(query))
            self.session.commit()
            print("All tables dropped successfully.")
        except Exception as e:
            self.session.rollback()
            print(f"Failed to drop tables: {e}")

    def fill_column(self, df):
        for col in ["open", "high", "low", "close", "volume", "adjusted_close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

                if col in ["volume", "adjusted_close"]:
                    df[col] = df[col].fillna(0)
    
    def print_get_period(self):
        print("ID\tName")
        for time in TimeEnum:
            print(f"{time.value}\t{time.name}")
        return input("ID: ")
    
    def import_csv_to_database(self, file_path, market_id, symbol, period, date_column, open_column, high_column, low_column, close_column, volume_column, adjusted_close_column, sep=","):
        try:
            df = pd.read_csv(file_path, sep=sep, parse_dates=[date_column], dayfirst=True)
            print(df.head())

            market_query = self.session.query(TMarket).filter_by(id=market_id).first()
            if not market_query:
                market_query = TMarket(id=market_id, symbol=symbol)
                self.session.add(market_query)
                self.session.commit()

            print("Importing...")
            for idx, row in df.iterrows():
                try:
                    # ISO8601
                    dt = datetime.fromisoformat(str(row[date_column]).replace('Z', '+00:00'))
                except ValueError:
                    # fallback YYYY-MM-DD
                    dt = datetime.strptime(str(row[date_column]), "%Y-%m-%d")
                
                timestamp = dt.date()
                print(timestamp)
                
                price = TPrice(
                    market_id=market_id,
                    period=period,
                    timestamp=timestamp,
                    open=row[open_column],
                    high=row[high_column],
                    low=row[low_column],
                    close=row[close_column],
                    volume=row[volume_column] if pd.notnull(row[volume_column]) else 0,
                    adjusted_close=row[adjusted_close_column]
                )
                self.session.merge(price)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print("Error importing CSV:", e)

    def import_json_to_database_etoro(self, market_id, period=None):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }

        market_query = self.session.query(TMarket).filter_by(id=market_id).first()
        if not market_query:
            target_url = f"https://www.etoro.com/sapi/instrumentsinfo/instruments/{market_id}/"
            response = requests.get(f"{target_url}", proxies=None, headers=headers)
            data_json = response.json()
            market_symbol = data_json["internalSymbolFull"]
            market_query = TMarket(id=market_id, symbol=market_symbol)
            self.session.add(market_query)
            self.session.commit()

        for time in TimeEnum:
            time_name = time.name
            time_value = time.value
            
            if period:
                if int(period) != int(time_value):
                    continue

            print(f"Name: {time_name}, Value: {time_value}")

            try:
                target_url = f"https://candle.etoro.com/candles/asc.json/{time_name}/1001/{market_id}/"
                response = requests.get(f"{target_url}", proxies=None, headers=headers)
                data_json = response.json()
                candles = data_json["Candles"][0]["Candles"]
                
                print("Importing...")
                for candle in candles:
                    print(candle['FromDate'])
                    price = TPrice(
                        market_id=market_id,
                        period=time_value,
                        timestamp=candle['FromDate'],
                        open=candle['Open'],
                        high=candle['High'],
                        low=candle['Low'],
                        close=candle['Close'],
                        volume=candle['Volume']
                    )
                    self.session.merge(price)
                self.session.commit()
            except Exception as e:
                self.session.rollback()
                print("Error:", e)

    def get_data(self, market_id=None, period=None):
        try:
            filters = []
            
            if market_id:
                filters.append(TPrice.market_id == market_id)
            if period:
                filters.append(TPrice.period == period)

            query = self.session.query(TPrice)
            if filters:
                query = query.filter(and_(*filters))

            results = query.all()
            
            if not results:
                raise ValueError(f"No data found for market_id={market_id} period={period}")
            
            data = [
                {c.name: getattr(row, c.name) for c in row.__table__.columns}
                for row in results
            ]

            df = pd.DataFrame(data)

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            self.fill_column(df)
            
            print(df)
            return df
        except Exception as e:
            raise RuntimeError(f"Error fetching data: {e}")
        
    def get_datasets(self, period):
        data = self.get_data(period=period)

        data.sort_values(['market_id', 'period'], inplace=True)

        # split per group
        def train_val_split(group, train_frac=0.8):
            n = len(group)
            train_end = int(n * train_frac)
            train = group.iloc[:train_end]
            val = group.iloc[train_end:]
            return train, val

        train_list = []
        val_list = []

        # group market_id + period
        for (m_id, per), group in data.groupby(['market_id', 'period']):
            train_group, val_group = train_val_split(group)
            train_list.append(train_group)
            val_list.append(val_group)

        # gabungkan
        train_df = pd.concat(train_list)
        val_df = pd.concat(val_list)

        print("\n== Train Dataset ==")
        print(train_df)
        train_counts = train_df.groupby(['market_id', 'period']).size().reset_index(name='train_count')
        print("Train counts per market_id & period:")
        print(train_counts)

        print("\n== Val Dataset ==")
        print(val_df)
        val_counts = val_df.groupby(['market_id', 'period']).size().reset_index(name='val_count')
        print("\nValidation counts per market_id & period:")
        print(val_counts)

        return train_df, val_df

