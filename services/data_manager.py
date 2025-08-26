from sqlalchemy import text
from database.connection import Connection
import pandas as pd
from database.model import TMarket, TPrice
from datetime import datetime, time
from sqlalchemy.exc import IntegrityError

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

    def import_csv_to_database(self, file_path, market_id, symbol, period, date_column, open_column, high_column, low_column, close_column, volume_column, adjusted_close_column, sep=","):
        try:
            df = pd.read_csv(file_path, sep=sep, parse_dates=[date_column], dayfirst=True)
            print(df.head())

            market_query = self.session.query(TMarket).filter_by(id=market_id).first()
            if not market_query:
                market_query = TMarket(id=market_id, symbol=symbol)
                self.session.add(market_query)
                self.session.commit()

            for idx, row in df.iterrows():
                try:
                    # ISO8601
                    dt = datetime.fromisoformat(str(row[date_column]).replace('Z', '+00:00'))
                except ValueError:
                    # fallback YYYY-MM-DD
                    dt = datetime.strptime(str(row[date_column]), "%Y-%m-%d")
                
                data_date = dt.date()
                data_time = dt.time() if dt.time() != time(0, 0) else None

                price = TPrice(
                    market_id=market_id,
                    period=period,
                    data_date=data_date,
                    data_time=data_time,
                    open=row[open_column],
                    high=row[high_column],
                    low=row[low_column],
                    close=row[close_column],
                    volume=row[volume_column] if pd.notnull(row[volume_column]) else 0,
                    adjusted_close=row[adjusted_close_column]
                )
                self.session.add(price)
                try:
                    self.session.commit()
                except IntegrityError:
                    self.session.rollback()
        except Exception as e:
            self.session.rollback()
            print("Error importing CSV:", e)

    def get_data(self):
        query = self.session.query(TPrice).all()

        data = [
            {c.name: getattr(row, c.name) for c in row.__table__.columns}
            for row in query
        ]

        df = pd.DataFrame(data)
        return df

