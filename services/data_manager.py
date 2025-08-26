from sqlalchemy import text
from database.connection import Connection

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