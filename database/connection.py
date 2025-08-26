import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.model import model_base
from typing import Generator
from sqlalchemy.orm import Session

load_dotenv()

class Connection:
    def __init__(
        self,
        db_hostname: str = None,
        db_port: str = None,
        db_database: str = None,
        db_username: str = None,
        db_password: str = None,
        database_url: str = None
    ):
        if database_url:
            self.database_url = database_url
        else:
            self.db_host = db_hostname or os.getenv("DB_HOST")
            self.db_port = db_port or os.getenv("DB_PORT")
            self.db_database = db_database or os.getenv("DB_DATABASE")
            self.db_username = db_username or os.getenv("DB_USERNAME")
            self.db_password = db_password or os.getenv("DB_PASSWORD")

            self.database_url = f'postgresql+psycopg2://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}'

        print(f"Using database URL: {self.database_url}")

        self.engine = create_engine(
            self.database_url,
            echo=False,  # log
            pool_pre_ping=True,
            pool_recycle=300,
        )

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        self.session: Session = None

    def create_tables(self):
        try:
            model_base.metadata.create_all(bind=self.engine)
        except Exception as e:
            raise e

    def get_session(self):
        self.session = self.SessionLocal()
        return self.session
    
    def close_session(self):
        if self.session:
            self.session.close()
            self.session = None

    def get_db(self) -> Generator[Session, None, None]:
        db = self.get_session()
        try:
            yield db
        finally:
            db.close()