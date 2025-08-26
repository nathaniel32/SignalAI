from sqlalchemy import Column, Integer, String, Numeric, Date, Time, ForeignKey, PrimaryKeyConstraint
from sqlalchemy.orm import relationship, declarative_base
from datetime import time

model_base = declarative_base()

class TMarket(model_base):
    __tablename__ = 't_market'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)

    prices = relationship("TPrice", back_populates="market", cascade="all, delete-orphan")

class TPrice(model_base):
    __tablename__ = 't_price'

    market_id = Column(Integer, ForeignKey('t_market.id', ondelete='CASCADE', onupdate='CASCADE'), nullable=False)
    period = Column(Integer, nullable=False)
    data_date = Column(Date, nullable=False)
    data_time = Column(Time(0), nullable=False, default=time(0, 0))
    open = Column(Numeric(18,8), nullable=False)
    high = Column(Numeric(18,8), nullable=False)
    low = Column(Numeric(18,8), nullable=False)
    close = Column(Numeric(18,8), nullable=False)
    volume = Column(Numeric(20,8))
    adjusted_close = Column(Numeric(18,8))

    __table_args__ = (
        PrimaryKeyConstraint('market_id', 'period', 'data_date', 'data_time', name='pk_t_price'),
    )

    market = relationship("TMarket", back_populates="prices")