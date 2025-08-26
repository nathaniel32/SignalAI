CREATE TABLE t_market (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL
);

CREATE TABLE t_price (
    market_id INTEGER NOT NULL,
    period INTEGER NOT NULL,
    data_date DATE NOT NULL,
    data_time TIME(0) NOT NULL,
    open NUMERIC(18,8) NOT NULL,
    high NUMERIC(18,8) NOT NULL,
    low NUMERIC(18,8) NOT NULL,
    close NUMERIC(18,8) NOT NULL,
    volume NUMERIC(20,8),
    adjusted_close NUMERIC(18,8),
    PRIMARY KEY (market_id, period, data_date, data_time),
    CONSTRAINT fk_market FOREIGN KEY (market_id) REFERENCES t_market(id) ON DELETE CASCADE ON UPDATE CASCADE
);