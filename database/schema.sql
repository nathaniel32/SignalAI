CREATE TABLE t_market (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(100)
);

CREATE TABLE t_price (
    market_id INTEGER NOT NULL,
    period INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    open NUMERIC(18,8) NOT NULL,
    high NUMERIC(18,8) NOT NULL,
    low NUMERIC(18,8) NOT NULL,
    close NUMERIC(18,8) NOT NULL,
    volume NUMERIC(20,8),
    adjusted_close NUMERIC(18,8),
    PRIMARY KEY (market_id, period, timestamp),
    CONSTRAINT fk_market FOREIGN KEY (market_id) REFERENCES t_market(id) ON DELETE CASCADE ON UPDATE CASCADE
);