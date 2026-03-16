-- MOEX Agent v2 Database Schema
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS candles (
  secid TEXT NOT NULL,
  board TEXT NOT NULL,
  interval INTEGER NOT NULL,
  ts TEXT NOT NULL,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  value REAL,
  volume REAL,
  PRIMARY KEY (secid, board, interval, ts)
);

CREATE TABLE IF NOT EXISTS quotes (
  secid TEXT NOT NULL,
  board TEXT NOT NULL,
  ts TEXT NOT NULL,
  last REAL,
  bid REAL,
  ask REAL,
  numtrades REAL,
  voltoday REAL,
  valtoday REAL,
  PRIMARY KEY (secid, board, ts)
);

CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_ts TEXT NOT NULL,
  secid TEXT NOT NULL,
  direction TEXT NOT NULL,
  horizon TEXT NOT NULL,
  p REAL NOT NULL,
  signal_type TEXT NOT NULL,
  entry REAL,
  take REAL,
  stop REAL,
  ttl_minutes INTEGER,
  anomaly_score REAL,
  payload_json TEXT,
  sent INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_ts TEXT NOT NULL,
  secid TEXT NOT NULL,
  direction TEXT NOT NULL,
  entry_price REAL NOT NULL,
  exit_price REAL,
  pnl REAL,
  status TEXT NOT NULL,
  horizon TEXT
);

CREATE TABLE IF NOT EXISTS state (
  key TEXT PRIMARY KEY,
  value TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_candles_ts ON candles(ts);
CREATE INDEX IF NOT EXISTS idx_candles_secid ON candles(secid);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_ts);
CREATE INDEX IF NOT EXISTS idx_trades_created ON trades(created_ts);
