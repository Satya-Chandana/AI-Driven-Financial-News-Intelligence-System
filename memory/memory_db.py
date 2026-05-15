"""SQLite-backed persistent memory for trading decisions and post-mortems."""
import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

DB_PATH = os.getenv("TRADINGAGENTS_DB_PATH", "tradingagents_memory.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            decision_date   TEXT NOT NULL,
            evaluate_after  TEXT NOT NULL,
            entry_price     REAL,
            final_decision  TEXT NOT NULL,
            conviction      INTEGER,
            bias            TEXT,
            confidence      TEXT,
            synthesis       TEXT,
            bull_argument   TEXT,
            bear_argument   TEXT,
            arbiter_verdict TEXT,
            risk_notes      TEXT,
            evaluated       INTEGER DEFAULT 0,
            outcome_price   REAL,
            outcome_return  REAL,
            outcome_label   TEXT,
            postmortem      TEXT,
            created_at      TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON decisions(ticker)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_eval ON decisions(evaluated, evaluate_after)")
    conn.commit()
    conn.close()


def save_decision(record: Dict) -> int:
    init_db()
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO decisions
            (ticker, decision_date, evaluate_after, entry_price,
             final_decision, conviction, bias, confidence, synthesis,
             bull_argument, bear_argument, arbiter_verdict, risk_notes,
             created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["ticker"],
        record["decision_date"],
        record["evaluate_after"],
        record.get("entry_price"),
        record["final_decision"],
        record.get("conviction"),
        record.get("bias"),
        record.get("confidence"),
        record.get("synthesis"),
        record.get("bull_argument"),
        record.get("bear_argument"),
        record.get("arbiter_verdict"),
        record.get("risk_notes"),
        datetime.utcnow().isoformat(),
    ))
    decision_id = cur.lastrowid
    conn.commit()
    conn.close()
    return decision_id


def get_recent_decisions(ticker: str, limit: int = 5) -> List[Dict]:
    init_db()
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM decisions
        WHERE ticker = ?
        ORDER BY decision_date DESC
        LIMIT ?
    """, (ticker.upper(), limit))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_pending_evaluations(as_of_date: Optional[str] = None) -> List[Dict]:
    """Return decisions whose T+5 evaluation window has elapsed but are not yet evaluated."""
    init_db()
    as_of_date = as_of_date or datetime.utcnow().date().isoformat()
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM decisions
        WHERE evaluated = 0 AND evaluate_after <= ?
        ORDER BY evaluate_after ASC
    """, (as_of_date,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def update_postmortem(decision_id: int, outcome_price: float,
                      outcome_return: float, outcome_label: str,
                      postmortem: str) -> None:
    init_db()
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        UPDATE decisions
           SET evaluated = 1,
               outcome_price = ?,
               outcome_return = ?,
               outcome_label = ?,
               postmortem = ?
         WHERE id = ?
    """, (outcome_price, outcome_return, outcome_label, postmortem, decision_id))
    conn.commit()
    conn.close()


def add_trading_days(start_date: str, n_days: int = 5) -> str:
    """Approximate T+N trading days using a 7/5 calendar conversion."""
    d = datetime.fromisoformat(start_date).date() if "T" not in start_date else \
        datetime.fromisoformat(start_date).date()
    # rough: skip weekends
    added = 0
    cur = d
    while added < n_days:
        cur = cur + timedelta(days=1)
        if cur.weekday() < 5:
            added += 1
    return cur.isoformat()
