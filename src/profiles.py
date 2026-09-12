"""SQLite-backed user profiles + conversation memory.

Schema:
  users(id, name, embedding BLOB, n_utts, spread REAL, prefs_json, created_at)
  turns(id, user_id, role, text, score, ts)
"""
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .config import path as cfg_path

EMB_DTYPE = np.float32


@dataclass
class User:
    id: int
    name: str
    centroid: np.ndarray
    n_utts: int
    spread: float
    prefs: dict
    created_at: str


def _connect():
    db_path = cfg_path("db")
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            embedding BLOB NOT NULL,
            n_utts INTEGER NOT NULL,
            spread REAL NOT NULL DEFAULT 0.0,
            prefs_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns(
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            score REAL,
            ts TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS summaries(
            user_id INTEGER PRIMARY KEY,
            summary TEXT NOT NULL DEFAULT '',
            updated_at TEXT
        )
    """)
    return conn


def _row_to_user(row) -> User:
    id_, name, emb_blob, n_utts, spread, prefs_json, created_at = row
    centroid = np.frombuffer(emb_blob, dtype=EMB_DTYPE).copy()
    return User(id=id_, name=name, centroid=centroid, n_utts=n_utts,
                spread=spread, prefs=json.loads(prefs_json), created_at=created_at)


def add_user(name: str, centroid: np.ndarray, n_utts: int, spread: float,
             prefs: dict | None = None) -> User:
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    emb_blob = centroid.astype(EMB_DTYPE).tobytes()
    prefs = prefs or {}
    with conn:
        conn.execute(
            "INSERT INTO users(name, embedding, n_utts, spread, prefs_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, emb_blob, n_utts, spread, json.dumps(prefs), now),
        )
    conn.close()
    return get_user_by_name(name)


def update_user_embedding(user_id: int, centroid: np.ndarray, n_utts: int, spread: float):
    conn = _connect()
    with conn:
        conn.execute(
            "UPDATE users SET embedding=?, n_utts=?, spread=? WHERE id=?",
            (centroid.astype(EMB_DTYPE).tobytes(), n_utts, spread, user_id),
        )
    conn.close()


def get_all_users() -> list[User]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, name, embedding, n_utts, spread, prefs_json, created_at FROM users"
    ).fetchall()
    conn.close()
    return [_row_to_user(r) for r in rows]


def get_user_by_name(name: str) -> User | None:
    conn = _connect()
    row = conn.execute(
        "SELECT id, name, embedding, n_utts, spread, prefs_json, created_at "
        "FROM users WHERE name=?", (name,)
    ).fetchone()
    conn.close()
    return _row_to_user(row) if row else None


def delete_user(name: str):
    conn = _connect()
    with conn:
        u = conn.execute("SELECT id FROM users WHERE name=?", (name,)).fetchone()
        if u:
            conn.execute("DELETE FROM users WHERE id=?", (u[0],))
            conn.execute("DELETE FROM turns WHERE user_id=?", (u[0],))
            conn.execute("DELETE FROM summaries WHERE user_id=?", (u[0],))
    conn.close()


def log_turn(user_id: int | None, role: str, text: str, score: float | None = None):
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            "INSERT INTO turns(user_id, role, text, score, ts) VALUES (?, ?, ?, ?, ?)",
            (user_id, role, text, score, now),
        )
    conn.close()


def recent_turns(user_id: int, limit: int = 10) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT role, text, ts FROM turns WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [{"role": r, "text": t, "ts": ts} for r, t, ts in reversed(rows)]


def get_summary(user_id: int) -> str:
    conn = _connect()
    row = conn.execute("SELECT summary FROM summaries WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return row[0] if row else ""


def set_summary(user_id: int, summary: str):
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        conn.execute(
            "INSERT INTO summaries(user_id, summary, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at",
            (user_id, summary, now),
        )
    conn.close()
