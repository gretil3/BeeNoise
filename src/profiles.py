"""SQLite-backed student profiles + attendance records.

Schema:
  users(id, name, student_id, embedding BLOB, n_utts, spread, created_at)
  attendance_attempts(id, user_id, session, attempt_number, expected_digits,
                       transcribed_digits, speaker_score, speaker_pass,
                       content_pass, passed, ts)
  attendance_log(id, user_id, session, ts, speaker_score, attempts_used)
  attendance_state(user_id, session, fail_count, locked_until)

`attendance_attempts` is the raw evaluation trail (every attempt, pass or
fail) — that's what the report's FAR/FRR numbers come from. `attendance_log`
holds one row per (student, session): the first successful verification.
`attendance_state` tracks the retry/lockout cycle described in the
blueprint: `attendance.max_attempts` failed attempts in a row lock a student
out of retrying for `attendance.lockout_minutes`.
"""
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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
    student_id: str | None
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect():
    db_path = cfg_path("db")
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            student_id TEXT,
            embedding BLOB NOT NULL,
            n_utts INTEGER NOT NULL,
            spread REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance_attempts(
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            session TEXT NOT NULL,
            attempt_number INTEGER NOT NULL,
            expected_digits TEXT NOT NULL,
            transcribed_digits TEXT,
            speaker_score REAL,
            speaker_pass INTEGER NOT NULL,
            content_pass INTEGER NOT NULL,
            passed INTEGER NOT NULL,
            ts TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance_log(
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            session TEXT NOT NULL,
            ts TEXT NOT NULL,
            speaker_score REAL NOT NULL,
            attempts_used INTEGER NOT NULL,
            UNIQUE(user_id, session)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance_state(
            user_id INTEGER NOT NULL,
            session TEXT NOT NULL,
            fail_count INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT,
            PRIMARY KEY(user_id, session)
        )
    """)
    return conn


def _row_to_user(row) -> User:
    id_, name, student_id, emb_blob, n_utts, spread, created_at = row
    centroid = np.frombuffer(emb_blob, dtype=EMB_DTYPE).copy()
    return User(id=id_, name=name, centroid=centroid, n_utts=n_utts,
                spread=spread, student_id=student_id, created_at=created_at)


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

def add_user(name: str, centroid: np.ndarray, n_utts: int, spread: float,
             student_id: str | None = None) -> User:
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO users(name, student_id, embedding, n_utts, spread, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, student_id, centroid.astype(EMB_DTYPE).tobytes(), n_utts, spread, _now()),
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
        "SELECT id, name, student_id, embedding, n_utts, spread, created_at FROM users "
        "ORDER BY name"
    ).fetchall()
    conn.close()
    return [_row_to_user(r) for r in rows]


def get_user_by_name(name: str) -> User | None:
    conn = _connect()
    row = conn.execute(
        "SELECT id, name, student_id, embedding, n_utts, spread, created_at "
        "FROM users WHERE name=?", (name,)
    ).fetchone()
    conn.close()
    return _row_to_user(row) if row else None


def find_user_ci(name: str) -> User | None:
    """Case-insensitive lookup — used when a claimed name is typed/spoken
    slightly differently than it was enrolled ('sam' vs 'Sam')."""
    conn = _connect()
    row = conn.execute(
        "SELECT id, name, student_id, embedding, n_utts, spread, created_at "
        "FROM users WHERE lower(name)=lower(?)", (name.strip(),)
    ).fetchone()
    conn.close()
    return _row_to_user(row) if row else None


def delete_user(name: str):
    conn = _connect()
    with conn:
        u = conn.execute("SELECT id FROM users WHERE name=?", (name,)).fetchone()
        if u:
            uid = u[0]
            conn.execute("DELETE FROM users WHERE id=?", (uid,))
            conn.execute("DELETE FROM attendance_attempts WHERE user_id=?", (uid,))
            conn.execute("DELETE FROM attendance_log WHERE user_id=?", (uid,))
            conn.execute("DELETE FROM attendance_state WHERE user_id=?", (uid,))
    conn.close()


# ---------------------------------------------------------------------------
# Attendance attempts — the raw trail behind every verification, pass or fail
# ---------------------------------------------------------------------------

def record_attempt(user_id: int, session: str, attempt_number: int, expected_digits: str,
                    transcribed_digits: str, speaker_score: float, speaker_pass: bool,
                    content_pass: bool, passed: bool):
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO attendance_attempts(user_id, session, attempt_number, "
            "expected_digits, transcribed_digits, speaker_score, speaker_pass, "
            "content_pass, passed, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, session, attempt_number, expected_digits, transcribed_digits,
             speaker_score, int(speaker_pass), int(content_pass), int(passed), _now()),
        )
    conn.close()


def get_attempts(user_id: int, session: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT attempt_number, expected_digits, transcribed_digits, speaker_score, "
        "speaker_pass, content_pass, passed, ts FROM attendance_attempts "
        "WHERE user_id=? AND session=? ORDER BY id",
        (user_id, session),
    ).fetchall()
    conn.close()
    keys = ["attempt_number", "expected_digits", "transcribed_digits", "speaker_score",
            "speaker_pass", "content_pass", "passed", "ts"]
    return [dict(zip(keys, r)) for r in rows]


# ---------------------------------------------------------------------------
# Attendance state — retry counter + lockout, per (student, session)
# ---------------------------------------------------------------------------

def get_attendance_state(user_id: int, session: str) -> dict:
    conn = _connect()
    row = conn.execute(
        "SELECT fail_count, locked_until FROM attendance_state WHERE user_id=? AND session=?",
        (user_id, session),
    ).fetchone()
    conn.close()
    if row is None:
        return {"fail_count": 0, "locked_until": None}
    return {"fail_count": row[0], "locked_until": row[1]}


def record_failure(user_id: int, session: str, max_attempts: int, lockout_minutes: int) -> dict:
    """Increments the fail counter; once it reaches max_attempts, sets a
    lockout timestamp lockout_minutes in the future. Returns the new state."""
    state = get_attendance_state(user_id, session)
    fail_count = state["fail_count"] + 1
    locked_until = state["locked_until"]
    if fail_count >= max_attempts:
        locked_until = (datetime.now(timezone.utc) + timedelta(minutes=lockout_minutes)).isoformat()

    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO attendance_state(user_id, session, fail_count, locked_until) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, session) DO UPDATE SET "
            "fail_count=excluded.fail_count, locked_until=excluded.locked_until",
            (user_id, session, fail_count, locked_until),
        )
    conn.close()
    return {"fail_count": fail_count, "locked_until": locked_until}


def reset_attendance_state(user_id: int, session: str):
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO attendance_state(user_id, session, fail_count, locked_until) "
            "VALUES (?, ?, 0, NULL) "
            "ON CONFLICT(user_id, session) DO UPDATE SET fail_count=0, locked_until=NULL",
            (user_id, session),
        )
    conn.close()


# ---------------------------------------------------------------------------
# Attendance log — one row per (student, session): the first successful check
# ---------------------------------------------------------------------------

def mark_present(user_id: int, session: str, speaker_score: float, attempts_used: int):
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO attendance_log(user_id, session, ts, speaker_score, attempts_used) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, session) DO UPDATE SET ts=excluded.ts, "
            "speaker_score=excluded.speaker_score, attempts_used=excluded.attempts_used",
            (user_id, session, _now(), speaker_score, attempts_used),
        )
    conn.close()


def is_present(user_id: int, session: str) -> bool:
    conn = _connect()
    row = conn.execute(
        "SELECT 1 FROM attendance_log WHERE user_id=? AND session=?", (user_id, session)
    ).fetchone()
    conn.close()
    return row is not None


def get_attendance(session: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT u.name, a.ts, a.speaker_score, a.attempts_used FROM attendance_log a "
        "JOIN users u ON u.id = a.user_id WHERE a.session=? ORDER BY a.ts",
        (session,),
    ).fetchall()
    conn.close()
    return [{"name": n, "ts": ts, "speaker_score": s, "attempts_used": au}
            for n, ts, s, au in rows]
