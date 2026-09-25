"""Enrolled voiceprints, stored in a local SQLite file (data/profiles.db).

Voiceprints are biometric data: the file is gitignored and must stay on the
machine that recorded it. Remove someone with
    python -m src.profiles delete <name>

Schema:
  speakers(id, name UNIQUE, embedding BLOB, n_chunks, spread, speech_sec, created_at)
"""
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .config import path as cfg_path

EMB_DTYPE = np.float32


@dataclass
class Profile:
    id: int
    name: str
    centroid: np.ndarray   # unit-norm ECAPA embedding
    n_chunks: int          # enrollment chunks that survived outlier rejection
    spread: float          # mean cosine chunk->centroid; low = inconsistent enrollment
    speech_sec: float
    created_at: str


def _connect():
    conn = sqlite3.connect(str(cfg_path("db")))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS speakers(
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            embedding BLOB NOT NULL,
            n_chunks INTEGER NOT NULL,
            spread REAL NOT NULL,
            speech_sec REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    return conn


_COLS = "id, name, embedding, n_chunks, spread, speech_sec, created_at"


def _row(row) -> Profile:
    id_, name, blob, n_chunks, spread, speech_sec, created_at = row
    return Profile(id_, name, np.frombuffer(blob, dtype=EMB_DTYPE).copy(),
                   n_chunks, spread, speech_sec, created_at)


def upsert(name: str, centroid: np.ndarray, n_chunks: int, spread: float,
           speech_sec: float) -> Profile:
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO speakers(name, embedding, n_chunks, spread, speech_sec, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
            "embedding=excluded.embedding, n_chunks=excluded.n_chunks, "
            "spread=excluded.spread, speech_sec=excluded.speech_sec, "
            "created_at=excluded.created_at",
            (name, centroid.astype(EMB_DTYPE).tobytes(), n_chunks, spread, speech_sec,
             datetime.now(timezone.utc).isoformat()),
        )
    conn.close()
    return get(name)


def get(name: str) -> Profile | None:
    conn = _connect()
    row = conn.execute(f"SELECT {_COLS} FROM speakers WHERE name=?", (name,)).fetchone()
    conn.close()
    return _row(row) if row else None


def get_all() -> list[Profile]:
    conn = _connect()
    rows = conn.execute(f"SELECT {_COLS} FROM speakers ORDER BY name").fetchall()
    conn.close()
    return [_row(r) for r in rows]


def delete(name: str) -> bool:
    conn = _connect()
    with conn:
        n = conn.execute("DELETE FROM speakers WHERE name=?", (name,)).rowcount
    conn.close()
    return n > 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Manage enrolled speakers.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    d = sub.add_parser("delete")
    d.add_argument("name")
    args = p.parse_args()

    if args.cmd == "list":
        profiles = get_all()
        if not profiles:
            print("No speakers enrolled yet. Run: python -m src.enroll --name <Name>")
        for pr in profiles:
            print(f"{pr.name:<20} speech={pr.speech_sec:5.1f}s  chunks={pr.n_chunks:3d}  "
                  f"spread={pr.spread:.3f}  enrolled={pr.created_at[:19]}")
    else:
        print("Deleted." if delete(args.name) else f"No speaker named {args.name!r}.")
