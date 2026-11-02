"""Content-addressed sqlite cache for LLM calls: makes every experiment resumable and cheap to rerun."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from typing import Any


def key_for(**kw: Any) -> str:
    return hashlib.sha256(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()


class Cache:
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, k: str) -> Any | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, k: str, v: Any) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
            self._db.commit()
