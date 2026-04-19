import sqlite3
from pathlib import Path
from typing import List, Literal

from app.core.config import CHAT_MEMORY_DB, logger
from app.schemas.chat_schema import Message

Role = Literal["user", "ai"]

DB_PATH = CHAT_MEMORY_DB

_HISTORY_FETCH_LIMIT = 6
_MAX_ROWS_PER_USER = 300


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'ai')),
                text TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_user_id_id ON messages(user_id, id)"
        )
        conn.commit()
    logger.info("SQLite message store ready at %s", DB_PATH)


def _prune_user(conn: sqlite3.Connection, user_id: str) -> None:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    count = int(row[0]) if row else 0
    excess = count - _MAX_ROWS_PER_USER
    if excess <= 0:
        return
    cur.execute(
        """
        SELECT id FROM messages
        WHERE user_id = ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (user_id, excess),
    )
    old_ids = [r[0] for r in cur.fetchall()]
    if not old_ids:
        return
    placeholders = ",".join("?" * len(old_ids))
    cur.execute(
        f"DELETE FROM messages WHERE user_id = ? AND id IN ({placeholders})",
        [user_id, *old_ids],
    )


def save_message(user_id: str, role: Role, text: str) -> int:
    uid = (user_id or "default").strip() or "default"
    if role not in ("user", "ai"):
        raise ValueError("role must be 'user' or 'ai'")
    body = (text or "").strip()
    if not body:
        raise ValueError("text must be non-empty")

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO messages (user_id, role, text) VALUES (?, ?, ?)",
            (uid, role, body),
        )
        mid = int(cur.lastrowid)
        _prune_user(conn, uid)
        conn.commit()

    print(
        "[memory] saved message",
        {"id": mid, "user_id": uid, "role": role, "text_preview": body[:120]},
        flush=True,
    )
    logger.info("Saved message id=%s user_id=%s role=%s", mid, uid, role)
    return mid


def get_last_messages(user_id: str, limit: int = _HISTORY_FETCH_LIMIT) -> List[Message]:
    uid = (user_id or "default").strip() or "default"
    lim = max(1, min(limit, _HISTORY_FETCH_LIMIT))

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT role, text, id, timestamp
            FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (uid, lim),
        )
        rows = cur.fetchall()

    rows_chrono = list(reversed(rows))
    out: List[Message] = [Message(role=str(r["role"]), text=str(r["text"])) for r in rows_chrono]

    print(
        "[memory] loaded history",
        {"user_id": uid, "count": len(out), "messages": [m.model_dump() for m in out]},
        flush=True,
    )
    logger.info("Loaded %s messages for user_id=%s", len(out), uid)
    return out
