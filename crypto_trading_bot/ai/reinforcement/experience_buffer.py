"""Experience buffer for storing and sampling RL training data."""

from __future__ import annotations

import json
import math
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

try:
    import aiosqlite
    _HAS_AIOSQLITE = True
except ImportError:
    import sqlite3 as _sqlite3_fallback  # noqa: F401
    _HAS_AIOSQLITE = False


class ExperienceBuffer:
    """Ring buffer for storing (state, action, reward, next_state) tuples.

    Supports prioritized replay with exponential decay on older experiences,
    and persistence to SQLite for durability across restarts.

    Usage:
        buf = ExperienceBuffer(max_size=10000)
        await buf.initialize()   # loads persisted data from DB
    """

    def __init__(
        self,
        max_size: int = 10000,
        priority_alpha: float = 0.6,
        db_path: Optional[Path] = None,
    ) -> None:
        """Initialize experience buffer.

        Args:
            max_size: Maximum number of experiences to store.
            priority_alpha: Exponent for prioritized replay (0 = uniform, 1 = full priority).
            db_path: Path to SQLite database for persistence. If None, uses in-memory only.
        """
        self.max_size = max_size
        self.priority_alpha = priority_alpha
        self.db_path = db_path or Path("data") / "rl_experience.db"

        # In-memory ring buffer
        self._buffer: deque = deque(maxlen=max_size)
        # Priorities for sampling (parallel to _buffer)
        self._priorities: deque = deque(maxlen=max_size)

        # SQLite connection (lazy init); may be aiosqlite.Connection or sqlite3.Connection
        self._conn: Optional[Any] = None
        # NOTE: Do NOT call _load_from_db here; await initialize() instead.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load persisted experiences from the database.

        Must be awaited after construction before using save/load.
        """
        if self.db_path.exists():
            await self._load_from_db()

    def add(
        self,
        state: Dict[str, float],
        action: int,
        reward: float,
        next_state: Dict[str, float],
        done: bool = False,
    ) -> None:
        """Add a new experience tuple to the buffer.

        Args:
            state: Context feature dict at time t.
            action: Strategy index selected.
            reward: Shaped reward received.
            next_state: Context feature dict at time t+1.
            done: Whether episode terminated (not used in continuous trading).
        """
        # Reject invalid rewards
        if not isinstance(reward, (int, float)) or math.isnan(reward) or math.isinf(reward):
            logger.warning("ExperienceBuffer: rejecting invalid reward: {}", reward)
            return

        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        self._buffer.append(experience)

        # New experiences get highest priority
        max_priority = max(self._priorities) if self._priorities else 1.0
        self._priorities.append(max_priority)

        logger.debug(
            f"ExperienceBuffer: added experience (action={action}, reward={reward:.3f}), "
            f"buffer size={len(self._buffer)}"
        )

    def sample(self, batch_size: int, prioritized: bool = True) -> List[Dict]:
        """Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample.
            prioritized: If True, sample with priority weights. If False, uniform.

        Returns:
            List of experience dicts.
        """
        if len(self._buffer) == 0:
            return []

        batch_size = min(batch_size, len(self._buffer))

        if prioritized and len(self._priorities) == len(self._buffer):
            # Compute sampling probabilities
            priorities = np.array(self._priorities, dtype=float)
            probs = priorities ** self.priority_alpha
            probs /= probs.sum()

            indices = np.random.choice(len(self._buffer), size=batch_size, replace=False, p=probs)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)

        batch = [self._buffer[i] for i in indices]
        return batch

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for specific experiences (after TD error computation).

        Args:
            indices: Indices of experiences to update.
            priorities: New priority values (typically proportional to |TD error| + eps).
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = priority

    def clear(self) -> None:
        """Clear all experiences from memory."""
        self._buffer.clear()
        self._priorities.clear()
        logger.info("ExperienceBuffer: cleared all experiences")

    async def save(self) -> None:
        """Persist all experiences to SQLite database."""
        await self._ensure_db()
        if self._conn is None:
            return

        if _HAS_AIOSQLITE:
            await self._conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT NOT NULL,
                    action INTEGER NOT NULL,
                    reward REAL NOT NULL,
                    next_state TEXT NOT NULL,
                    done INTEGER NOT NULL,
                    priority REAL NOT NULL
                )
            """)
            await self._conn.execute("DELETE FROM experiences")
            for exp, priority in zip(self._buffer, self._priorities):
                await self._conn.execute(
                    """
                    INSERT INTO experiences (state, action, reward, next_state, done, priority)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        json.dumps(exp["state"]),
                        exp["action"],
                        exp["reward"],
                        json.dumps(exp["next_state"]),
                        1 if exp["done"] else 0,
                        priority,
                    ),
                )
            await self._conn.commit()
        else:
            cursor = self._conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT NOT NULL,
                    action INTEGER NOT NULL,
                    reward REAL NOT NULL,
                    next_state TEXT NOT NULL,
                    done INTEGER NOT NULL,
                    priority REAL NOT NULL
                )
            """)
            cursor.execute("DELETE FROM experiences")
            for exp, priority in zip(self._buffer, self._priorities):
                cursor.execute(
                    """
                    INSERT INTO experiences (state, action, reward, next_state, done, priority)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        json.dumps(exp["state"]),
                        exp["action"],
                        exp["reward"],
                        json.dumps(exp["next_state"]),
                        1 if exp["done"] else 0,
                        priority,
                    ),
                )
            self._conn.commit()

        logger.info(f"ExperienceBuffer: saved {len(self._buffer)} experiences to {self.db_path}")

    async def load(self) -> None:
        """Load experiences from SQLite database."""
        await self._load_from_db()

    def __len__(self) -> int:
        """Return number of experiences in buffer."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ExperienceBuffer(size={len(self)}/{self.max_size}, db={self.db_path})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_db(self) -> None:
        """Ensure SQLite connection is established."""
        if self._conn is not None:
            return

        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            if _HAS_AIOSQLITE:
                self._conn = await aiosqlite.connect(str(self.db_path))
            else:
                warnings.warn(
                    "aiosqlite is not installed; falling back to synchronous sqlite3. "
                    "This will block the event loop. Install aiosqlite for async I/O.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                import sqlite3
                self._conn = sqlite3.connect(str(self.db_path))
            logger.debug(f"ExperienceBuffer: connected to SQLite at {self.db_path}")
        except Exception as exc:
            logger.error(f"ExperienceBuffer: failed to connect to SQLite: {exc}")
            self._conn = None

    async def _load_from_db(self) -> None:
        """Load experiences from SQLite database into memory buffer."""
        await self._ensure_db()
        if self._conn is None:
            return

        if _HAS_AIOSQLITE:
            async with self._conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='experiences'
            """) as cursor:
                if not await cursor.fetchone():
                    logger.debug("ExperienceBuffer: no experiences table found, starting fresh")
                    return

            async with self._conn.execute("""
                SELECT state, action, reward, next_state, done, priority
                FROM experiences
                ORDER BY id DESC
                LIMIT ?
            """, (self.max_size,)) as cursor:
                rows = await cursor.fetchall()
        else:
            cursor = self._conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='experiences'
            """)
            if not cursor.fetchone():
                logger.debug("ExperienceBuffer: no experiences table found, starting fresh")
                return
            cursor.execute("""
                SELECT state, action, reward, next_state, done, priority
                FROM experiences
                ORDER BY id DESC
                LIMIT ?
            """, (self.max_size,))
            rows = cursor.fetchall()

        self._buffer.clear()
        self._priorities.clear()

        for row in reversed(rows):  # Reverse to maintain chronological order
            state_json, action, reward, next_state_json, done, priority = row
            experience = {
                "state": json.loads(state_json),
                "action": action,
                "reward": reward,
                "next_state": json.loads(next_state_json),
                "done": bool(done),
            }
            self._buffer.append(experience)
            self._priorities.append(priority)

        logger.info(f"ExperienceBuffer: loaded {len(self._buffer)} experiences from {self.db_path}")

    def __del__(self) -> None:
        """Close SQLite connection on cleanup."""
        if self._conn is not None:
            if _HAS_AIOSQLITE:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self._conn.close())
                    else:
                        loop.run_until_complete(self._conn.close())
                except Exception:
                    pass
            else:
                try:
                    self._conn.close()
                except Exception:
                    pass
