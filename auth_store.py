from __future__ import annotations

import json
import os
import threading
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class UserRecord:
    username: str
    password_hash: str


class AuthStore:
    """Very small local authentication store.

    - Stores users in a JSON file.
    - Passwords are hashed with sha256 (demo only; not production-grade).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, str]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # expected: {"username": "hash"}
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        return {}

    def _save(self, users: Dict[str, str]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
        os.replace(tmp, self.path)

    @staticmethod
    def hash_password(password: str) -> str:
        # Demo hashing only
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def create_user(self, username: str, password: str) -> bool:
        username = (username or "").strip()
        if not username or not password:
            return False

        with self._lock:
            users = self._load()
            if username in users:
                return False
            users[username] = self.hash_password(password)
            self._save(users)
            return True

    def verify_user(self, username: str, password: str) -> bool:
        username = (username or "").strip()
        if not username or not password:
            return False
        with self._lock:
            users = self._load()
            expected = users.get(username)
            if not expected:
                return False
            return expected == self.hash_password(password)

    def get_user(self, username: str) -> Optional[UserRecord]:
        username = (username or "").strip()
        if not username:
            return None
        with self._lock:
            users = self._load()
            if username not in users:
                return None
            return UserRecord(username=username, password_hash=users[username])

