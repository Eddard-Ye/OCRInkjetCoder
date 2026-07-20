# -*- coding: utf-8 -*-
"""Salted password store for hik_camera_ui login (PBKDF2-HMAC-SHA256)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from typing import Any, Optional

DEFAULT_PASSWORD = "admin123"
DEFAULT_AUTH_FILENAME = "hik_camera_ui_auth.json"
PBKDF2_ITERATIONS = 120_000
SALT_BYTES = 16
DK_LEN = 32


def default_auth_path() -> str:
    """Prefer user home so credentials are not committed with the repo."""
    return os.path.join(os.path.expanduser("~"), DEFAULT_AUTH_FILENAME)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def hash_password(password: str, *, salt: Optional[bytes] = None) -> dict[str, Any]:
    if salt is None:
        salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
        dklen=DK_LEN,
    )
    return {
        "algo": "pbkdf2_sha256",
        "iterations": int(PBKDF2_ITERATIONS),
        "salt": _b64(salt),
        "hash": _b64(digest),
    }


def verify_password(password: str, record: dict[str, Any]) -> bool:
    try:
        salt = _b64d(str(record["salt"]))
        expected = _b64d(str(record["hash"]))
        iterations = int(record.get("iterations", PBKDF2_ITERATIONS))
    except (KeyError, TypeError, ValueError, Exception):
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
        dklen=len(expected) or DK_LEN,
    )
    return hmac.compare_digest(digest, expected)


@dataclass
class AuthStore:
    path: str

    @classmethod
    def default(cls) -> AuthStore:
        return cls(path=default_auth_path())

    def load_record(self) -> Optional[dict[str, Any]]:
        if not os.path.isfile(self.path):
            return None
        try:
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "salt" in raw and "hash" in raw:
                return raw
        except Exception:
            return None
        return None

    def save_record(self, record: dict[str, Any]) -> None:
        parent = os.path.dirname(self.path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        tmp = self.path + ".tmp"
        payload = {
            "algo": str(record.get("algo", "pbkdf2_sha256")),
            "iterations": int(record.get("iterations", PBKDF2_ITERATIONS)),
            "salt": str(record["salt"]),
            "hash": str(record["hash"]),
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def ensure_initialized(self, default_password: str = DEFAULT_PASSWORD) -> bool:
        """
        If no credential file exists, create one with ``default_password``.

        Returns True if a new file was created.
        """
        if self.load_record() is not None:
            return False
        self.save_record(hash_password(default_password))
        return True

    def verify(self, password: str) -> bool:
        record = self.load_record()
        if record is None:
            self.ensure_initialized()
            record = self.load_record()
        if record is None:
            return False
        return verify_password(password, record)

    def change_password(self, old_password: str, new_password: str) -> tuple[bool, str]:
        if not new_password:
            return False, "new password empty"
        if not self.verify(old_password):
            return False, "old password incorrect"
        try:
            self.save_record(hash_password(new_password))
        except Exception as e:
            return False, f"save failed: {e}"
        return True, ""
