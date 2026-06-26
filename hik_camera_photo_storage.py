# -*- coding: utf-8 -*-
"""HikCameraPhotos storage quota and scheduled cleanup."""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional


DEFAULT_PHOTO_STORAGE_LIMIT_GB = 50
DEFAULT_CLEANUP_HOUR = 12
DEFAULT_CLEANUP_MINUTE = 0


@dataclass
class PhotoStorageCleanupResult:
    root: str
    reason: str
    limit_bytes: int
    size_before_bytes: int
    size_after_bytes: int
    deleted_dirs: list[str] = field(default_factory=list)
    skipped: bool = False
    message: str = ""

    @property
    def deleted_count(self) -> int:
        return len(self.deleted_dirs)


def photo_storage_limit_bytes(limit_gb: float = DEFAULT_PHOTO_STORAGE_LIMIT_GB) -> int:
    return int(float(limit_gb) * 1024**3)


def directory_size_bytes(path: str) -> int:
    if not path or not os.path.isdir(path):
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                continue
    return total


def _child_dir_entries(root: str) -> list[tuple[str, float, int]]:
    entries: list[tuple[str, float, int]] = []
    try:
        names = os.listdir(root)
    except OSError:
        return entries
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        entries.append((path, mtime, directory_size_bytes(path)))
    entries.sort(key=lambda item: (item[1], item[0]))
    return entries


def cleanup_photo_storage(
    root: str,
    *,
    limit_bytes: Optional[int] = None,
    reason: str = "manual",
) -> PhotoStorageCleanupResult:
    """Delete oldest child folders until ``root`` is below ``limit_bytes``."""
    limit = int(limit_bytes or photo_storage_limit_bytes())
    root = os.path.abspath(os.path.expanduser(root))
    size_before = directory_size_bytes(root)

    if size_before <= limit:
        return PhotoStorageCleanupResult(
            root=root,
            reason=reason,
            limit_bytes=limit,
            size_before_bytes=size_before,
            size_after_bytes=size_before,
            skipped=True,
            message="under_limit",
        )

    deleted: list[str] = []
    size_after = size_before

    for path, _mtime, _folder_size in _child_dir_entries(root):
        if size_after <= limit:
            break
        if not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path)
            deleted.append(path)
            size_after = directory_size_bytes(root)
        except OSError as exc:
            return PhotoStorageCleanupResult(
                root=root,
                reason=reason,
                limit_bytes=limit,
                size_before_bytes=size_before,
                size_after_bytes=size_after,
                deleted_dirs=deleted,
                message=f"delete_failed:{path}:{exc}",
            )

    return PhotoStorageCleanupResult(
        root=root,
        reason=reason,
        limit_bytes=limit,
        size_before_bytes=size_before,
        size_after_bytes=size_after,
        deleted_dirs=deleted,
        message="done" if size_after <= limit else "still_over_limit",
    )


def milliseconds_until_daily_time(
    hour: int = DEFAULT_CLEANUP_HOUR,
    minute: int = DEFAULT_CLEANUP_MINUTE,
    *,
    now: Optional[datetime] = None,
) -> int:
    current = now or datetime.now()
    target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if current >= target:
        target += timedelta(days=1)
    delay_ms = int((target - current).total_seconds() * 1000)
    return max(1000, delay_ms)


def format_bytes(num: int) -> str:
    if num >= 1024**3:
        return f"{num / 1024**3:.2f} GiB"
    if num >= 1024**2:
        return f"{num / 1024**2:.1f} MiB"
    if num >= 1024:
        return f"{num / 1024:.1f} KiB"
    return f"{num} B"


def cleanup_result_to_log_line(result: PhotoStorageCleanupResult) -> str:
    return (
        f"[photo_storage] reason={result.reason} "
        f"before={format_bytes(result.size_before_bytes)} "
        f"after={format_bytes(result.size_after_bytes)} "
        f"limit={format_bytes(result.limit_bytes)} "
        f"deleted={result.deleted_count} "
        f"status={result.message}"
    )


def cleanup_result_to_dict(result: PhotoStorageCleanupResult) -> dict[str, Any]:
    data = asdict(result)
    data["size_before_human"] = format_bytes(result.size_before_bytes)
    data["size_after_human"] = format_bytes(result.size_after_bytes)
    data["limit_human"] = format_bytes(result.limit_bytes)
    return data
