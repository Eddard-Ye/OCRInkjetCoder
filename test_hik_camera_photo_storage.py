# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

from hik_camera_photo_storage import (
    cleanup_photo_storage,
    directory_size_bytes,
    photo_storage_limit_bytes,
)


class PhotoStorageCleanupTest(unittest.TestCase):
    def test_cleanup_deletes_oldest_dirs_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = os.path.join(tmp, "2026-01-01")
            new = os.path.join(tmp, "2026-06-01")
            os.makedirs(old, exist_ok=True)
            os.makedirs(new, exist_ok=True)
            with open(os.path.join(old, "a.bin"), "wb") as f:
                f.write(b"x" * 2048)
            with open(os.path.join(new, "b.bin"), "wb") as f:
                f.write(b"y" * 1024)
            os.utime(old, (1, 1))
            os.utime(new, None)

            limit = directory_size_bytes(tmp) - 512
            result = cleanup_photo_storage(tmp, limit_bytes=limit, reason="test")

            self.assertFalse(result.skipped)
            self.assertEqual(result.deleted_dirs, [old])
            self.assertTrue(os.path.isdir(new))
            self.assertFalse(os.path.exists(old))
            self.assertLessEqual(directory_size_bytes(tmp), limit)

    def test_skip_when_under_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            limit = photo_storage_limit_bytes(50)
            result = cleanup_photo_storage(tmp, limit_bytes=limit, reason="test")
            self.assertTrue(result.skipped)
            self.assertEqual(result.deleted_dirs, [])


if __name__ == "__main__":
    unittest.main()
