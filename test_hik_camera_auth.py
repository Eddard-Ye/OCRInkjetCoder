# -*- coding: utf-8 -*-
"""Tests for hik_camera_auth salted password store."""

import os
import tempfile
import unittest

from hik_camera_auth import (
    DEFAULT_PASSWORD,
    AuthStore,
    compute_auth_restricted_button_states,
    hardware_trigger_toggle_blocked,
    hash_password,
    startup_should_switch_hardware_trigger,
    verify_password,
)


class HashPasswordTest(unittest.TestCase):
    def test_roundtrip(self) -> None:
        rec = hash_password("secret")
        self.assertTrue(verify_password("secret", rec))
        self.assertFalse(verify_password("wrong", rec))
        self.assertNotIn("secret", rec["salt"])
        self.assertNotIn("secret", rec["hash"])

    def test_different_salts(self) -> None:
        a = hash_password("same")
        b = hash_password("same")
        self.assertNotEqual(a["salt"], b["salt"])
        self.assertNotEqual(a["hash"], b["hash"])


class AuthStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmpdir.name, "auth.json")
        self.store = AuthStore(path=self.path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_ensure_initialized_default(self) -> None:
        created = self.store.ensure_initialized()
        self.assertTrue(created)
        self.assertTrue(os.path.isfile(self.path))
        self.assertTrue(self.store.verify(DEFAULT_PASSWORD))
        self.assertFalse(self.store.ensure_initialized())

    def test_change_password(self) -> None:
        self.store.ensure_initialized()
        ok, msg = self.store.change_password(DEFAULT_PASSWORD, "new-pass")
        self.assertTrue(ok, msg)
        self.assertFalse(self.store.verify(DEFAULT_PASSWORD))
        self.assertTrue(self.store.verify("new-pass"))

    def test_change_password_wrong_old(self) -> None:
        self.store.ensure_initialized()
        ok, msg = self.store.change_password("bad", "new-pass")
        self.assertFalse(ok)
        self.assertTrue(self.store.verify(DEFAULT_PASSWORD))

    def test_no_plaintext_on_disk(self) -> None:
        self.store.ensure_initialized(default_password="plain-secret-xyz")
        with open(self.path, encoding="utf-8") as f:
            text = f.read()
        self.assertNotIn("plain-secret-xyz", text)


class AuthUiHelpersTest(unittest.TestCase):
    def test_compute_states_matrix(self) -> None:
        self.assertEqual(
            compute_auth_restricted_button_states(
                logged_in=False, camera_connected=True
            ),
            ("disabled", "disabled"),
        )
        self.assertEqual(
            compute_auth_restricted_button_states(
                logged_in=True, camera_connected=True
            ),
            ("normal", "normal"),
        )

    def test_startup_switch_predicate(self) -> None:
        self.assertTrue(
            startup_should_switch_hardware_trigger(
                startup_hardware_trigger=True,
                use_hw_trigger=False,
                camera_connected=True,
            )
        )

    def test_toggle_auth_gate(self) -> None:
        self.assertFalse(
            hardware_trigger_toggle_blocked(require_auth=False, logged_in=False)
        )


if __name__ == "__main__":
    unittest.main()
