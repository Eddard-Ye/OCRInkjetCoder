# -*- coding: utf-8 -*-
"""RSA-signed license verification (Scheme A): machine-bound trial / permanent."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Optional

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
except ImportError:  # pragma: no cover
    hashes = None  # type: ignore
    serialization = None  # type: ignore
    padding = None  # type: ignore
    rsa = None  # type: ignore
    InvalidSignature = Exception  # type: ignore

_LICENSE_VERSION = 1
_LICENSE_BASENAME = "license.lic"
_PUBLIC_KEY_BASENAME = "license_public.pem"
_VENDOR_STATE_DIR = "OCRInkjetCoder"
_SKIP_ENV = "OCR_INKJET_SKIP_LICENSE"
_PERMANENT_EXPIRY = date(2099, 12, 31)

_MSG_NO_CRYPTO = (
    "Missing cryptography package. Run: pip install cryptography"
)
_MSG_NO_PUBLIC_KEY = (
    "Public key license_public.pem not found. Contact vendor."
)
_MSG_BAD_FORMAT = "Invalid license file format"
_MSG_BAD_DATE = "Invalid date in license field {0!r}: {1!r}"
_MSG_NO_SIG = "License file has no signature"
_MSG_BAD_SIG_FMT = "Invalid signature encoding"
_MSG_SIG_FAIL = "Signature verification failed (file may be tampered)"
_MSG_VERIFY_FAIL = "License verification failed: {0}"
_MSG_BAD_VERSION = "Unsupported license file version"
_MSG_MACHINE = (
    "License does not match this computer.\n"
    "Local machine id: {local}\n"
    "Licensed machine id: {lic}\n"
    "Send your machine id to the vendor for a new license."
)
_MSG_UNKNOWN_TYPE = "Unknown license_type: {0!r}"
_MSG_EXPIRED = (
    "License expired on {expiry}.\n"
    "Customer: {customer}\n"
    "Machine id: {mid}\n"
    "Contact vendor for renewal (license.lic)."
)
_MSG_CLOCK = (
    "System clock rollback detected. License check rejected.\n"
    "Set the correct date/time and retry."
)
_MSG_NOT_FOUND = (
    "license.lic not found.\n\n"
    "Machine id: {mid}\n\n"
    "Send the machine id to the vendor, then place license.lic in:\n"
    "{paths}"
)
_MSG_UNNAMED = "\u672a\u547d\u540d\u5ba2\u6237"


class LicenseError(Exception):
    """License missing, invalid, expired, or machine mismatch."""


@dataclass(frozen=True)
class LicenseInfo:
    customer: str
    machine_id: str
    license_type: str
    expiry: date
    expiry_at: datetime
    issued_at: date
    path: str

    @property
    def is_permanent(self) -> bool:
        return self.license_type == "permanent" or self.expiry >= _PERMANENT_EXPIRY

    @property
    def days_remaining(self) -> int:
        return (self.expiry - date.today()).days

    @property
    def seconds_remaining(self) -> float:
        if self.is_permanent:
            return float("inf")
        return max(0.0, (self.expiry_at - datetime.now()).total_seconds())

    @property
    def hours_remaining(self) -> int:
        if self.is_permanent:
            return 0
        return max(0, int(self.seconds_remaining // 3600))

    @property
    def is_expired(self) -> bool:
        if self.is_permanent:
            return False
        return datetime.now() > self.expiry_at


def project_root() -> Path:
    return Path(__file__).resolve().parent


def _program_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
    else:
        base = "/var/lib"
    return Path(base) / _VENDOR_STATE_DIR


def _state_file_path() -> Path:
    d = _program_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "license_state.json"


def get_machine_id() -> str:
    parts: list[str] = []
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
            )
            mg, _ = winreg.QueryValueEx(key, "MachineGuid")
            parts.append(str(mg))
            winreg.CloseKey(key)
        except OSError:
            pass
    parts.append(platform.node())
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest().upper()
    return "-".join(digest[i : i + 4] for i in range(0, 16, 4))


def _public_key_path() -> Path:
    return project_root() / _PUBLIC_KEY_BASENAME


def license_search_paths() -> list[Path]:
    roots: list[Path] = [project_root()]
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(sys.executable).resolve().parent)
    roots.append(_program_data_dir())
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        p = (root / _LICENSE_BASENAME).resolve()
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def find_license_path() -> Optional[str]:
    for p in license_search_paths():
        if p.is_file():
            return str(p)
    return None


def load_license_file(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise LicenseError(_MSG_BAD_FORMAT)
    return raw


def _parse_iso_date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError as e:
        raise LicenseError(_MSG_BAD_DATE.format(field, value)) from e


def _parse_expiry_value(value: Any, field: str) -> tuple[date, datetime]:
    """
    ``expiry`` in license JSON:

    - ``YYYY-MM-DD`` ?? valid through end of that day (23:59:59)
    - ``YYYY-MM-DDTHH:MM:SS`` ?? exact expiry moment (hour-level trials)
    """
    raw = str(value).strip()
    if len(raw) > 10 and ("T" in raw or " " in raw[10:]):
        try:
            dt = datetime.fromisoformat(raw.replace(" ", "T")[:19])
            return dt.date(), dt
        except ValueError as e:
            raise LicenseError(_MSG_BAD_DATE.format(field, value)) from e
    d = _parse_iso_date(raw, field)
    return d, datetime.combine(d, time(23, 59, 59))


def _payload_bytes(data: dict[str, Any]) -> bytes:
    ordered = {k: data[k] for k in sorted(data) if k != "signature"}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _load_public_key():
    if serialization is None:
        raise LicenseError(_MSG_NO_CRYPTO)
    pem_path = _public_key_path()
    if not pem_path.is_file():
        raise LicenseError(_MSG_NO_PUBLIC_KEY)
    return serialization.load_pem_public_key(pem_path.read_bytes())


def verify_license_dict(data: dict[str, Any], *, path: str = "") -> LicenseInfo:
    if os.environ.get(_SKIP_ENV, "").strip().lower() in ("1", "true", "yes"):
        today = date.today()
        return LicenseInfo(
            customer="DEV-SKIP",
            machine_id=get_machine_id(),
            license_type="permanent",
            expiry=_PERMANENT_EXPIRY,
            expiry_at=datetime.combine(_PERMANENT_EXPIRY, time(23, 59, 59)),
            issued_at=today,
            path=path or "(skip)",
        )

    sig_b64 = data.get("signature")
    if not sig_b64 or not isinstance(sig_b64, str):
        raise LicenseError(_MSG_NO_SIG)

    try:
        signature = base64.b64decode(sig_b64.encode("ascii"), validate=True)
    except Exception as e:
        raise LicenseError(_MSG_BAD_SIG_FMT) from e

    public_key = _load_public_key()
    try:
        public_key.verify(
            signature,
            _payload_bytes(data),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
    except InvalidSignature as e:
        raise LicenseError(_MSG_SIG_FAIL) from e
    except Exception as e:
        raise LicenseError(_MSG_VERIFY_FAIL.format(e)) from e

    if int(data.get("version", 0)) != _LICENSE_VERSION:
        raise LicenseError(_MSG_BAD_VERSION)

    lic_machine = str(data.get("machine_id", "")).strip().upper()
    local_machine = get_machine_id().upper()
    if lic_machine != local_machine:
        raise LicenseError(
            _MSG_MACHINE.format(local=local_machine, lic=lic_machine)
        )

    license_type = str(data.get("license_type", "trial")).strip().lower()
    if license_type not in ("trial", "permanent", "standard"):
        raise LicenseError(_MSG_UNKNOWN_TYPE.format(license_type))

    expiry, expiry_at = _parse_expiry_value(data.get("expiry"), "expiry")
    issued_at = _parse_iso_date(data.get("issued_at"), "issued_at")
    customer = str(data.get("customer", "")).strip() or _MSG_UNNAMED

    if license_type != "permanent" and datetime.now() > expiry_at:
        exp_show = expiry_at.strftime("%Y-%m-%d %H:%M:%S")
        raise LicenseError(
            _MSG_EXPIRED.format(
                expiry=exp_show,
                customer=customer,
                mid=local_machine,
            )
        )

    _check_clock_rollback()
    _touch_last_run()

    return LicenseInfo(
        customer=customer,
        machine_id=local_machine,
        license_type=license_type,
        expiry=expiry,
        expiry_at=expiry_at,
        issued_at=issued_at,
        path=path,
    )


def _check_clock_rollback() -> None:
    today = date.today()
    last = _read_last_run(_state_file_path())
    if last is not None and today < last - timedelta(days=1):
        raise LicenseError(_MSG_CLOCK)


def _read_last_run(state_path: Path) -> Optional[date]:
    if not state_path.is_file():
        return None
    try:
        with open(state_path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and raw.get("last_run"):
            return date.fromisoformat(str(raw["last_run"])[:10])
    except Exception:
        return None
    return None


def _touch_last_run() -> None:
    try:
        with open(_state_file_path(), "w", encoding="utf-8") as f:
            json.dump({"last_run": date.today().isoformat()}, f, indent=2)
    except OSError:
        pass


def check_license() -> LicenseInfo:
    path = find_license_path()
    if not path:
        mid = get_machine_id()
        searched = "\n".join(f"  - {p}" for p in license_search_paths())
        raise LicenseError(_MSG_NOT_FOUND.format(mid=mid, paths=searched))
    return verify_license_dict(load_license_file(path), path=path)


def format_runtime_remaining(info: LicenseInfo) -> str:
    """Short label for UI: remaining licensed runtime (hours / minutes)."""
    if info.is_permanent:
        return "\u53ef\u8fd0\u884c\u65f6\u95f4\uff1a\u6c38\u4e45"
    if info.is_expired:
        return "\u53ef\u8fd0\u884c\u65f6\u95f4\uff1a\u5df2\u8fc7\u671f"
    until = info.expiry_at.strftime("%Y-%m-%d %H:%M")
    secs = info.seconds_remaining
    h = int(secs // 3600)
    if h >= 1:
        return f"\u53ef\u8fd0\u884c\u65f6\u95f4\uff1a\u5269\u4f59 {h} \u5c0f\u65f6 (\u81f3 {until})"
    mins = max(0, int(secs // 60))
    return f"\u53ef\u8fd0\u884c\u65f6\u95f4\uff1a\u5269\u4f59 {mins} \u5206\u9499 (\u81f3 {until})"


def format_license_status(info: LicenseInfo) -> str:
    if info.is_permanent:
        term = "\u6c38\u4e45"
    else:
        d = info.days_remaining
        term = f"\u5269\u4f59 {d} \u5929" if d >= 0 else "\u5df2\u8fc7\u671f"
    label = "\u6388\u6743"
    exp = "\u5230\u671f"
    return f"{label}: {info.customer} | {term} | {exp} {info.expiry.isoformat()}"
