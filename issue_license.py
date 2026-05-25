#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vendor-only tool: generate RSA keys and signed license.lic files.

NEVER ship license_keys/private.pem to customers.

Examples:
  python issue_license.py init-keys
  python issue_license.py machine-id
  python issue_license.py issue --customer "Acme" --machine-id XXXX-... --trial-days 30
  python issue_license.py issue --customer "Acme" --machine-id XXXX-... --expiry-hours 1
  python issue_license.py issue --customer "Acme" --machine-id XXXX-... --expiry-minutes 1
  python issue_license.py verify license.lic
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

import license_manager as lm

_ROOT = Path(__file__).resolve().parent
_PRIVATE_DIR = _ROOT / "license_keys"
_PRIVATE_PEM = _PRIVATE_DIR / "private.pem"
_PUBLIC_PEM = _ROOT / lm._PUBLIC_KEY_BASENAME  # noqa: SLF001


def _load_private_key():
    if not _PRIVATE_PEM.is_file():
        print(
            f"Private key missing: {_PRIVATE_PEM}\n"
            "Run first: python issue_license.py init-keys",
            file=sys.stderr,
        )
        sys.exit(1)
    return serialization.load_pem_private_key(_PRIVATE_PEM.read_bytes(), password=None)


def cmd_init_keys(_args: argparse.Namespace) -> None:
    _PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    if _PRIVATE_PEM.is_file():
        print(f"Private key exists: {_PRIVATE_PEM}")
        key = _load_private_key()
    else:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        _PRIVATE_PEM.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        print(f"Created private key (keep secret): {_PRIVATE_PEM}")

    _PUBLIC_PEM.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    print(f"Wrote public key (ship with app): {_PUBLIC_PEM}")


def cmd_machine_id(_args: argparse.Namespace) -> None:
    print(lm.get_machine_id())


def _sign_payload(payload: dict) -> str:
    key = _load_private_key()
    sig = key.sign(
        lm._payload_bytes(payload),  # noqa: SLF001
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("ascii")


def cmd_issue(args: argparse.Namespace) -> None:
    machine_id = str(args.machine_id).strip().upper()
    if not machine_id:
        print("--machine-id is required", file=sys.stderr)
        sys.exit(1)

    issued = date.today()
    expiry_at: datetime | None = None
    if args.permanent:
        license_type = "permanent"
        expiry = lm._PERMANENT_EXPIRY  # noqa: SLF001
        expiry_value: str = expiry.isoformat()
    elif args.expiry_hours:
        license_type = "trial"
        expiry_at = datetime.now().replace(microsecond=0) + timedelta(
            hours=int(args.expiry_hours)
        )
        expiry = expiry_at.date()
        expiry_value = expiry_at.strftime("%Y-%m-%dT%H:%M:%S")
    elif args.expiry_minutes:
        license_type = "trial"
        expiry_at = datetime.now().replace(microsecond=0) + timedelta(
            minutes=int(args.expiry_minutes)
        )
        expiry = expiry_at.date()
        expiry_value = expiry_at.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        license_type = "trial" if args.trial_days else "standard"
        if args.expiry:
            expiry = date.fromisoformat(args.expiry)
        elif args.trial_days:
            expiry = issued + timedelta(days=int(args.trial_days))
        else:
            print(
                "Use --trial-days, --expiry-hours, --expiry-minutes, --expiry, or --permanent",
                file=sys.stderr,
            )
            sys.exit(1)
        expiry_value = expiry.isoformat()

    payload = {
        "version": 1,
        "customer": str(args.customer).strip() or "Customer",
        "machine_id": machine_id,
        "license_type": license_type,
        "issued_at": issued.isoformat(),
        "expiry": expiry_value,
    }
    payload["signature"] = _sign_payload(payload)

    out = Path(args.output).resolve() if args.output else (_ROOT / lm._LICENSE_BASENAME)  # noqa: SLF001
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Issued: {out}")
    print(f"  customer: {payload['customer']}")
    print(f"  machine_id: {machine_id}")
    print(f"  license_type: {license_type}")
    print(f"  expiry: {expiry_value}")
    if expiry_at is not None:
        print(f"  expires_at: {expiry_at.strftime('%Y-%m-%d %H:%M:%S')}")


def cmd_verify(args: argparse.Namespace) -> None:
    path = str(args.path)
    data = lm.load_license_file(path)
    sig_b64 = data.get("signature")
    if not sig_b64:
        print("No signature", file=sys.stderr)
        sys.exit(1)
    pub = lm._load_public_key()  # noqa: SLF001
    try:
        pub.verify(
            base64.b64decode(str(sig_b64).encode("ascii")),
            lm._payload_bytes(data),  # noqa: SLF001
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
    except InvalidSignature:
        print("Invalid signature", file=sys.stderr)
        sys.exit(1)
    display = {k: v for k, v in data.items() if k != "signature"}
    print("Signature OK")
    print(json.dumps(display, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="OCRInkjetCoder license issuer (vendor only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-keys", help="Generate RSA key pair")
    sub.add_parser("machine-id", help="Print this computer machine id")

    pi = sub.add_parser("issue", help="Issue signed license.lic")
    pi.add_argument("--customer", required=True, help="Customer name")
    pi.add_argument("--machine-id", required=True, help="Target machine id")
    pi.add_argument("--trial-days", type=int, default=0, help="Trial length in days")
    pi.add_argument(
        "--expiry-hours",
        type=int,
        default=0,
        help="Trial length in hours (exact expiry datetime, for testing)",
    )
    pi.add_argument(
        "--expiry-minutes",
        type=int,
        default=0,
        help="Trial length in minutes (exact expiry datetime, for testing)",
    )
    pi.add_argument("--expiry", help="Explicit expiry YYYY-MM-DD")
    pi.add_argument("--permanent", action="store_true", help="Permanent license")
    pi.add_argument("-o", "--output", help="Output path (default: ./license.lic)")

    pv = sub.add_parser("verify", help="Verify signature (vendor)")
    pv.add_argument("path", help="license.lic path")

    args = p.parse_args()
    handlers = {
        "init-keys": cmd_init_keys,
        "machine-id": cmd_machine_id,
        "issue": cmd_issue,
        "verify": cmd_verify,
    }
    handlers[args.cmd](args)


if __name__ == "__main__":
    main()
