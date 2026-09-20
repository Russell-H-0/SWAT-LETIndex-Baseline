#!/usr/bin/env python
"""Verify the frozen common-substrate snapshot of SWAT-LETIndex-Baseline.

Checks that every file recorded in ``provenance/common-substrate.sha256`` is present in
this repository with exactly the recorded SHA-256, and that the manifest covers the whole
frozen ``codes/src/enhanced_letindex/`` package.

This is a *local* check: it never contacts or reads EnhancedLETIndex, and it does not
care whether the source repository still exists at the recorded path.  It answers one
question only: "has this repository's frozen snapshot been modified?"

Usage (from the repository root, or anywhere):

    python provenance/verify_manifest.py [--root <repo root>]

Exit status: 0 when the snapshot is intact, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MANIFEST_RELATIVE = Path("provenance") / "common-substrate.sha256"
FROZEN_PACKAGE = Path("codes") / "src" / "enhanced_letindex"


def repo_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[1]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise SystemExit(f"{path}:{number}: malformed manifest line: {line!r}")
        digest, relative = parts
        entries.append((digest, relative))
    if not entries:
        raise SystemExit(f"{path}: manifest is empty")
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="repository root (default: inferred)")
    parser.add_argument("--quiet", action="store_true", help="only print the verdict")
    args = parser.parse_args(argv)

    root = repo_root(args.root)
    manifest_path = root / MANIFEST_RELATIVE
    if not manifest_path.is_file():
        print(f"FAIL: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    entries = read_manifest(manifest_path)
    failures: list[str] = []
    verified = 0

    for digest, relative in entries:
        target = root / relative
        if not target.is_file():
            failures.append(f"missing file: {relative}")
            continue
        actual = sha256_of(target)
        if actual != digest:
            failures.append(
                f"hash mismatch: {relative}\n    manifest {digest}\n    actual   {actual}"
            )
            continue
        verified += 1

    # The manifest must cover the whole frozen package: an unlisted file appearing in
    # codes/src/enhanced_letindex/ is a provenance violation (it would mean something was
    # copied in outside the recorded snapshot).
    listed = {relative for _, relative in entries}
    package_dir = root / FROZEN_PACKAGE
    if package_dir.is_dir():
        present = sorted(
            f"{FROZEN_PACKAGE.as_posix()}/{item.name}"
            for item in package_dir.glob("*.py")
        )
        for relative in present:
            if relative not in listed:
                failures.append(f"unlisted file in frozen package: {relative}")
        for relative in sorted(listed):
            if relative.startswith(FROZEN_PACKAGE.as_posix() + "/") and relative not in present:
                failures.append(f"manifest lists a file that is not in the package: {relative}")

    if failures:
        print(f"FAIL: frozen common substrate is NOT intact ({len(failures)} problem(s)):")
        for problem in failures:
            print(f"  - {problem}")
        return 1

    if not args.quiet:
        print(f"manifest: {MANIFEST_RELATIVE.as_posix()}")
        print(f"entries : {len(entries)}")
        print(f"verified: {verified}")
    print("OK: frozen common substrate is intact (all SHA-256 entries match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
