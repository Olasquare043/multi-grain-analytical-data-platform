"""Filesystem helpers: checksums, sizes, atomic writes, directory hygiene.

Constraint 2.6 requires every run to log input filenames, row counts and
checksums. Everything that computes those lives here so the manifest and the
quality report agree by construction.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from config import settings
from src.utils.logging_setup import get_logger

LOG = get_logger(__name__)

_HASH_CHUNK = 1 << 20


def ensure_dirs(dirs: Iterable[Path] | None = None) -> None:
    """Create every directory the pipeline writes into."""
    for directory in (dirs if dirs is not None else settings.ALL_DIRS):
        Path(directory).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk: int = _HASH_CHUNK) -> str:
    """SHA-256 of a file, streamed so multi-hundred-megabyte inputs are safe."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def file_size(path: Path) -> int:
    """Size in bytes, or 0 when the path does not exist."""
    p = Path(path)
    return p.stat().st_size if p.exists() else 0


def dir_size(path: Path) -> int:
    """Recursive on-disk size in bytes of a directory tree."""
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def human_bytes(num: float) -> str:
    """Format a byte count for human-readable reports.

    >>> human_bytes(1536)
    '1.5 KB'
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0 or unit == "TB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024.0
    return f"{num:.1f} TB"


def write_json(path: Path, payload: Any, indent: int = 2) -> Path:
    """Write JSON atomically so a crashed run never leaves a truncated manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_text(path: Path, text: str) -> Path:
    """Write UTF-8 text atomically, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def reset_dir(path: Path) -> Path:
    """Delete and recreate a directory.

    Used before repartitioned writes so that reruns cannot leave orphaned
    Parquet fragments behind, which is what idempotence (constraint 2.7)
    actually requires at the storage layer.
    """
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def describe_input(path: Path, row_count: int | None = None) -> dict[str, Any]:
    """Provenance record for one input file, as embedded in the run manifest."""
    p = Path(path)
    return {
        "filename": p.name,
        "path": str(p.relative_to(settings.ROOT)) if p.is_absolute() and
        str(p).startswith(str(settings.ROOT)) else str(p),
        "bytes": file_size(p),
        "bytes_human": human_bytes(file_size(p)),
        "sha256": sha256_file(p) if p.exists() and p.is_file() else None,
        "row_count": row_count,
    }


def relative_to_root(path: Path) -> str:
    """Repo-relative path string for reporting, tolerant of outside paths."""
    p = Path(path)
    try:
        return str(p.relative_to(settings.ROOT)).replace(os.sep, "/")
    except ValueError:
        return str(p).replace(os.sep, "/")
