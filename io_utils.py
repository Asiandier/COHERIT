from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


class IncompleteOutputError(ValueError):
    """A stage cannot be resumed because its output artifacts are incomplete."""


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: str | Path, value: object) -> None:
    """Publish a complete JSON file or leave the previous file untouched."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def output_manifest(paths: Iterable[str | Path]) -> dict[str, int]:
    """Record closed output files without reading large effect/state arrays."""
    manifest = {}
    for name in paths:
        path = Path(name).resolve()
        if not path.is_file() or path.stat().st_size <= 0:
            raise IncompleteOutputError(f"Missing or empty output: {path}")
        manifest[str(path)] = path.stat().st_size
    return manifest


def validate_output_manifest(
    manifest: Mapping[str, int], *, required_paths: Iterable[str | Path],
) -> None:
    """Reject missing/truncated files and incomplete completion manifests."""
    if not isinstance(manifest, dict) or not manifest:
        raise IncompleteOutputError("Missing output completion manifest.")
    required = {str(Path(path).resolve()) for path in required_paths}
    if not required.issubset(manifest):
        raise IncompleteOutputError("Completion manifest omits required outputs.")
    for name, size in manifest.items():
        path = Path(name)
        if (
            not isinstance(size, int) or size <= 0 or not path.is_file()
            or path.stat().st_size != size
        ):
            raise IncompleteOutputError(f"Missing or size-mismatched output: {path}")


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_joined_rows(
    path: str,
    header: str,
    rows: Iterable[str],
    *,
    chunk_size: int = 8192,
) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        chunk: list[str] = []
        for row in rows:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                f.write("".join(chunk))
                chunk.clear()
        if chunk:
            f.write("".join(chunk))


__all__ = [
    "ensure_parent_dir", "write_joined_rows", "read_json", "atomic_json",
    "IncompleteOutputError", "output_manifest", "validate_output_manifest",
]
