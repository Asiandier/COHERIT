"""LD-score input and automatic computation for adaptive sparse COHERIT.

The adaptive partition is ordered by the per-variant score

    1 + sum_j r(i, j)^2,

where pairs are restricted to the same chromosome and a physical window.
Automatic calculation deliberately delegates the pairwise dosage correlation
kernel to PLINK2: this is the exact definition used by the COHERIT pilots and
is substantially faster and more memory efficient than materialising an LD
matrix in Python.  The pipeline owns the calculation, parses and validates the
result, caches only the compact score table, and removes PLINK's large pair
table after a successful conversion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

import numpy as np

from .variant_io import VariantRecord, iter_variant_records_for_prefix


def load_variant_records(prefix: str, genotype_format: str) -> list[VariantRecord]:
    records = list(iter_variant_records_for_prefix(prefix, genotype_format))
    if not records:
        raise ValueError("The genotype source contains no variants.")
    ids = [record.variant_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Adaptive LD partitioning requires unique BIM/PVAR variant IDs."
        )
    return records


def load_aligned_ld_scores(
    path: str | Path,
    variant_records: Sequence[VariantRecord],
) -> np.ndarray:
    """Load an ``ID, ld_score`` table and align it to source variant order."""
    score_path = Path(path)
    values: dict[str, float] = {}
    with score_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"ID", "ld_score"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"{score_path}: LD-score table requires ID and ld_score columns."
            )
        for row_number, row in enumerate(reader, start=2):
            marker_id = str(row.get("ID", "")).strip()
            if not marker_id:
                raise ValueError(f"{score_path}: empty ID at row {row_number}.")
            if marker_id in values:
                raise ValueError(
                    f"{score_path}: duplicate LD-score ID {marker_id!r}."
                )
            try:
                values[marker_id] = float(row["ld_score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{score_path}: invalid ld_score at row {row_number}."
                ) from exc

    source_ids = [record.variant_id for record in variant_records]
    missing = [marker_id for marker_id in source_ids if marker_id not in values]
    if missing:
        raise ValueError(
            f"{score_path}: missing LD score for source variant {missing[0]!r}."
        )
    source_set = set(source_ids)
    extra_count = sum(marker_id not in source_set for marker_id in values)
    if extra_count:
        raise ValueError(
            f"{score_path}: contains {extra_count} IDs absent from the genotype source."
        )
    scores = np.asarray([values[marker_id] for marker_id in source_ids], dtype=np.float64)
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
        raise ValueError("LD scores must be finite and nonnegative.")
    return scores


def parse_plink_vcor(
    vcor_path: str | Path,
    variant_records: Sequence[VariantRecord],
) -> np.ndarray:
    """Convert one PLINK2 tabular unphased-r2 report to per-SNP scores."""
    ids = [record.variant_id for record in variant_records]
    id_to_index = {marker_id: index for index, marker_id in enumerate(ids)}
    scores = np.ones(len(ids), dtype=np.float64)
    seen_pairs = 0
    with Path(vcor_path).open(encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        id_columns = [
            index for index, name in enumerate(header) if "ID" in name.upper()
        ]
        r2_columns = [
            index
            for index, name in enumerate(header)
            if name.upper() in {"R2", "UNPHASED_R2"}
            or "R2" in name.upper()
        ]
        if len(id_columns) < 2 or not r2_columns:
            raise ValueError(
                f"Could not identify two ID columns and R2 in {vcor_path}: {header}"
            )
        first_id, second_id = id_columns[:2]
        r2_column = r2_columns[0]
        required_column = max(first_id, second_id, r2_column)
        for row_number, line in enumerate(handle, start=2):
            fields = line.split()
            if len(fields) <= required_column:
                raise ValueError(
                    f"{vcor_path}: malformed PLINK2 LD row {row_number}."
                )
            marker_a = fields[first_id]
            marker_b = fields[second_id]
            try:
                r_squared = float(fields[r2_column])
            except ValueError as exc:
                raise ValueError(
                    f"{vcor_path}: invalid R2 at row {row_number}."
                ) from exc
            if not np.isfinite(r_squared) or not 0.0 <= r_squared <= 1.0 + 1e-8:
                raise ValueError(
                    f"{vcor_path}: R2 outside [0, 1] at row {row_number}."
                )
            if marker_a == marker_b:
                continue
            try:
                index_a = id_to_index[marker_a]
                index_b = id_to_index[marker_b]
            except KeyError as exc:
                raise ValueError(
                    f"{vcor_path}: LD pair references unknown variant {exc.args[0]!r}."
                ) from exc
            scores[index_a] += r_squared
            scores[index_b] += r_squared
            seen_pairs += 1
    if seen_pairs == 0:
        # A source with isolated variants is valid; the all-one score is then
        # correct.  Retain this case instead of manufacturing a split signal.
        scores.fill(1.0)
    return scores


def write_ld_score_table(
    path: str | Path,
    variant_records: Sequence[VariantRecord],
    scores: np.ndarray,
    *,
    window_kb: int,
) -> None:
    output = Path(path)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size != len(variant_records):
        raise ValueError("LD-score vector does not align with variant metadata.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["ID", "ld_score", "ld_window_kb"])
        for record, score in zip(variant_records, values, strict=True):
            writer.writerow([record.variant_id, format(float(score), ".17g"), int(window_kb)])
    os.replace(temporary, output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _calculation_identity(
    *,
    prefix: str,
    genotype_format: str,
    keep_path: str,
    window_kb: int,
) -> dict[str, object]:
    source_suffix = ".bed" if genotype_format == "bed" else ".pgen"
    variant_suffix = ".bim" if genotype_format == "bed" else ".pvar"
    sample_suffix = ".fam" if genotype_format == "bed" else ".psam"
    return {
        "genotype_format": genotype_format,
        "genotype": _file_identity(Path(prefix + source_suffix)),
        "variants": _file_identity(Path(prefix + variant_suffix)),
        "samples": _file_identity(Path(prefix + sample_suffix)),
        "keep_path": str(Path(keep_path).resolve()),
        "keep_sha256": _sha256(Path(keep_path)),
        "ld_window_kb": int(window_kb),
        "definition": "1_plus_sum_unphased_dosage_r_squared",
    }


def compute_ld_scores_with_plink2(
    *,
    prefix: str,
    genotype_format: str,
    keep_path: str,
    output_path: str | Path,
    window_kb: int = 1000,
    threads: int = 1,
    plink2: str = "plink2",
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute and cache LD scores on the training samples."""
    if genotype_format not in {"bed", "pgen"}:
        raise ValueError("genotype_format must be 'bed' or 'pgen'.")
    if int(window_kb) < 1 or int(threads) < 1:
        raise ValueError("LD window and thread count must be positive.")
    executable = shutil.which(plink2) if os.path.sep not in plink2 else plink2
    if not executable or not Path(executable).is_file():
        raise FileNotFoundError(
            "Automatic LD-score calculation requires PLINK2. Install plink2, "
            "put it on PATH, or supply --plink2 /path/to/plink2."
        )

    records = load_variant_records(prefix, genotype_format)
    output = Path(output_path)
    metadata_path = Path(str(output) + ".metadata.json")
    identity = _calculation_identity(
        prefix=prefix,
        genotype_format=genotype_format,
        keep_path=keep_path,
        window_kb=window_kb,
    )
    if output.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("calculation_identity") == identity:
                cached = load_aligned_ld_scores(output, records)
                return cached, {**metadata, "cache_reused": True}
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_prefix = output.parent / f".{output.stem}.plink2.{os.getpid()}"
    command = [
        str(executable),
        "--bfile" if genotype_format == "bed" else "--pfile",
        prefix,
        "--keep",
        keep_path,
        "--r2-unphased",
        "cols=id",
        "--ld-window-kb",
        str(int(window_kb)),
        "--ld-window-r2",
        "0",
        "--threads",
        str(int(threads)),
        "--out",
        str(temporary_prefix),
    ]
    log_path = Path(str(output) + ".plink2.log")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PLINK2 LD-score calculation failed ({completed.returncode}); see {log_path}."
        )
    candidates = sorted(output.parent.glob(temporary_prefix.name + "*.vcor"))
    if len(candidates) != 1:
        raise RuntimeError(
            "PLINK2 did not emit exactly one plain-text .vcor table. "
            f"See {log_path}."
        )
    pair_path = candidates[0]
    scores = parse_plink_vcor(pair_path, records)
    write_ld_score_table(output, records, scores, window_kb=window_kb)
    pair_path.unlink()
    for suffix in (".log", ".nosex"):
        generated = Path(str(temporary_prefix) + suffix)
        if generated.exists():
            generated.unlink()
    metadata = {
        "schema_version": 1,
        "source": "computed_by_plink2",
        "calculation_identity": identity,
        "plink2": str(Path(executable).resolve()),
        "command": command,
        "score_path": str(output.resolve()),
        "plink2_log": str(log_path.resolve()),
        "n_variants": len(records),
        "cache_reused": False,
    }
    temporary_metadata = metadata_path.with_name(
        f".{metadata_path.name}.tmp.{os.getpid()}"
    )
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)
    return scores, metadata


def write_ld_rank_artifact(
    path: str | Path,
    scores: np.ndarray,
    *,
    bins: int,
) -> dict[str, object]:
    """Persist the deterministic LD-rank order used by all adaptive layers."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size < 2 or int(bins) < 2:
        raise ValueError("LD-rank splitting requires at least two SNPs and bins.")
    effective_bins = min(int(bins), int(values.size))
    source_index = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((source_index, values))
    boundaries = np.asarray(
        [int((index * values.size) // effective_bins) for index in range(1, effective_bins)],
        dtype=np.int64,
    )
    if np.any(np.diff(np.concatenate([[0], boundaries, [values.size]])) <= 0):
        raise RuntimeError("Balanced LD-rank bins contain an empty interval.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        source_order=order,
        boundary_positions=boundaries,
        requested_bins=np.asarray(int(bins), dtype=np.int64),
        effective_bins=np.asarray(effective_bins, dtype=np.int64),
        marker_count=np.asarray(values.size, dtype=np.int64),
        score_min=np.asarray(float(np.min(values))),
        score_max=np.asarray(float(np.max(values))),
    )
    os.replace(temporary, output)
    return {
        "path": str(output.resolve()),
        "marker_count": int(values.size),
        "requested_bins": int(bins),
        "effective_bins": effective_bins,
    }


__all__ = [
    "compute_ld_scores_with_plink2",
    "load_aligned_ld_scores",
    "load_variant_records",
    "parse_plink_vcor",
    "write_ld_rank_artifact",
    "write_ld_score_table",
]
