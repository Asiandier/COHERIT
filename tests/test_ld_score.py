import importlib
import os
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
PKG = importlib.import_module(os.path.basename(REPO_ROOT))
LD_SCORE = importlib.import_module(f"{PKG.__name__}.ld_score")
VARIANT_IO = importlib.import_module(f"{PKG.__name__}.variant_io")
load_aligned_ld_scores = LD_SCORE.load_aligned_ld_scores
parse_plink_vcor = LD_SCORE.parse_plink_vcor
write_ld_rank_artifact = LD_SCORE.write_ld_rank_artifact
VariantRecord = VARIANT_IO.VariantRecord


def _records() -> list[VariantRecord]:
    return [
        VariantRecord("1", "v2", "0", "20", "A", "G"),
        VariantRecord("1", "v1", "0", "10", "C", "T"),
        VariantRecord("1", "v3", "0", "30", "G", "A"),
    ]


def test_load_ld_scores_aligns_by_id_not_table_order(tmp_path: Path) -> None:
    table = tmp_path / "score.tsv"
    table.write_text(
        "ID\tld_score\n"
        "v1\t1.5\n"
        "v3\t3.5\n"
        "v2\t2.5\n",
        encoding="utf-8",
    )
    observed = load_aligned_ld_scores(table, _records())
    np.testing.assert_allclose(observed, [2.5, 1.5, 3.5])


def test_load_ld_scores_rejects_extra_marker(tmp_path: Path) -> None:
    table = tmp_path / "score.tsv"
    table.write_text(
        "ID\tld_score\n"
        "v1\t1\n"
        "v2\t1\n"
        "v3\t1\n"
        "other\t1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absent from the genotype"):
        load_aligned_ld_scores(table, _records())


def test_parse_plink_vcor_accumulates_each_pair_for_both_snps(
    tmp_path: Path,
) -> None:
    vcor = tmp_path / "pairs.vcor"
    vcor.write_text(
        "ID_A\tID_B\tUNPHASED_R2\n"
        "v1\tv2\t0.25\n"
        "v1\tv3\t0.50\n",
        encoding="utf-8",
    )
    observed = parse_plink_vcor(vcor, _records())
    # Source order is v2, v1, v3; every score includes self-LD = 1.
    np.testing.assert_allclose(observed, [1.25, 1.75, 1.50])


def test_write_ld_rank_artifact_is_deterministic_and_balanced(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rank.npz"
    metadata = write_ld_rank_artifact(
        path,
        np.asarray([3.0, 1.0, 1.0, 2.0, 4.0]),
        bins=4,
    )
    with np.load(path, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["source_order"], [1, 2, 3, 0, 4])
        np.testing.assert_array_equal(payload["boundary_positions"], [1, 2, 3])
    assert metadata["effective_bins"] == 4
