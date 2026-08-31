import importlib
import os
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO_ROOT)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
PKG = importlib.import_module(os.path.basename(REPO_ROOT))
EFFECT_IO = importlib.import_module(f"{PKG.__name__}.effect_io")
REML_MODEL = importlib.import_module(f"{PKG.__name__}.reml_model")
VARIANT_IO = importlib.import_module(f"{PKG.__name__}.variant_io")
write_sparse_effect_outputs = EFFECT_IO.write_sparse_effect_outputs
EffectEstimates = REML_MODEL.EffectEstimates
VariantRecord = VARIANT_IO.VariantRecord


def test_sparse_effect_output_adds_sparse_and_background_effects(
    tmp_path: Path,
) -> None:
    effects = EffectEstimates(
        fixed_effects=jnp.empty((0,)),
        random_effect=jnp.asarray([0.1, 0.2]),
        random_effect_components=(jnp.asarray([0.1, 0.2]),),
        snp_effects=(jnp.asarray([0.01, 0.02, 0.03]),),
        pcg_rel_res=1e-4,
        pcg_iters=4,
        y_mean=0.0,
        y_scale=1.0,
    )
    records = [
        VariantRecord("1", f"v{index}", "0", str(index + 1), "A", "G")
        for index in range(3)
    ]
    prefix = str(tmp_path / "fit")
    paths = write_sparse_effect_outputs(
        out_prefix=prefix,
        background_effects=effects,
        nuisance_fixed_effects=[0.5],
        sample_ids=["a", "b"],
        variant_records=records,
        support_source_variant_indices=[1],
        sparse_effects=[0.4],
    )
    rows = Path(paths["sparse_effects"]).read_text(encoding="utf-8").splitlines()
    values = [row.split("\t") for row in rows[1:]]
    np.testing.assert_allclose(
        [float(row[-1]) for row in values],
        [0.01, 0.42, 0.03],
    )
    assert Path(paths["sparse_effect_metadata"]).is_file()
