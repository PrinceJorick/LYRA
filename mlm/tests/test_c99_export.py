"""C99 export agreement (Stage 4). Needs the frozen models (mlm/models/, gitignored) and the export
(mlm/export/c99/); skipped when either is absent. Test vectors are SYNTHETIC: float32 values at,
just below and just above the forest's own split thresholds, so every decision boundary is used."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS, OUT = ROOT / "models", ROOT / "export" / "c99"
pytestmark = pytest.mark.skipif(not (OUT / "export_manifest.json").exists() or not (MODELS / "frozen_pipeline.json").exists()
                                or shutil.which("cc") is None, reason="no frozen models / C99 export / C compiler")
MAIN = """#include <stdio.h>
#include "{h}"
int main(void) {{
    float x[{n}];
    double p[2];
    int i;
    for (;;) {{
        for (i = 0; i < {n}; ++i) {{
            if (scanf("%f", &x[i]) != 1) {{
                return 0;
            }}
        }}
        if ({fn}(x, {n}, p) != 0) {{
            return 1;
        }}
        printf("%.17g %.17g\\n", p[0], p[1]);
    }}
}}
"""


def boundary_vectors(model, n_features: int, n: int = 1200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    thr = [[] for _ in range(n_features)]
    for est in model.estimators_:
        t = est.tree_
        for f, v in zip(t.feature, t.threshold):
            if f >= 0:
                thr[f].append(v)
    x = np.empty((n, n_features), np.float32)
    for f in range(n_features):
        base = np.array(thr[f]) if thr[f] else np.array([0.0])
        pick = base[rng.integers(len(base), size=n)]
        step = np.where(rng.random(n) < 1 / 3, 0.0, rng.choice([-1, 1], n) * np.maximum(np.abs(pick), 1e-3) * 1e-4)
        x[:, f] = (pick + step).astype(np.float32)
    return x


@pytest.mark.parametrize("track", ["multimodal", "simenv"])
def test_c99_forest_matches_sklearn(track, tmp_path):
    import joblib
    meta = json.loads((OUT / "export_manifest.json").read_text())["models"][track]
    model = joblib.load(MODELS / meta["model_file"])
    n, name = len(meta["feature_names"]), f"lyra_{track}_rf"
    (tmp_path / "main.c").write_text(MAIN.format(h=f"{name}.h", n=n, fn=f"{name}_predict"))
    exe = tmp_path / "rf"
    cc = subprocess.run(["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", f"-I{OUT}", "-o", str(exe),
                         str(tmp_path / "main.c"), str(OUT / f"{name}.c")], capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    x = boundary_vectors(model, n)
    run = subprocess.run([str(exe)], input="\n".join(" ".join(f"{v:.9g}" for v in row) for row in x),
                         capture_output=True, text=True)
    assert run.returncode == 0
    c = np.loadtxt(run.stdout.splitlines())
    p = model.predict_proba(x)
    assert c.shape == p.shape == (1200, 2)
    assert np.abs(c - p).max() < 1e-5
    assert (c.argmax(axis=1) == model.predict(x)).all()
    for thr in (0.65, 0.45):  # entry / exit thresholds of the event confirmation
        assert ((c[:, 1] >= thr) == (p[:, 1] >= thr)).all()
