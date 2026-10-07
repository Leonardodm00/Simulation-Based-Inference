"""Block 4 (e4_numbers.py): spec-derived tests.

Oracles (SPEC Block 4): B3 Clopper-Pearson intervals equal
scipy.stats.binomtest(...).proportion_ci(method="exact"), and the chance of no
negative probe in n is (1 - f)^n; B4 the eight features agree with numpy
(std ddof=1, linear quantile) on the shard, and with torch running the FREEZE
`FixedStatsSummary.forward`; B2 the reached weights follow from AdamW applied to
the stated gradient sequences (checked here against torch.optim.AdamW and
clip_grad_norm_); the temporary shard is created outside the repository and
removed; quotes of E4 equal the printed values; two runs identical; hpc/joint
must be identical to FREEZE (and SPEC sec. 3: nothing read from the working
tree without the identity check).
"""
import glob
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from conftest import REPO, TOOLS, doc_text, floats, git_show, line_with, load_functions, \
    run_script, script_output

sys.path.insert(0, TOOLS)
STAGE1 = os.path.join(REPO, "hpc", "joint", "stage1")
FEATURES = ("mean", "sd", "max", "q90", "burst_frac", "acf1", "skew", "rough")


@pytest.fixture(scope="module")
def out():
    return script_output("e4_numbers.py")


# ---------------------------------------------------------------- B3
def test_b3_clopper_pearson_against_scipy(out):
    blk = out.split("B3  what the rho_grad record")[1].split("B4  A_ref")[0]
    rows = [ln for ln in blk.splitlines() if ln.strip().startswith("n = ")]
    assert len(rows) == 17
    for ln in rows:
        n, k, lo, hi, width = floats(ln)
        ci = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95, method="exact")
        assert "%.4f" % ci.low == "%.4f" % lo and "%.4f" % ci.high == "%.4f" % hi, ln
        # beta-quantile form (independent of binomtest)
        bl = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
        bh = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
        assert abs(bl - ci.low) < 1e-9 and abs(bh - ci.high) < 1e-9


def test_b3_no_negative_probability(out):
    for f in (0.05, 0.10, 0.20, 0.30):
        ln = line_with(out, "f = %.2f: (1 - f)^10" % f)
        assert "(1 - f)^10 = %.4f; (1 - f)^50 = %.4f" % ((1 - f) ** 10, (1 - f) ** 50) in ln
    assert "= %.4f" % (1 - 0.05 ** 0.1) in line_with(out, "in 10 probes")
    assert "= %.4f" % (1 - 0.05 ** 0.02) in line_with(out, "in 50 probes")


# ---------------------------------------------------------------- B1 arithmetic
def test_b1_stream_laws(out):
    n_plan = 10 * 25
    for N_c, per, label in ((918, 16, "class 0: 918 rows"), (972, 16, "class 1: 972 rows"),
                            (72, 10, "class 0: 72 rows"), (108, 10, "class 1: 108 rows")):
        ln = line_with(out, label)
        p_rep = 1 - np.prod([1 - j / N_c for j in range(per)])     # birthday problem
        assert "P(repeat) %.4f" % p_rep in ln
        assert "draws/row %.3f (%.3f with probes)" % (n_plan * per / N_c, 260 * per / N_c) in ln
    assert "P(a given row is drawn) %.6f" % (1 - (1 - 1 / 512) ** 6400) in out
    assert "P(a given row is drawn) %.6f" % (1 - (1 - 1 / 16384) ** 6400) in out


# ---------------------------------------------------------------- B4
@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("bench_shard"))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "build_latent_bank.py", "--out-dir", tmp, "--provider", "bench"],
                   cwd=STAGE1, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    files = sorted(glob.glob(os.path.join(tmp, "*.npz")))
    assert len(files) == 1
    with np.load(files[0], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _numpy_features(x):
    x = np.asarray(x, np.float64)
    mu = x.mean(-1)
    sd = x.std(-1, ddof=1) + 1e-8
    xc = (x - mu[:, None]) / sd[:, None]
    return np.stack([mu, sd, x.max(-1), np.quantile(x, 0.9, axis=-1, method="linear"),
                     (x > (mu + sd)[:, None]).mean(-1), (xc[:, 1:] * xc[:, :-1]).mean(-1),
                     (xc ** 3).mean(-1), np.abs(np.diff(x, axis=-1)).mean(-1)], axis=1)


def _printed_table(out):
    blk = out.split("B4  A_ref's eight fixed features")[1]
    tab = {}
    for name in FEATURES:
        ln = [l for l in blk.splitlines() if l.startswith(name + " ")][0]
        tab[name] = floats(ln.replace(name, "", 1))
    return tab


def test_b4_features_numpy(out, shard):
    x = shard["x"]
    x = x.reshape(x.shape[0], -1) if x.ndim == 3 else x
    assert x.shape == (512, 3000)
    F = _numpy_features(x)
    tab = _printed_table(out)
    for j, name in enumerate(FEATURES):
        want = [F[:, j].mean(), F[:, j].std(ddof=1), F[:, j].min(), F[:, j].max()]
        assert ["%.5f" % v for v in want] == ["%.5f" % v for v in tab[name]], name


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_b4_features_torch_freeze_forward(out, shard, dtype):
    torch = pytest.importorskip("torch")
    ns = load_functions(git_show("hpc/joint/stage3/run_joint_arms.py"), ("FixedStatsSummary",),
                        {"torch": torch})
    net = ns["FixedStatsSummary"]()
    x = shard["x"]
    x = x.reshape(x.shape[0], -1) if x.ndim == 3 else x
    with torch.no_grad():
        F = net(torch.tensor(x, dtype=getattr(torch, dtype))).double().numpy()
    G = _numpy_features(x)
    if dtype == "float64":
        # same math, float64 rounding only -- except burst_frac, which the FREEZE code
        # casts with `.float()` (float32) whatever the input dtype: 1e-6 relative there
        assert_allclose(np.delete(F, 4, 1), np.delete(G, 4, 1), rtol=1e-10, atol=1e-12)
        assert_allclose(F[:, 4], G[:, 4], rtol=1e-6)
    else:
        # the runner computes in float32: report the largest relative deviation of the
        # column statistics the document quotes (tolerance: 4 printed significant digits)
        for j in range(8):
            assert F[:, j].std(ddof=1) == pytest.approx(G[:, j].std(ddof=1), rel=5e-4), FEATURES[j]
            assert F[:, j].mean() == pytest.approx(G[:, j].mean(), rel=5e-4, abs=1e-6), FEATURES[j]


def test_b4_ratios_quoted(out):
    tab = _printed_table(out)
    sds = {k: v[1] for k, v in tab.items()}
    assert "%.1f" % (max(sds.values()) / min(sds.values())) in ("199.7", "199.6", "199.8")
    e4 = doc_text("E4_JOINT_OBJECTIVE_AND_LOOP.md")
    for q in ("from 0.0052 (roughness) to\n1.031 (skewness), a factor of 199.7",
              "from 0.0128 to 3.757", "168 of its standard deviations"):
        assert q in e4, q


# ---------------------------------------------------------------- B2
def test_b2_toy_against_torch_adamw_and_clip():
    torch = pytest.importorskip("torch")
    import e4_numbers as e4
    a, b, c = e4.toy_gradients()
    for lam in (0.0, 1.0, 10.0):
        psi, om, coefs = e4.run_toy(a, b, c, lam, 5.0, 1e-3, 0.9, 0.999)
        tp = torch.nn.Parameter(torch.zeros(a.shape[1], dtype=torch.float64))
        to = torch.nn.Parameter(torch.zeros(c.shape[1], dtype=torch.float64))
        opt = torch.optim.AdamW([tp, to], lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=0.0, foreach=False)
        for t in range(a.shape[0]):
            tp.grad = torch.tensor(a[t] + lam * b[t])
            to.grad = torch.tensor(c[t].copy())
            torch.nn.utils.clip_grad_norm_([tp, to], 5.0, foreach=False)
            opt.step()
        # float64; 250 steps of O(1e-3) updates -> agreement to 1e-10 absolute
        assert_allclose(psi, tp.detach().numpy(), atol=1e-10)
        assert_allclose(om, to.detach().numpy(), atol=1e-10)


def test_b2_quotes(out):
    shifts = [floats(line_with(out, "   lambda %6s: clipped" % lam).split("||omega_free|| =")[1])[0]
              for lam in ("1", "3", "10", "100")]
    assert "%.1f" % (100 * min(shifts)) == "2.4" and "%.1f" % (100 * max(shifts)) == "2.6"
    assert "at lambda = 0.7342" in out and "||a|| / ||b|| = 0.3043" in out
    e4 = doc_text("E4_JOINT_OBJECTIVE_AND_LOOP.md")
    assert "a 2.4 % to\n  2.6 % shift" in e4 and "midpoint at 0.734" in e4 and "0.304" in e4


# ---------------------------------------------------------------- temp shard, freeze guard
def test_temp_shard_outside_repo_and_removed(clone, tmp_path):
    tmpd = tmp_path / "tmpdir"
    tmpd.mkdir()
    before = subprocess.run(["git", "-C", clone, "status", "--porcelain", "--ignored"],
                            stdout=subprocess.PIPE, text=True).stdout
    tools = os.path.join(clone, "hpc", "joint", "docs", "tools")
    res = run_script("e4_numbers.py", cwd=tools, extra_env={"TMPDIR": str(tmpd)})
    assert res.returncode == 0, res.stderr[-1000:]
    assert os.listdir(tmpd) == [] or all(n.startswith("spec_notorch") for n in os.listdir(tmpd))
    after = subprocess.run(["git", "-C", clone, "status", "--porcelain", "--ignored"],
                           stdout=subprocess.PIPE, text=True).stdout
    new = set(after.splitlines()) - set(before.splitlines())
    assert not [n for n in new if ".npz" in n or "shard" in n], new


def test_refuses_modified_stage2(clone):
    with open(os.path.join(clone, "hpc", "joint", "stage2", "joint_train.py"), "a") as fh:
        fh.write("\n# modified by the spec test\n")
    res = run_script("e4_numbers.py", cwd=os.path.join(clone, "hpc", "joint", "docs", "tools"))
    assert res.returncode != 0
    assert "differs from 834eb41" in (res.stdout + res.stderr)


def test_no_unchecked_working_tree_import(clone, tmp_path):
    """SPEC sec. 3: the bench shard is built by stage1 code, which imports
    hpc/dsn/latent_burst_generator.py from the working tree; e4's check covers
    hpc/joint/stage1..4 only. A modified copy must be refused, not executed."""
    flag = tmp_path / "executed"
    with open(os.path.join(clone, "hpc", "dsn", "latent_burst_generator.py"), "a") as fh:
        fh.write("\nopen(%r, 'w').close()\n" % str(flag))
    res = run_script("e4_numbers.py", cwd=os.path.join(clone, "hpc", "joint", "docs", "tools"))
    assert not (flag.exists() and res.returncode == 0), \
        "modified hpc/dsn/latent_burst_generator.py was executed and the run exited 0"


def test_two_runs_identical(out):
    assert run_script("e4_numbers.py").stdout == out
