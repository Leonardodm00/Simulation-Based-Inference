"""Smoke test for Stage 3b (plan v0.6).

Run:  python3 smoke_test_stage3b.py

The decomposition is tested against SYNTHETIC run records with a known answer,
not against a trained model: eq. (9) is an algebraic identity, so the right
test is one where the three terms are known exactly in advance. The probes are
tested against constructed embedding clouds whose off-support structure is
built in.

Pure ASCII, LF only.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "stage1"), os.path.join(_HERE, "..", "stage2"),
           os.path.join(_HERE, "..", "stage3"), _HERE):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from domain_objective import (REQUIRED_ARMS, decompose,  # noqa: E402
                              decomposition_table, joint_side,
                              per_axis_contraction_split, primary_score,
                              sigma_seed)
from encoder_probes import (embedding_direction_probe,  # noqa: E402
                            flag_degenerate_layers, layer_activation_stats)

RESULTS = []


def report(name, status, detail):
    RESULTS.append(status)
    print("[%s] %-48s %s" % (status, name, detail))


def ok(name, cond, detail):
    report(name, "PASS" if cond else "FAIL", detail)


def make_rec(arm, seed, nll, group, split_hash="H", contraction=None):
    return {"arm": arm, "seed": seed, "split_hash": split_hash,
            "L": float(np.mean(nll)),
            "L_pseudo_real": float(np.mean(nll)),
            "_nll": np.asarray(nll, dtype=np.float64),
            "_group": np.asarray(group),
            "_endpoint": "pseudo-real",
            "contraction": (contraction if contraction is not None
                            else [0.0, 0.0, 0.0])}


# ---------------------------------------------------------------------------
# B1 -- the decomposition is exact on a constructed case
# ---------------------------------------------------------------------------

def test_b1():
    rng = np.random.default_rng(0)
    n = 200
    grp = np.repeat(np.arange(20), 10)
    base = rng.standard_normal(n)
    # Known truth: domain effect 0.30, objective effect -0.10, total 0.20.
    a1 = base
    a0s = base - 0.10
    a0 = a0s + 0.30
    by = {"A1": {0: make_rec("A1", 0, a1, grp)},
          "A0s": {0: make_rec("A0s", 0, a0s, grp)},
          "A0": {0: make_rec("A0", 0, a0, grp)}}

    r = decompose(by, 0)
    ok("B1a D_domain recovers the injected 0.30",
       abs(r["D_domain"] - 0.30) < 1e-12, "%.12f" % r["D_domain"])
    ok("B1b D_objective recovers the injected -0.10",
       abs(r["D_objective"] + 0.10) < 1e-12, "%.12f" % r["D_objective"])
    ok("B1c D_total recovers 0.20",
       abs(r["D_total"] - 0.20) < 1e-12, "%.12f" % r["D_total"])
    ok("B1d the identity residual is zero",
       abs(r["identity_residual"]) < 1e-12,
       "residual = %.2e (eq. 9 is algebraic; anything else means the arms "
       "were scored on different rows)" % r["identity_residual"])


# ---------------------------------------------------------------------------
# B2 -- the guards that make the identity meaningful
# ---------------------------------------------------------------------------

def test_b2():
    rng = np.random.default_rng(1)
    grp = np.repeat(np.arange(10), 5)
    v = rng.standard_normal(50)
    by = {"A1": {0: make_rec("A1", 0, v, grp, "H1")},
          "A0s": {0: make_rec("A0s", 0, v, grp, "H1")},
          "A0": {0: make_rec("A0", 0, v, grp, "H2")}}
    r = decompose(by, 0)
    ok("B2a a differing split_hash is refused",
       "error" in r and "split_hash" in r["error"], r.get("error", "")[:60])

    by2 = {"A1": {0: make_rec("A1", 0, v, grp)},
           "A0s": {0: make_rec("A0s", 0, v, grp)},
           "A0": {0: make_rec("A0", 0, v[:40], grp[:40])}}
    r2 = decompose(by2, 0)
    ok("B2b differing row counts are refused",
       "error" in r2 and "row counts" in r2["error"], r2.get("error", "")[:60])

    by3 = {"A1": {0: make_rec("A1", 0, v, grp)},
           "A0": {0: make_rec("A0", 0, v, grp)}}
    r3 = decompose(by3, 0)
    ok("B2c a missing A0s is refused rather than guessed",
       "error" in r3 and "A0s" in r3["error"],
       "without A0s the total cannot be split at all")

    mixed = make_rec("A0s", 0, v, grp)
    mixed["_endpoint"] = "simulated"
    by4 = {"A1": {0: make_rec("A1", 0, v, grp)},
           "A0s": {0: mixed},
           "A0": {0: make_rec("A0", 0, v, grp)}}
    r4 = decompose(by4, 0)
    ok("B2d mixing endpoints across arms is refused",
       "error" in r4 and "endpoint" in r4["error"],
       "P6 says the two rankings need not agree, so they cannot be mixed")


# ---------------------------------------------------------------------------
# B3 -- sigma_seed, the joint-side split, the per-axis split
# ---------------------------------------------------------------------------

def test_b3():
    grp = np.arange(20)
    runs = [make_rec("A1", s, np.full(20, 1.0 + 0.1 * s), grp)
            for s in range(3)]
    ok("B3a sigma_seed is the across-seed SD of the primary score",
       abs(sigma_seed(runs) - float(np.std([1.0, 1.1, 1.2], ddof=1))) < 1e-12,
       "%.6f" % sigma_seed(runs))
    ok("B3b and is NaN with one seed",
       np.isnan(sigma_seed(runs[:1])), "two clauses need two seeds")

    by = {"A2": {0: make_rec("A2", 0, np.full(20, 2.0), grp)},
          "A2s": {0: make_rec("A2s", 0, np.full(20, 1.5), grp)}}
    js = joint_side(by, 0)
    ok("B3c the joint-side split L(A2) - L(A2s) is computed",
       abs(js["D_joint_domain"] - 0.5) < 1e-12, "%.6f" % js["D_joint_domain"])

    by2 = {a: {0: make_rec(a, 0, np.zeros(20), grp,
                           contraction=[c, c, c])}
           for a, c in (("A1", 0.1), ("A0s", 0.3), ("A0", 0.6))}
    pax = per_axis_contraction_split(by2, 0)
    ok("B3d the per-axis split is on contraction and says so",
       "contraction" in pax["quantity"]
       and abs(pax["domain"][0] - 0.3) < 1e-12
       and abs(pax["objective"][0] - 0.2) < 1e-12,
       "domain 0.3, objective 0.2, total %.1f" % pax["total"][0])


# ---------------------------------------------------------------------------
# B4 -- probe 1: hooks must not be corrupted by in-place activations
# ---------------------------------------------------------------------------

class InPlaceNet(nn.Module):
    """A conv followed by an INPLACE ReLU -- the DSN backbone's arrangement.

    Without cloning in the hook, the conv's recorded output is the ReLU's, and
    the probe attributes the activation's dead fraction to the convolution.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 4, 3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.act(self.conv(x))
        return torch.nn.functional.normalize(self.fc(h.mean(-1)), dim=-1)


def test_b4():
    torch.manual_seed(0)
    net = InPlaceNet()
    xs = torch.randn(64, 32)
    xr = torch.randn(64, 32)
    stats = layer_activation_stats(net, xs, xr)

    with torch.no_grad():
        true_conv = net.conv(xs.unsqueeze(1))
    true_dead = float((true_conv == 0).float().mean())
    seen = stats["conv"]["sim"]["dead_frac"]
    ok("B4a the conv's recorded output is not the in-place ReLU's",
       abs(seen - true_dead) < 1e-9,
       "recorded %.4f vs true %.4f; the ReLU's own dead fraction is %.4f"
       % (seen, true_dead, stats["act"]["sim"]["dead_frac"]))
    ok("B4b and the activation itself does show dead units",
       stats["act"]["sim"]["dead_frac"] > 0.2,
       "act dead frac = %.3f" % stats["act"]["sim"]["dead_frac"])

    # A layer genuinely dead on sim but alive on real must be flagged.
    class Dead(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(32, 4)
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.lin(x) - 10.0 * float(x.mean() > 0.4))

    d = Dead()
    xs2 = torch.rand(64, 32) * 0 + 0.9        # pushes the shift on
    xr2 = torch.rand(64, 32) * 0 + 0.1
    st2 = layer_activation_stats(d, xs2, xr2)
    flags = flag_degenerate_layers(st2)
    ok("B4c a layer dead on sim but alive on real is flagged",
       any(f["dead"] for f in flags),
       "flagged: %s" % [f["layer"] for f in flags])


# ---------------------------------------------------------------------------
# B5 -- probe 2: constructed off-support and on-support clouds
# ---------------------------------------------------------------------------

def test_b5():
    rng = np.random.default_rng(3)
    E = 6
    v = np.zeros(E)
    v[0] = 1.0
    real = rng.standard_normal((300, E)) * 0.05 + np.outer(
        rng.standard_normal(300), v)

    shifted = real + 6.0 * v                    # piled beyond the real range
    p = embedding_direction_probe(shifted, real)
    d0 = p["directions"][0]
    ok("B5a a shifted cloud is called off-support",
       "off-support" in d0["reading"]
       and d0["sim_frac_above_real_range"] > 0.5,
       "%.0f%% above the real range" % (100 * d0["sim_frac_above_real_range"]))

    narrow = real * 0.2
    p2 = embedding_direction_probe(narrow, real)
    ok("B5b a narrow cloud inside the range is called O2, not off-support",
       "O2" in p2["directions"][0]["reading"],
       p2["directions"][0]["reading"][:70])

    same = rng.standard_normal((300, E)) * 0.05 + np.outer(
        rng.standard_normal(300), v)
    p3 = embedding_direction_probe(same, real)
    ok("B5c a matching cloud shows no off-support signature",
       "no off-support signature" in p3["directions"][0]["reading"],
       "KS = %.3f" % p3["directions"][0]["ks"])

    ok("B5d r_eff is reported for both clouds",
       p3["r_eff_real"] > 1.0 and p3["r_eff_sim"] > 1.0,
       "real %.2f, sim %.2f" % (p3["r_eff_real"], p3["r_eff_sim"]))

    try:
        embedding_direction_probe(real[:, :3], real)
        caught = False
    except ValueError:
        caught = True
    ok("B5e mismatched embedding widths are refused", caught,
       "comparing two different encoders' clouds is meaningless")


# ---------------------------------------------------------------------------
# B6 -- the CLI dry run
# ---------------------------------------------------------------------------

def test_b6():
    tmp = tempfile.mkdtemp()
    try:
        import json
        grp = np.arange(20)
        for arm in ("A0", "A0s", "A1"):
            rec = {"arm": arm, "seed": 0, "split_hash": "H", "L": 1.0,
                   "L_pseudo_real": 1.0, "contraction": [0.1, 0.2],
                   "history": [{}]}
            with open(os.path.join(tmp, "%s_seed0.json" % arm), "w") as fh:
                json.dump(rec, fh)
            np.savez_compressed(os.path.join(tmp, "%s_seed0_perrow.npz" % arm),
                                nll=np.ones(20), group=grp,
                                nll_pseudo_real=np.ones(20),
                                group_pseudo_real=grp, theta=np.zeros((20, 2)))
        r = subprocess.run([sys.executable,
                            os.path.join(_HERE, "run_stage3b.py"),
                            "--runs-dir", tmp, "--dry-run"],
                           capture_output=True, text=True, cwd=_HERE)
        ok("B6a the dry run reports what is possible",
           r.returncode == 0 and "eq. (9) possible : yes" in r.stdout,
           r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[:80])

        out_md = os.path.join(tmp, "r.md")
        r2 = subprocess.run([sys.executable,
                             os.path.join(_HERE, "run_stage3b.py"),
                             "--runs-dir", tmp, "--out", out_md],
                            capture_output=True, text=True, cwd=_HERE)
        text = open(out_md).read() if os.path.isfile(out_md) else ""
        ok("B6b the report is written and states the missing interval",
           r2.returncode == 0 and "bootstrap_paired" in text
           and "no call" in text,
           "%d characters" % len(text))

        os.remove(os.path.join(tmp, "A0s_seed0.json"))
        r3 = subprocess.run([sys.executable,
                             os.path.join(_HERE, "run_stage3b.py"),
                             "--runs-dir", tmp],
                            capture_output=True, text=True, cwd=_HERE)
        ok("B6c without A0s it says the split is not computable",
           "Not computable" in r3.stdout,
           "and does not fall back to reporting the total as one effect")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("=" * 78)
    print("Smoke test: Stage 3b -- domain/objective decomposition and probes")
    print("=" * 78)
    test_b1()
    test_b2()
    test_b3()
    test_b4()
    test_b5()
    test_b6()
    print("-" * 78)
    n_fail = RESULTS.count("FAIL")
    print("%d passed, %d failed, %d skipped"
          % (RESULTS.count("PASS"), n_fail, RESULTS.count("SKIP")))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
