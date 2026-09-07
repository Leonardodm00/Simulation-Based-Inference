"""Smoke test for Stage 3: the runner, the arm dispatch, and the report.

Run:  python3 smoke_test_joint_arms.py

Needs a small two-arm bench bank, which it BUILDS ITSELF with the Stage 1
fixture provider, so the test has no external inputs. It does not need the DSN
repo except for the arms whose encoder is fitted by l_DSN (A0, A0s, A2, A2s,
A3), which are skipped with a reason when DSN_MAIN_DIR is unset.

Pure ASCII, LF only.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "stage1"), os.path.join(_HERE, "..", "stage2"),
           _HERE):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint_diagnostics import (effective_rank, information_gain,  # noqa: E402
                               p_eff_from_spectrum_result,
                               per_axis_contraction, spectrum_is_estimable)
import run_joint_arms as RJA  # noqa: E402
from run_joint_arms import ARMS, FixedStatsSummary, arm_config, grouped_split  # noqa: E402

RESULTS = []


def report(name, status, detail):
    RESULTS.append(status)
    print("[%s] %-46s %s" % (status, name, detail))


def ok(name, cond, detail):
    report(name, "PASS" if cond else "FAIL", detail)


# ---------------------------------------------------------------------------
# R1 -- arm dispatch is exhaustive and distinct
# ---------------------------------------------------------------------------

class _Args(object):
    lambda_dsn = 0.1
    lambda_rep = 0.05


def test_r1():
    cfgs = {a: arm_config(a, _Args()) for a in ARMS}
    ok("R1a every declared arm has a config", len(cfgs) == len(ARMS),
       "%d arms: %s" % (len(ARMS), ", ".join(ARMS)))

    # A1 is the nested baseline: no DSN, no replicate, nothing frozen.
    a1 = cfgs["A1"]
    ok("R1b A1 is the bare joint arm",
       a1["lambda_dsn"] == 0 and a1["lambda_rep"] == 0
       and not a1["freeze_encoder"],
       "A2 and A5 must nest it")

    ok("R1c A0 and A0s differ only in the DSN domain",
       {k: v for k, v in cfgs["A0"].items() if k != "dsn_domain"}
       == {k: v for k, v in cfgs["A0s"].items() if k != "dsn_domain"}
       and cfgs["A0"]["dsn_domain"] == "real"
       and cfgs["A0s"]["dsn_domain"] == "sim",
       "this is what makes eq. (9) a decomposition rather than a guess")

    ok("R1d shuffled is the only arm that permutes theta",
       [a for a in ARMS if cfgs[a]["shuffle_theta"]] == ["shuffled"],
       "delta_min comes from exactly one arm")

    ok("R1e A_ref is the only fixed-summary arm",
       [a for a in ARMS if cfgs[a]["fixed_summary"]] == ["A_ref"],
       "a trainable reference would beg the question")

    try:
        arm_config("A9", _Args())
        caught = False
    except ValueError:
        caught = True
    ok("R1f an unknown arm raises", caught, "no silent default")

    # [CHANGE Stage 4] The five encoder axes became flags so the joint tuner
    # can reach them. Their DEFAULTS must reproduce what make_backbone
    # hardcoded before, or every Stage 3 number already recorded silently
    # refers to a different architecture than a re-run would build.
    d = vars(RJA.build_parser().parse_args(
        ["--arm", "A1", "--sim-shards", "x", "--out-dir", "y"]))
    ok("R0e encoder flags default to the pre-Stage-4 hardcoded values",
       (d["depth_exponent"] == 3 and abs(d["width_multiplier"] - 2.0) < 1e-12
        and d["block_family"] == 0 and d["head_fusion"] == 0
        and abs(d["dropout"]) < 1e-12),
       "defaults moved: %r" % {k: d[k] for k in
                               ("depth_exponent", "width_multiplier",
                                "block_family", "head_fusion", "dropout")})
    ok("R0f make_backbone accepts an encoder override",
       "encoder" in RJA.make_backbone.__code__.co_varnames,
       "the tuner needs to pass the searched encoder axes")
    # [CHANGE Stage 4] The DSN-loss flags must default to DSNLossConfig()'s
    # defaults, or a flagless Stage 3 run builds a different loss than it did
    # before the flags existed. Read from the adapter, not from a literal.
    from dsn_loss_adapter import DSNLossConfig
    base = DSNLossConfig().to_dict()
    moved = {k: (d[k], base[k]) for k in
             ("loss_type", "mining_strategy", "margin", "angular_alpha_deg",
              "lambda_sep", "sep_warmup_frac") if d[k] != base[k]}
    if bool(d["strict_semihard"]) != bool(base["strict_semihard"]):
        moved["strict_semihard"] = (d["strict_semihard"],
                                    base["strict_semihard"])
    ok("R0g DSN-loss flags default to DSNLossConfig() defaults",
       not moved, "defaults moved: %r" % moved)
    ok("R0h beta1 default reproduces AdamW's",
       abs((1.0 - d["one_minus_beta1"]) - 0.9) < 1e-12,
       "one_minus_beta1 default %r gives beta1 %r"
       % (d["one_minus_beta1"], 1.0 - d["one_minus_beta1"]))


# ---------------------------------------------------------------------------
# R2 -- the grouped split never straddles a donor
# ---------------------------------------------------------------------------

def test_r2():
    donors = np.repeat(np.arange(20), 5)
    idx, digest = grouped_split(donors, seed=3)
    straddle = [d for d in np.unique(donors)
                if len(np.unique(idx[donors == d])) > 1]
    ok("R2a no donor straddles two splits", not straddle,
       "%d donors, %d rows, split sizes %s"
       % (20, len(donors), [int((idx == k).sum()) for k in (0, 1, 2)]))

    idx2, digest2 = grouped_split(donors, seed=3)
    ok("R2b the split is deterministic given the seed",
       np.array_equal(idx, idx2) and digest == digest2,
       "hash %s" % digest[:16])

    _, digest3 = grouped_split(donors, seed=4)
    ok("R2c and changes with the seed", digest3 != digest,
       "a different assignment must not carry the same hash")

    ok("R2d all three splits are non-empty",
       all((idx == k).sum() > 0 for k in (0, 1, 2)),
       "70/15/15 of 20 donors")


# ---------------------------------------------------------------------------
# R3 -- the diagnostics guard rails
# ---------------------------------------------------------------------------

def test_r3():
    ok("R3a the spectrum is refused when the denominator is rank-deficient",
       not spectrum_is_estimable(14, 6) and spectrum_is_estimable(40, 6),
       "14 rows at d=6 refused, 40 accepted (need >= 24)")

    p, note = p_eff_from_spectrum_result(np.array([1e300, 0.5]), n_rows=10, d=6)
    ok("R3b and returns None with a reason rather than a number", p is None,
       "%s" % note)

    p, note = p_eff_from_spectrum_result(np.array([1.4, 0.6, 0.0]),
                                         n_rows=100, d=3)
    ok("R3c eigenvalues above 1 are clipped and the clip is reported",
       abs(p - 1.6) < 1e-12 and "clipped" in note,
       "p_eff = %.3f, note: %s" % (p, note))

    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 1)) @ rng.standard_normal((1, 8))
    ok("R3d effective_rank is 1 for a rank-one cloud",
       abs(effective_rank(z) - 1.0) < 1e-6,
       "r_eff = %.6f (the collapse value C-1 at C=2)" % effective_rank(z))

    zi = rng.standard_normal((400, 5))
    ok("R3e and near d for an isotropic one",
       abs(effective_rank(zi) - 5.0) < 0.5,
       "r_eff = %.3f of 5" % effective_rank(zi))

    s = rng.standard_normal((50, 64, 3)) * 0.1
    kap = per_axis_contraction(s, np.ones(3))
    ok("R3f contraction is near 1 when the posterior is much narrower",
       np.all(kap > 0.9), "kappa = %s" % np.round(kap, 3).tolist())

    ok("R3g information_gain is L0 - L", abs(information_gain(
        np.array([1.0, 3.0]), 5.0) - 3.0) < 1e-12, "5 - 2 = 3")


# ---------------------------------------------------------------------------
# R4 -- the fixed summary
# ---------------------------------------------------------------------------

def test_r4():
    import torch
    f = FixedStatsSummary()
    x = torch.rand(7, 40)
    z = f(x)
    ok("R4a the fixed summary has the declared width",
       tuple(z.shape) == (7, FixedStatsSummary.N_STATS),
       "%s stats per window" % FixedStatsSummary.N_STATS)
    ok("R4b it has no trainable parameter",
       all(not p.requires_grad for p in f.parameters()),
       "one frozen anchor exists only because sbi calls next(net.parameters())")
    ok("R4c but parameters() is non-empty, or sbi raises StopIteration",
       len(list(f.parameters())) > 0, "this is why the anchor is there")


# ---------------------------------------------------------------------------
# R5 -- end to end: build a bench, run two arms, write the report
# ---------------------------------------------------------------------------

def test_r5():
    tmp = tempfile.mkdtemp()
    try:
        s1 = os.path.abspath(os.path.join(_HERE, "..", "stage1"))
        env = dict(os.environ)
        for arm, n, extra in (("S", 40, []), ("R", 16, ["--pi", "0.4",
                                                        "--gap-modes",
                                                        "range_shift,drift"])):
            cmd = [sys.executable, os.path.join(s1, "build_latent_bank.py"),
                   "--out-dir", os.path.join(tmp, arm), "--arm", arm,
                   "--n-traces", str(n), "--wells-per-donor", "2",
                   "--n-windows", "4", "--T-win", "1.28", "--fs", "50",
                   "--n-neurons", "20", "--n-classes", "3",
                   "--provider", "reference", "--seed", "1"] + extra
            r = subprocess.run(cmd, cwd=s1, env=env, capture_output=True,
                               text=True)
            if r.returncode:
                ok("R5 bench bank builds", False, r.stderr.strip()[-200:])
                return

        runs = os.path.join(tmp, "runs")
        for arm in ("A1", "shuffled"):
            cmd = [sys.executable, os.path.join(_HERE, "run_joint_arms.py"),
                   "--arm", arm, "--seed", "0",
                   "--sim-shards", os.path.join(tmp, "S", "*.npz"),
                   "--real-shards", os.path.join(tmp, "R", "*.npz"),
                   "--out-dir", runs, "--epochs", "2", "--steps-per-epoch", "3",
                   "--b-sim", "24", "--n-posterior-draws", "32"]
            r = subprocess.run(cmd, cwd=_HERE, env=env, capture_output=True,
                               text=True)
            if r.returncode:
                ok("R5 arm %s runs" % arm, False, r.stderr.strip()[-300:])
                return

        jsons = sorted(glob.glob(os.path.join(runs, "*_seed0.json")))
        ok("R5a both arms wrote a record", len(jsons) == 2,
           "%s" % [os.path.basename(p) for p in jsons])

        with open(jsons[0]) as fh:
            rec = json.load(fh)
        need = {"L", "L0", "delta_hat", "r_eff", "history", "split_hash",
                "contraction", "p_eff_spectrum", "spectrum_source"}
        ok("R5b the record carries everything the report needs",
           need.issubset(rec), "missing: %s" % sorted(need - set(rec)))

        pr = sorted(glob.glob(os.path.join(runs, "*_perrow.npz")))
        with np.load(pr[0]) as z:
            has = set(z.files)
        ok("R5c per-row NLL and group labels are persisted",
           {"nll", "group"}.issubset(has),
           "without these the paired bootstrap cannot be built after the fact")

        a, b = [json.load(open(p)) for p in jsons]
        ok("R5c2 the PRIMARY (pseudo-real) endpoint is scored and persisted",
           rec.get("L_pseudo_real") is not None
           and "nll_pseudo_real" in has,
           "endpoint = %s, L_primary = %s"
           % (rec.get("endpoint_primary"), rec.get("L_pseudo_real")))

        ok("R5d the two arms share one split hash",
           a["split_hash"] == b["split_hash"],
           "pairing at the row level requires it")

        out_md = os.path.join(tmp, "report.md")
        r = subprocess.run([sys.executable,
                            os.path.join(_HERE, "report_joint_arms.py"),
                            "--runs-dir", runs, "--out", out_md],
                           cwd=_HERE, env=env, capture_output=True, text=True)
        text = open(out_md).read() if os.path.isfile(out_md) else ""
        ok("R5e the report is written", r.returncode == 0 and text,
           "%d characters" % len(text))
        ok("R5f and states plainly that the interval is missing",
           "Incomplete" in text and "bootstrap_paired" in text,
           "bootstrap_paired is not on feat/misspec-gate as of 2026-09-02")
        ok("R5g and reports delta_min from the shuffled arm",
           "delta_min" in text, "G1 has no meaning without it")
        ok("R5h and labels the bank-level floor PROVISIONAL and "
           "recipe-matched to A1",
           "PROVISIONAL" in text and "recipe-matched" in text,
           "HANDOFF_DELTA_MIN_PER_CONFIG_v1 Tier-1: the single floor is a "
           "screen; the authoritative delta_min is per-config at the "
           "finalist tier")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("=" * 74)
    print("Smoke test: Stage 3 runner, diagnostics and report")
    print("=" * 74)
    test_r1()
    test_r2()
    test_r3()
    test_r4()
    test_r5()
    print("-" * 74)
    n_fail = RESULTS.count("FAIL")
    print("%d passed, %d failed, %d skipped"
          % (RESULTS.count("PASS"), n_fail, RESULTS.count("SKIP")))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
