#!/usr/bin/env python3
"""
smoke_test_misspec.py -- does the gate fire when the simulator is wrong,
and stay quiet when it is right?

Run:
    python smoke_test_misspec.py
    python smoke_test_misspec.py -k G3
    python smoke_test_misspec.py --fast     # skip the many-seed rate tests

Embeddings here are synthetic points on S^{E-1}, built so the correct
verdict is known by construction. No DSN, no network, no data.

  G0  Plumbing. MMD is ~0 for two samples from one distribution, positive
      for a shifted one, symmetric in its arguments; the bandwidth grid is
      ordered and finite.
  G1  Quiet when the real cloud is drawn from the simulated distribution.
  G2  Fires on a shifted real cloud, with G1's setting as the negative
      control under the identical criterion.
  G3  THE TEST THAT JUSTIFIES THE MODULE. Real data that is CLUSTERED --
      R recordings, W windows each, windows of one recording correlated --
      but drawn from the SAME distribution as the simulations. The correct
      verdict is PASS. The i.i.d. null rejects anyway; the group-aware null
      does not.
  G4  False-positive rate of the group-aware verdict over many seeds.
  G5  Per-recording scores localise a minority of corrupted recordings and
      leave the rest alone.
  G6  Per-class: one class shifted, the other not. The per-class verdict
      catches it even where pooled is diluted.
  G7  minimum_detectable_shift is monotone in delta and returns a finite
      answer at a shift the gate visibly detects.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npe_misspec import (  # noqa: E402
    bandwidth_grid, minimum_detectable_shift, misspecification_gate,
    mmd2_multiscale,
)

RESULTS: List[Tuple[str, str, str]] = []
E_DIM = 16          # matches the exporter's E; nothing here depends on it


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run(test_id: str, fn: Callable[[], str], selector: str = "") -> None:
    if selector and selector not in test_id:
        return
    try:
        detail = fn() or ""
        RESULTS.append((test_id, "PASS", detail))
        print("  PASS  %-26s %s" % (test_id, detail), flush=True)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-1]
        RESULTS.append((test_id, "FAIL", "%s | %s" % (exc, tb)))
        print("  FAIL  %-26s %s" % (test_id, exc), flush=True)


# ---------------------------------------------------------------------------
# Synthetic embedding clouds with a known answer
# ---------------------------------------------------------------------------

SPREAD = 0.125          # marginal spread of one window, in both arms


def sphere(n: int, rng, centre: np.ndarray,
           spread: float = SPREAD) -> np.ndarray:
    """n points on S^{E-1}, loosely spread around `centre`.

    Not a von Mises-Fisher sampler: an isotropic Gaussian of standard
    deviation `spread` around the centre, projected back to the sphere.
    Enough for a diagnostic test and dependency-free, but `spread` is not a
    kappa and these clouds should not be checked against any analytic vMF
    result.
    """
    x = centre[None, :] + spread * rng.normal(size=(n, centre.size))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def base_centre() -> np.ndarray:
    c = np.zeros(E_DIM)
    c[0] = 1.0
    return c


def clustered_real(n_groups: int, n_windows: int, rng,
                   centre: np.ndarray, spread: float = SPREAD,
                   icc: float = 0.7):
    """Real-like data: recordings differ, windows within a recording agree.

    THE MARGINAL LAW OF ONE WINDOW MUST MATCH sphere(..., spread) EXACTLY,
    or G3 is not testing what it claims. A recording centre drawn with
    spread s_b and windows drawn around it with spread s_w give a marginal
    spread of sqrt(s_b^2 + s_w^2), so the two are set by a variance
    decomposition:

        s_b = spread * sqrt(icc),    s_w = spread * sqrt(1 - icc)

    with `icc` the intraclass correlation, i.e. the fraction of variance
    that sits BETWEEN recordings. Only the dependence structure then
    differs from the simulated arm; the one-window marginal is identical,
    so the correct verdict is PASS and any rejection is a false positive.

    An earlier version of this helper set the two spreads independently.
    The real cloud was then genuinely wider than the simulated one, the
    gate correctly rejected, and the test read as a failure of the module.
    """
    s_b = spread * np.sqrt(icc)
    s_w = spread * np.sqrt(1.0 - icc)
    Z, groups = [], []
    for r in range(n_groups):
        c = centre + s_b * rng.normal(size=centre.size)
        Z.append(sphere(n_windows, rng, c / np.linalg.norm(c), spread=s_w))
        groups += ["rec_%02d" % r] * n_windows
    return np.concatenate(Z, axis=0), np.asarray(groups)


def sim_cloud(n: int, rng, centre=None, spread: float = SPREAD):
    return sphere(n, rng, base_centre() if centre is None else centre, spread)


# ---------------------------------------------------------------------------
# G0 -- plumbing
# ---------------------------------------------------------------------------

def g0_plumbing() -> str:
    rng = np.random.default_rng(0)
    A = sim_cloud(300, rng)
    B = sim_cloud(300, rng)
    c2 = base_centre().copy()
    c2[1] = 0.6
    c2 /= np.linalg.norm(c2)
    C = sim_cloud(300, rng, centre=c2)

    bw = bandwidth_grid(A, B, rng=rng)
    check(bw.size == 5 and np.all(np.isfinite(bw)) and np.all(bw > 0),
          "bandwidth grid is not finite and positive: %s" % bw)
    check(np.all(np.diff(bw) > 0), "bandwidth grid is not ascending")

    m_same = mmd2_multiscale(A, B, bw)
    m_diff = mmd2_multiscale(A, C, bw)
    check(m_diff > m_same,
          "MMD did not increase for a shifted cloud (%.5f vs %.5f)"
          % (m_diff, m_same))
    check(abs(m_same) < 0.05,
          "MMD between two samples of one distribution is %.5f, not near 0"
          % m_same)
    check(abs(mmd2_multiscale(A, C, bw) - mmd2_multiscale(C, A, bw)) < 1e-12,
          "MMD is not symmetric in its arguments")
    check(abs(mmd2_multiscale(A, A, bw)) < 1e-12,
          "MMD of a sample with itself is not zero")
    return ("MMD same=%.5f shifted=%.5f; grid %.3f..%.3f"
            % (m_same, m_diff, bw[0], bw[-1]))


# ---------------------------------------------------------------------------
# G1, G2 -- does it fire when it should, and only then
# ---------------------------------------------------------------------------

def _gate_iid(shift: float, seed: int, n_groups: int = 36,
              n_windows: int = 6, n_sim: int = 4000, n_null: int = 200):
    """Real rows drawn INDEPENDENTLY, one per pseudo-recording group."""
    rng = np.random.default_rng(seed)
    z_sim = sim_cloud(n_sim, rng)
    c = base_centre().copy()
    if shift:
        c[1] += shift
        c /= np.linalg.norm(c)
    z_real = sphere(n_groups * n_windows, rng, c)
    groups = np.repeat(["rec_%02d" % r for r in range(n_groups)], n_windows)
    return misspecification_gate(z_sim, z_real, groups, n_null=n_null,
                                 n_window_choices=10, seed=seed)["pooled"]


def g1_quiet_when_well_specified() -> str:
    r = _gate_iid(0.0, seed=0)
    check(r.passes,
          "gate rejected a well-specified model: p_group=%.4f" % r.p_group)
    return ("p_group=%.3f  MMD=%.5f  (%d recordings)"
            % (r.p_group, r.mmd_group, r.n_groups))


def g2_fires_when_misspecified() -> str:
    bad = _gate_iid(0.5, seed=0)
    check(not bad.passes,
          "gate did NOT reject a clearly shifted real cloud: p_group=%.4f"
          % bad.p_group)
    check(bad.reject_fraction >= 0.9,
          "only %.0f%% of window choices rejected a gross shift"
          % (100 * bad.reject_fraction))
    ok = _gate_iid(0.0, seed=0)
    check(ok.passes,
          "NEGATIVE CONTROL under the identical criterion failed: the gate "
          "also rejected the well-specified model (p=%.4f)" % ok.p_group)
    return ("shifted: p=%.4f (%.0f%% of choices); well-specified: p=%.3f"
            % (bad.p_group, 100 * bad.reject_fraction, ok.p_group))


# ---------------------------------------------------------------------------
# G3 -- the test that justifies the module
# ---------------------------------------------------------------------------

def _g3_one_seed(seed: int, n_groups: int = 36, n_windows: int = 6,
                 n_sim: int = 4000, n_null: int = 200):
    rng = np.random.default_rng(seed)
    z_sim = sim_cloud(n_sim, rng)
    z_real, groups = clustered_real(n_groups, n_windows, rng, base_centre())
    r = misspecification_gate(z_sim, z_real, groups, n_null=n_null,
                              n_window_choices=10, seed=seed)["pooled"]
    return bool(r.passes), bool(r.p_iid < r.alpha), r


def g3_clustered_null() -> str:
    """Clustered real data from the RIGHT model. Correct verdict: PASS.

    The i.i.d. null compares 216 dependent rows against subsets of 216
    independent simulated rows. The observed cloud is tighter than any such
    subset, so the null sits in the wrong place and the gate rejects a model
    that is correct. The group-aware verdict uses one window per recording
    and is comparing like with like.

    Reported as a rate over seeds: a single-seed assertion on either side
    would be flaky, and the claim is about calibration, not about one draw.
    """
    n_seeds = 6
    grp_ok = iid_rej = 0
    for s in range(n_seeds):
        a, b, _ = _g3_one_seed(s)
        grp_ok += int(a)
        iid_rej += int(b)
    check(grp_ok >= 5,
          "the GROUP-aware gate rejected a well-specified clustered cohort "
          "on %d/%d seeds -- the correction does not work"
          % (n_seeds - grp_ok, n_seeds))
    check(iid_rej >= 4,
          "the i.i.d. null did NOT misbehave on clustered data (%d/%d seeds "
          "rejected); if it is fine, this module is unnecessary and that is "
          "the finding to report" % (iid_rej, n_seeds))
    return ("over %d seeds: group-aware passes %d, i.i.d. null falsely "
            "rejects %d" % (n_seeds, grp_ok, iid_rej))


# ---------------------------------------------------------------------------
# G4 -- rate
# ---------------------------------------------------------------------------

def make_g4(n_seeds: int) -> Callable[[], str]:
    def g4_false_positive_rate() -> str:
        rej = 0
        for s in range(n_seeds):
            ok, _, _ = _g3_one_seed(s, n_sim=2500, n_null=150)
            rej += int(not ok)
        rate = rej / float(n_seeds)
        check(rate <= 0.25,
              "group-aware false-positive rate %.2f over %d seeds on "
              "well-specified clustered data" % (rate, n_seeds))
        return "false-positive rate %.2f over %d seeds" % (rate, n_seeds)
    return g4_false_positive_rate


# ---------------------------------------------------------------------------
# G5, G6 -- localisation and classes
# ---------------------------------------------------------------------------

def g5_per_recording_localisation() -> str:
    seed = 0
    rng = np.random.default_rng(seed)
    n_groups, n_windows, n_bad = 30, 6, 6
    z_sim = sim_cloud(4000, rng)
    z_real, groups = clustered_real(n_groups, n_windows, rng, base_centre())

    bad_centre = base_centre().copy()
    bad_centre[1] += 0.8
    bad_centre /= np.linalg.norm(bad_centre)
    bad_names = ["rec_%02d" % r for r in range(n_bad)]
    for name in bad_names:
        m = groups == name
        z_real[m] = sphere(int(m.sum()), rng, bad_centre,
                           spread=SPREAD * np.sqrt(0.3))

    r = misspecification_gate(z_sim, z_real, groups, n_null=300,
                              n_window_choices=10, seed=seed)["pooled"]
    fwer, fdr = r.family()
    flagged = set(np.asarray(r.per_group_names)[fdr.rejected].tolist())
    hit = len(flagged & set(bad_names))
    false_hit = len(flagged - set(bad_names))
    check(hit >= n_bad - 1,
          "per-recording scores found only %d of %d corrupted recordings"
          % (hit, n_bad))
    check(false_hit <= 2,
          "per-recording scores flagged %d healthy recordings" % false_hit)
    return ("%d/%d corrupted recordings flagged, %d false (BH); Holm flags %d"
            % (hit, n_bad, false_hit, fwer.n_rejected))


def g6_per_class() -> str:
    seed = 1
    rng = np.random.default_rng(seed)
    n_groups, n_windows = 18, 6
    z_sim = sim_cloud(4000, rng)

    ctrl_z, ctrl_g = clustered_real(n_groups, n_windows, rng, base_centre())
    shifted = base_centre().copy()
    shifted[1] += 0.55
    shifted /= np.linalg.norm(shifted)
    path_z, path_g = clustered_real(n_groups, n_windows, rng, shifted)
    path_g = np.asarray(["p_" + g for g in path_g])

    z_real = np.concatenate([ctrl_z, path_z], axis=0)
    groups = np.concatenate([ctrl_g, path_g])
    classes = np.asarray(["control"] * ctrl_z.shape[0]
                         + ["patho"] * path_z.shape[0])

    res = misspecification_gate(z_sim, z_real, groups, classes=classes,
                                n_null=300, n_window_choices=10, seed=seed)
    check(set(res.keys()) == {"pooled", "control", "patho"},
          "expected pooled + one entry per class, got %s" % sorted(res))
    check(not res["patho"].passes,
          "the shifted class was not caught: p=%.4f" % res["patho"].p_group)
    check(res["control"].passes,
          "the unshifted class was falsely flagged: p=%.4f"
          % res["control"].p_group)

    only = misspecification_gate(z_sim, z_real, groups, classes=classes,
                                 n_null=200, n_window_choices=10, seed=seed,
                                 only_class="control")["pooled"]
    check(only.passes, "only_class='control' disagreed with the per-class "
                       "result computed alongside pooled")
    return ("control p=%.3f (pass), patho p=%.4f (fail), pooled p=%.4f"
            % (res["control"].p_group, res["patho"].p_group,
               res["pooled"].p_group))


# ---------------------------------------------------------------------------
# G7 -- power
# ---------------------------------------------------------------------------

def g7_minimum_detectable_shift() -> str:
    rng = np.random.default_rng(0)
    z_sim = sim_cloud(3000, rng)
    mde, rates = minimum_detectable_shift(
        z_sim, n_groups=36, deltas=(0.02, 0.05, 0.1, 0.2, 0.4),
        n_null=200, n_rep=25, seed=0)
    check(np.all(np.diff(rates) >= -0.15),
          "rejection rate is not broadly increasing in the shift: %s"
          % np.round(rates, 2).tolist())
    check(rates[-1] >= 0.8,
          "a shift of 0.4 was detected only %.0f%% of the time; the gate has "
          "essentially no power and a PASS would mean nothing"
          % (100 * rates[-1]))
    check(mde is not None, "no delta in the grid reached the target power")
    return ("MDE=%.3f at 80%% power with 36 recordings; rates %s"
            % (mde, np.round(rates, 2).tolist()))


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", dest="selector", default="")
    ap.add_argument("--fast", action="store_true", help="skip G4")
    ap.add_argument("--seeds", type=int, default=12)
    args = ap.parse_args()

    print("=" * 74)
    print("Misspecification gate validated on synthetic spherical embeddings")
    print("=" * 74)

    print("\n[plumbing]")
    run("G0_plumbing", g0_plumbing, args.selector)

    print("\n[does it fire when it should, and only then]")
    run("G1_quiet_well_specified", g1_quiet_when_well_specified, args.selector)
    run("G2_fires_misspecified", g2_fires_when_misspecified, args.selector)
    run("G3_clustered_null", g3_clustered_null, args.selector)

    if not args.fast:
        print("\n[rate over %d seeds]" % args.seeds)
        run("G4_false_positive_rate", make_g4(args.seeds), args.selector)
    else:
        print("\n[rate test skipped (--fast)]")

    print("\n[localisation, classes, power]")
    run("G5_per_recording", g5_per_recording_localisation, args.selector)
    run("G6_per_class", g6_per_class, args.selector)
    run("G7_minimum_detectable", g7_minimum_detectable_shift, args.selector)

    n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print("\n" + "=" * 74)
    print("%d passed, %d failed, %d total" % (n_pass, n_fail, len(RESULTS)))
    print("=" * 74)
    if n_fail:
        print("\nFailures:")
        for tid, status, detail in RESULTS:
            if status == "FAIL":
                print("  %s: %s" % (tid, detail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
