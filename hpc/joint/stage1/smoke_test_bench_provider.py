"""Smoke test for Stages E and F of the bench burst model.

Run:  python3 smoke_test_bench_provider.py

Stage E (bench_burst_provider): the affine map against hand-computed
endpoints, the gap-(a) shift semantics, protocol purity and seed
sensitivity (property S10), the O-2 per-unit-mean scale asserted
bit-for-bit against the manual chain AND against the eq. (14) moment
identity with a derived tolerance. Stage F (build_latent_bank wiring):
the bench spec's structural shape, the truthful per-provider
scale_convention, the 4 * d_theta draws bound at d_theta = 10, and the
--dry-run path end to end.

Tests needing the DSN's compute_ifr_trace SKIP loudly when DSN_MAIN_DIR
is not resolvable, mirroring how the Stage 1 suite treats the DSN
provider. numpy + stdlib only otherwise.

Pure ASCII, LF only.
"""

import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.abspath(os.path.join(_HERE, "..", "stage4"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_burst_generator import expected_mfr, generate_spike_times  # noqa: E402
from bench_burst_provider import (BENCH_AXES, BENCH_FREE_IDX,          # noqa: E402
                                  BENCH_LABEL_IDX, BenchBurstProvider,
                                  load_bench_provider, phi_to_bench_params)

RESULTS = []


def report(name, status, detail):
    RESULTS.append(status)
    print("[%s] %-52s %s" % (status, name, detail))


def ok(name, cond, detail):
    report(name, "PASS" if cond else "FAIL", detail)


def _dsn_dir():
    d = os.environ.get("DSN_MAIN_DIR")
    if d and os.path.isfile(os.path.join(d, "generate_burst_data.py")):
        return d
    return None


# ---------------------------------------------------------------------------
# E1 -- the affine map hits the spec S4 table exactly
# ---------------------------------------------------------------------------

def test_e1():
    lo = phi_to_bench_params(np.zeros(10), n_neurons=50, duration_s=10.0)
    hi = phi_to_bench_params(np.ones(10), n_neurons=50, duration_s=10.0)
    mid = phi_to_bench_params(np.full(10, 0.5), n_neurons=50, duration_s=10.0)
    # field order follows the spec S4 axis order
    fields = ("lambda_b", "sigma_d", "ibi_cv", "n_frag_mean", "lambda_burst",
              "p_mean", "d_med", "duty", "kappa", "lambda_bg")
    ok("E1a phi = 0 maps every axis to L_k",
       all(getattr(lo, f) == BENCH_AXES[k][1]
           for k, f in enumerate(fields)),
       "10/10 lower endpoints")
    ok("E1b phi = 1 maps every axis to U_k",
       all(getattr(hi, f) == BENCH_AXES[k][2]
           for k, f in enumerate(fields)),
       "10/10 upper endpoints")
    ok("E1c phi = 0.5 maps every axis to the midpoint",
       all(abs(getattr(mid, f) - 0.5 * (BENCH_AXES[k][1] + BENCH_AXES[k][2]))
           < 1e-12 for k, f in enumerate(fields)),
       "linear in the interpretable quantity (spec convention ii)")
    try:
        phi_to_bench_params(np.zeros(6), n_neurons=50, duration_s=10.0)
        ok("E1d a 6-vector phi is refused", False, "no error raised")
    except ValueError as exc:
        ok("E1d a 6-vector phi is refused", "10" in str(exc),
       "the bench is 10 axes, not the DSN's 6")


# ---------------------------------------------------------------------------
# E2 -- gap (a): the shift displaces FREE axis ranges only
# ---------------------------------------------------------------------------

def test_e2():
    phi = np.full(10, 0.25)
    base = phi_to_bench_params(phi, n_neurons=50, duration_s=10.0)
    shft = phi_to_bench_params(phi, n_neurons=50, duration_s=10.0,
                               free_axis_shift=0.2)
    fields = ("lambda_b", "sigma_d", "ibi_cv", "n_frag_mean", "lambda_burst",
              "p_mean", "d_med", "duty", "kappa", "lambda_bg")
    label_same = all(getattr(base, fields[k]) == getattr(shft, fields[k])
                     for k in BENCH_LABEL_IDX)
    free_moved = all(abs(getattr(shft, fields[k]) - getattr(base, fields[k])
                         - 0.2 * (BENCH_AXES[k][2] - BENCH_AXES[k][1]))
                     < 1e-12 for k in BENCH_FREE_IDX)
    ok("E2a class-bearing images are untouched by the shift", label_same,
       "perturbation (a) never leaks into the label axes")
    ok("E2b free images move by exactly shift * (U - L)", free_moved,
       "same semantics as DSNBurstProvider.burst_params")
    zero = phi_to_bench_params(phi, n_neurons=50, duration_s=10.0,
                               free_axis_shift=0.0)
    ok("E2c shift = 0 is the identity",
       all(getattr(base, f) == getattr(zero, f) for f in fields),
       "pi = 0 (arm S) costs nothing")


# ---------------------------------------------------------------------------
# E3 -- unknown override keys are refused, not ignored
# ---------------------------------------------------------------------------

def test_e3():
    class _GBDStub(object):
        BurstParams = None
        compute_ifr_trace = None
    prov = BenchBurstProvider(_GBDStub())
    try:
        prov(np.full(10, 0.5), 2, 100, 50.0, 20, seed=1,
             param_overrides={"no_such_key": 1.0})
        ok("E3 an unbound override key raises", False, "no error raised")
    except NotImplementedError as exc:
        ok("E3 an unbound override key raises",
           "no_such_key" in str(exc),
           "silent degradation is the heavy_tail failure; refuse instead")


# ---------------------------------------------------------------------------
# E4 -- protocol: shape, dtype, non-negativity, purity      [needs DSN dir]
# E5 -- seed and phi sensitivity (property S10)             [needs DSN dir]
# E6 -- O-2 bit-for-bit: provider == manual chain / N       [needs DSN dir]
# E7 -- O-2 moment identity, eq. (14)                       [needs DSN dir]
# ---------------------------------------------------------------------------

def test_e4_to_e7():
    d = _dsn_dir()
    if d is None:
        for t in ("E4", "E5", "E6", "E7"):
            report(t, "SKIP", "DSN_MAIN_DIR not resolvable; "
                              "compute_ifr_trace unavailable")
        return
    prov = load_bench_provider(d)
    phi = np.array([0.3, 0.2, 0.4, 0.0, 0.5, 0.6, 0.3, 0.7, 1.0, 0.5])
    args = dict(n_windows=2, W=500, fs=50.0, n_neurons=40)

    x1 = prov(phi, seed=7, **args)
    x2 = prov(phi, seed=7, **args)
    ok("E4a shape and dtype follow the protocol",
       x1.shape == (2, 500) and x1.dtype == np.float64,
       "(n_windows, W) float64")
    ok("E4b the trace is non-negative", bool(np.all(x1 >= 0.0)),
       "clip inside compute_ifr_trace survives the division")
    ok("E4c pure: same arguments, same bytes",
       np.array_equal(x1, x2), "protocol requirement, verbatim")

    x3 = prov(phi, seed=8, **args)
    phi_b = phi.copy()
    phi_b[0] = 0.9
    x4 = prov(phi_b, seed=7, **args)
    ok("E5a a different seed changes the trace",
       not np.array_equal(x1, x3),
       "S10: seed is where the realisation enters")
    ok("E5b a different phi changes the trace",
       not np.array_equal(x1, x4),
       "the bench must be informative about phi")

    # E6: the O-2 divisor, pinned bit-for-bit against the manual chain.
    from latent_sbi_simulator import load_dsn_modules
    _lbg, gbd = load_dsn_modules(d)
    need = args["n_windows"] * args["W"]
    dt = 1.0 / args["fs"]
    dur = (need + BenchBurstProvider.PAD_BINS) * dt
    params = phi_to_bench_params(phi, n_neurons=args["n_neurons"],
                                 duration_s=dur)
    spikes, _ = generate_spike_times(params,
                                     np.random.default_rng(7 % (2 ** 63)))
    ifr_params = gbd.BurstParams(n_neurons=args["n_neurons"], duration_s=dur,
                                 w_size=dt, gaussian_window=0.04)
    raw, _fs = gbd.compute_ifr_trace(spikes, ifr_params)
    manual = (np.asarray(raw[:need], dtype=np.float64)
              / float(args["n_neurons"])).reshape(2, 500)
    ok("E6 provider output is the manual chain / n_neurons, bit-for-bit",
       np.array_equal(x1, manual),
       "O-2: per-unit mean, divisor = N, nothing else")

    # E7: eq. (14) as a trace-level diagnostic -- the payoff of the mean
    # convention. Parameters chosen in the low-overlap, unfragmented,
    # high-kappa regime so eq. (14)'s ignore-the-overlap-policy caveat
    # costs ~1.6% (lambda_b * E[D] = 0.10 * 0.157) and delta_eff = 1.
    # Tolerance: rel sd of the trace mean over T = 960 s is
    # sqrt(2.6%^2 [renewal count, CV/sqrt(lambda_b T)]
    #    + 3.1%^2 [lognormal duration mean, sqrt((e^{sigma^2}-1)/J)]
    #    + 2.8%^2 [participation mean over N = 100 at kappa = 15]
    #    + 0.4%^2 [Poisson spikes]) ~= 4.9%; 4 sigma + 1.6% bias < 0.22.
    phi_m = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 1.0, 0.5])
    m_args = dict(n_windows=8, W=6000, fs=50.0, n_neurons=100)
    xm = prov(phi_m, seed=11, **m_args)
    dur_m = (8 * 6000 + BenchBurstProvider.PAD_BINS) / 50.0
    params_m = phi_to_bench_params(phi_m, n_neurons=100, duration_s=dur_m)
    target = expected_mfr(params_m)
    measured = float(xm.mean()) * m_args["fs"]
    rel = abs(measured - target) / target
    ok("E7 time-mean(x) * fs matches expected_mfr within 4 sigma",
       rel < 0.22,
       "measured %.4f vs eq.(14) %.4f Hz/unit, rel err %.3f"
       % (measured, target, rel))


# ---------------------------------------------------------------------------
# F1 -- make_provider_and_spec: the bench branch and the truthful record
# F2 -- the 4 * d_theta draws bound resolves to 40 at d_theta = 10
# F3 -- build_latent_bank --dry-run runs end to end          [needs DSN dir]
# ---------------------------------------------------------------------------

def test_f1():
    from build_latent_bank import build_parser, make_provider_and_spec
    _prov, tag, spec, sc = make_provider_and_spec(
        build_parser().parse_args(["--out-dir", "/tmp/x",
                                   "--provider", "reference"]))
    ok("F1a the reference record is now truthful",
       tag == "reference-fixture" and sc == "sum_over_units",
       "the per_electrode_mean claim no provider implemented is gone")
    d = _dsn_dir()
    if d is None:
        report("F1b", "SKIP", "DSN_MAIN_DIR not resolvable")
        return
    _prov, tag, spec, sc = make_provider_and_spec(
        build_parser().parse_args(["--out-dir", "/tmp/x", "--provider",
                                   "bench", "--dsn-main-dir", d]))
    ok("F1b the bench spec is structural: 10 axes, 7 class-bearing",
       tag == "bench" and sc == "per_unit_mean"
       and spec.n_latent == 10 and spec.label_idx == (0, 1, 2, 3, 4, 5, 6)
       and spec.free_idx == (7, 8, 9) and spec.class_centres.shape == (3, 7),
       "n_latent and label_idx come from BENCH_AXES, not from flags")


def test_f2():
    from joint_space import default_joint_space
    sp = default_joint_space(p=10, embedding_dim=10, d_theta=10)
    ok("F2 n_posterior_draws lower bound is 4 * d_theta = 40",
       sp.n_posterior_draws[0] == 40,
       "the 24 -> 40 ripple is parametric; verified, not patched")


def test_f3():
    d = _dsn_dir()
    if d is None:
        report("F3", "SKIP", "DSN_MAIN_DIR not resolvable")
        return
    r = subprocess.run(
        [sys.executable, os.path.join(_HERE, "build_latent_bank.py"),
         "--out-dir", "/tmp/bench_dry", "--provider", "bench",
         "--dsn-main-dir", d, "--n-traces", "9", "--wells-per-donor", "9",
         "--dry-run"],
        capture_output=True, text=True, cwd=_HERE)
    out = r.stdout + r.stderr
    ok("F3 --dry-run with --provider bench runs end to end",
       r.returncode == 0 and "provider            : bench" in out
       and "W                   : 3000" in out,
       "the load path (DSN import included) is exercised, nothing written")


def main():
    print("=" * 80)
    print("Smoke test: Stages E and F -- bench provider and bank wiring")
    print("=" * 80)
    test_e1()
    test_e2()
    test_e3()
    test_e4_to_e7()
    test_f1()
    test_f2()
    test_f3()
    print("-" * 80)
    n_fail = RESULTS.count("FAIL")
    print("%d passed, %d failed, %d skipped"
          % (RESULTS.count("PASS"), n_fail, RESULTS.count("SKIP")))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
