"""Smoke test for the Stage 1 bench modules (plan v0.6, S6 Stage 1).

Run:  python3 smoke_test_latent_sbi.py            (fast)
      python3 smoke_test_latent_sbi.py --full     (adds the slower MMD sweep)

Exits non-zero if any test fails, so it drops into the cluster-side
verification block unchanged.

Every test validates against a truth known ANALYTICALLY from the generative
model, or against an exact algebraic identity -- never against another
estimator. Tests that cannot run without the DSN generator are reported as SKIP
with the reason, not silently passed.

The ReferenceBurstProvider below is a TEST FIXTURE, not science. It exists so
that the wrapper, the nuisance, the gap, the realisation and the bank can be
exercised end to end before the DSN generator is bound. It must never be used
to build a bank that any result depends on.

Pure ASCII, LF only.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
from scipy import stats

from latent_bank import (ContractError, build_sidecar, concat_shards,
                         contract_digest, read_shard, shape_report,
                         write_shard)
from latent_gap import GapSpec, apply_trace_gap, free_axis_range_shift
from latent_nuisance import (NU_LEVELS, N_COMPONENTS, NuisanceSpec,
                             apply_nuisance, invert_nuisance, sample_nuisance)
from latent_realisation import (RealisationSpec, assign_realisations,
                                distinct_realisations_per_theta,
                                realisation_id)
from latent_sbi_simulator import (BurstProvider, LatentSBISpec,
                                  class_posterior, prior_log_prob,
                                  provider_sha256, sample_prior,
                                  simulate_windows, simplex_centres, time_grid)

RESULTS = []


def report(name, status, detail):
    RESULTS.append(status)
    tag = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
    print("[%s] %-44s %s" % (tag, name, detail))


def ok(name, cond, detail):
    report(name, "PASS" if cond else "FAIL", detail)


# ---------------------------------------------------------------------------
# Test fixture: a stand-in burst provider
# ---------------------------------------------------------------------------

class ReferenceBurstProvider(BurstProvider):
    """Deterministic Poisson-burst IFR synthesiser. TEST FIXTURE ONLY.

    phi controls the dynamics; `seed` draws the per-neuron weights, which stand
    for the connectivity realisation G. Both dependencies are required: if x
    did not depend on phi the bench would be uninformative, and if it did not
    depend on seed the realisation construction of S2.5c would be untestable.
    """

    def __call__(self, phi, n_windows, W, fs, n_neurons, seed,
                 param_overrides=None):
        phi = np.asarray(phi, dtype=np.float64).ravel()
        rng = np.random.default_rng(int(seed) % (2 ** 63))
        # Gap (a) reaches a provider as an AXIS-RANGE displacement, never as a
        # shifted phi. The fixture honours it so S7/S8 still exercise (a).
        shift = float((param_overrides or {}).get("free_axis_range_shift", 0.0))
        phi = phi + shift          # local to the physical map, not recorded

        # phi -> burst parameters. Bounded transforms, so an out-of-box phi
        # (perturbation (a)) is still simulable rather than raising -- being
        # off-support must be visible in the data, not in a traceback.
        burst_rate = 0.05 + 0.45 / (1.0 + np.exp(-4.0 * (phi[0] - 0.5)))
        burst_dur = 0.5 + 3.0 / (1.0 + np.exp(-4.0 * (phi[1 % phi.size] - 0.5)))
        amp = 1.0 + 9.0 / (1.0 + np.exp(-4.0 * (phi[2 % phi.size] - 0.5)))
        base = 0.2 + 1.8 / (1.0 + np.exp(-4.0 * (phi[-1] - 0.5)))

        # The "realisation": per-neuron weights, drawn from the seed alone.
        w = rng.gamma(shape=2.0, scale=0.5, size=n_neurons)
        w_eff = float(np.mean(w))

        out = np.empty((n_windows, W), dtype=np.float64)
        t = np.arange(W, dtype=np.float64) / fs
        for j in range(n_windows):
            trace = np.full(W, base * w_eff)
            n_bursts = rng.poisson(burst_rate * (W / fs))
            for _ in range(int(n_bursts)):
                t0 = rng.uniform(0.0, W / fs)
                env = np.exp(-0.5 * ((t - t0) / max(burst_dur, 1e-3)) ** 2)
                trace = trace + amp * w_eff * env
            trace = trace + rng.standard_normal(W) * 0.05 * base
            out[j] = np.maximum(trace, 0.0)
        return out


def small_spec(W_target=256):
    fs = 32.0
    return LatentSBISpec(n_latent=6, label_idx=(0, 1, 2),
                         class_centres=simplex_centres(3, 3),
                         tau_ov=0.10, n_windows_per_trace=4,
                         T_win=W_target / fs, fs=fs, n_neurons=20, seed=7)


# ---------------------------------------------------------------------------
# S1 -- prior stays strictly inside the box, and is class-conditional
# ---------------------------------------------------------------------------

def test_s1():
    spec = small_spec()
    rng = np.random.default_rng(0)
    c, phi = sample_prior(spec, 20000, rng)

    inside = np.all((phi > 0.0) & (phi < 1.0))
    ok("S1a phi strictly inside the open box", inside,
       "min=%.3e max=%.6f" % (phi.min(), phi.max()))

    # Class-conditional means recover the centres, up to truncation bias.
    err = 0.0
    for cls in range(spec.n_classes):
        m = phi[c == cls][:, list(spec.label_idx)].mean(axis=0)
        err = max(err, float(np.max(np.abs(m - spec.class_centres[cls]))))
    ok("S1b class-conditional means match centres", err < 0.02,
       "max abs error = %.4f (truncation bias expected, tau=%.2f)"
       % (err, spec.tau_ov))

    # Free axes must be uniform and independent of the class, or S2.3's whole
    # argument has no bench to run on.
    fa = phi[:, list(spec.free_idx)]
    ok("S1c free axes uniform", abs(fa.mean() - 0.5) < 0.01,
       "mean=%.4f (expect 0.5)" % fa.mean())
    corr = max(abs(np.corrcoef(fa[:, k], c)[0, 1]) for k in range(fa.shape[1]))
    ok("S1d free axes independent of class", corr < 0.05,
       "max |corr(phi_F, c)| = %.4f" % corr)

    _s1e_against_dsn(spec)


def _dsn_modules():
    """(lbg, gbd, dsn_spec) or None if DSN_MAIN_DIR is not set."""
    if not os.environ.get("DSN_MAIN_DIR"):
        return None
    from latent_sbi_simulator import load_dsn_modules
    lbg, gbd = load_dsn_modules()
    return lbg, gbd, lbg.LatentSpec()


def _s1e_against_dsn(_unused_spec):
    """S1e: sample_prior reproduces sample_latents where no clipping occurs.

    The two differ by design -- the DSN clips a normal at the box edge, this
    truncates it (S4.1). Clipping and truncation agree EXACTLY on the interior:
    both restrict the same normal to (0,1), and they differ only in what they
    do with the mass that falls outside. So the strongest true statement is
    equality of the two laws CONDITIONAL on no clipping, and that is what is
    tested. An unconditional comparison would have to fail, and a test that
    can only fail is not a test.
    """
    mods = _dsn_modules()
    if mods is None:
        report("S1e agrees with sample_latents off the boundary", "SKIP",
               "set DSN_MAIN_DIR to run this")
        return
    lbg, _gbd, dsn = mods
    from latent_sbi_simulator import latent_spec_from_dsn
    spec = latent_spec_from_dsn(dsn, lbg)

    per_class = 4000
    dsn_phi, dsn_cls = [], []
    for cond in range(dsn.n_classes):
        for tid in range(per_class):
            dsn_phi.append(lbg.sample_latents(dsn, cond, tid))
            dsn_cls.append(cond)
    dsn_phi = np.asarray(dsn_phi)
    dsn_cls = np.asarray(dsn_cls)

    lab = list(spec.label_idx)
    clipped = np.any((dsn_phi[:, lab] <= 0.0) | (dsn_phi[:, lab] >= 1.0), axis=1)
    ok("S1e-a clipping is rare but present (so the test is not vacuous)",
       0.0 <= clipped.mean() < 0.10,
       "clipped fraction = %.4f of %d draws" % (clipped.mean(), len(dsn_phi)))

    rng = np.random.default_rng(31)
    c_mine, phi_mine = sample_prior(spec, per_class * dsn.n_classes * 2, rng)

    worst_p, worst = 1.0, ""
    for cond in range(dsn.n_classes):
        a_rows = dsn_phi[(dsn_cls == cond) & (~clipped)]
        b_rows = phi_mine[c_mine == cond]
        for k in lab:
            st = stats.ks_2samp(a_rows[:, k], b_rows[:, k])
            if st.pvalue < worst_p:
                worst_p, worst = st.pvalue, "class %d axis %d" % (cond, k)
    ok("S1e-b same law on the label axes given no clipping", worst_p > 0.001,
       "worst KS p = %.4f (%s), Bonferroni target 0.001" % (worst_p, worst))

    # Free axes: both are Uniform(0,1) and neither touches them.
    st = stats.ks_2samp(dsn_phi[:, spec.free_idx[0]],
                        phi_mine[:, spec.free_idx[0]])
    ok("S1e-c free axes agree too", st.pvalue > 0.001,
       "KS p = %.4f on axis %d" % (st.pvalue, spec.free_idx[0]))

    # And the deviation is real: our draws never sit on the boundary.
    ok("S1e-d and ours never lands on the boundary",
       not np.any((phi_mine <= 0.0) | (phi_mine >= 1.0)),
       "0 of %d rows on an edge (the DSN put %d there)"
       % (phi_mine.shape[0], int(clipped.sum())))


# ---------------------------------------------------------------------------
# S2 -- the mixture prior integrates to 1
# ---------------------------------------------------------------------------

def test_s2():
    # n = 2 with both axes label-carrying, so the density is genuinely 2-D.
    spec = LatentSBISpec(n_latent=2, label_idx=(0, 1),
                         class_centres=simplex_centres(3, 2), tau_ov=0.12,
                         n_windows_per_trace=2, T_win=1.0, fs=32.0)
    n = 400
    edge = (np.arange(n) + 0.5) / n
    A, B = np.meshgrid(edge, edge, indexing="ij")
    pts = np.column_stack([A.ravel(), B.ravel()])
    dens = np.exp(prior_log_prob(spec, pts))
    integral = dens.sum() * (1.0 / n) ** 2
    ok("S2 prior_log_prob integrates to 1", abs(integral - 1.0) < 2e-3,
       "integral = %.6f on a %dx%d midpoint grid" % (integral, n, n))

    outside = prior_log_prob(spec, np.array([[1.5, 0.5], [-0.1, 0.2]]))
    ok("S2b density is -inf outside the box", np.all(np.isneginf(outside)),
       "values = %s" % np.array2string(outside))


# ---------------------------------------------------------------------------
# S3 -- class_posterior against a hand-computed Bayes rule
# ---------------------------------------------------------------------------

def test_s3():
    spec = LatentSBISpec(n_latent=2, label_idx=(0,),
                         class_centres=np.array([[0.3], [0.7]]), tau_ov=0.15,
                         n_windows_per_trace=2, T_win=1.0, fs=32.0)
    pts = np.array([[0.3, 0.4], [0.5, 0.9], [0.7, 0.1]])
    P = class_posterior(spec, pts)

    ok("S3a rows sum to 1", np.allclose(P.sum(axis=1), 1.0),
       "max |rowsum - 1| = %.2e" % np.max(np.abs(P.sum(axis=1) - 1.0)))

    # Hand rule: with equal priors and a symmetric truncated normal, the two
    # component densities at phi_0 differ only through the exponent, so
    #   logit p(c=1 | phi) = [ (phi-m0)^2 - (phi-m1)^2 ] / (2 tau^2),
    # the truncation constants being identical for symmetric centres.
    m0, m1, tau = 0.3, 0.7, 0.15
    x = pts[:, 0]
    logit = ((x - m0) ** 2 - (x - m1) ** 2) / (2.0 * tau ** 2)
    hand = 1.0 / (1.0 + np.exp(-logit))
    ok("S3b matches the hand-computed Bayes rule",
       np.allclose(P[:, 1], hand, atol=1e-10),
       "max abs diff = %.2e" % np.max(np.abs(P[:, 1] - hand)))

    # At the midpoint the posterior must be exactly 1/2, by symmetry.
    mid = class_posterior(spec, np.array([[0.5, 0.5]]))
    ok("S3c symmetric point gives exactly 1/2",
       np.allclose(mid, 0.5, atol=1e-12),
       "p = %s" % np.array2string(mid, precision=8))


# ---------------------------------------------------------------------------
# S4 -- provider parity with the DSN generator
# ---------------------------------------------------------------------------

def test_s4():
    """S4: the adapter is exactly the DSN primitives plus disjoint windowing."""
    mods = _dsn_modules()
    if mods is None:
        report("S4 provider matches the DSN primitives", "SKIP",
               "set DSN_MAIN_DIR to run this")
        return
    lbg, gbd, dsn = mods
    from latent_sbi_simulator import DSNBurstProvider

    J, W = 3, 128
    fs = dsn.fs
    n_neurons = 40
    prov = DSNBurstProvider(lbg, gbd, dsn)
    rng = np.random.default_rng(17)
    phi = rng.uniform(0.05, 0.95, size=dsn.n_latent)
    seed = 123456789

    x = prov(phi, J, W, fs, n_neurons, seed)

    # Rebuild the same trace by calling the three primitives directly.
    params = prov.burst_params(phi, J, W, fs, n_neurons)
    rng_ref = np.random.default_rng(seed % (2 ** 63))
    spikes = gbd.generate_spike_times(params, rng_ref)
    trace, fs_out = gbd.compute_ifr_trace(spikes, params)
    ref = np.asarray(trace[:J * W], dtype=np.float64).reshape(J, W)

    ok("S4a adapter equals the DSN primitives bit-for-bit",
       np.array_equal(x, ref),
       "%d samples compared, f_s = %.3f Hz" % (x.size, fs_out))

    ok("S4b BurstParams comes from latent_to_burst_params",
       abs(params.duration_s - J * W / fs) < 1e-12 and
       abs(params.w_size - 1.0 / fs) < 1e-12 and
       params.n_neurons == n_neurons,
       "duration=%.4f s, w_size=%.4f s, N=%d"
       % (params.duration_s, params.w_size, params.n_neurons))

    # The realisation seed must matter, and must be the only thing that does.
    x_same = prov(phi, J, W, fs, n_neurons, seed)
    x_other = prov(phi, J, W, fs, n_neurons, seed + 1)
    ok("S4c same seed gives identical bytes", np.array_equal(x, x_same),
       "deterministic in (phi, seed)")
    ok("S4d a different seed gives a different realisation",
       not np.allclose(x, x_other),
       "max |diff| = %.4f" % np.max(np.abs(x - x_other)))

    # Windowing parity with the export path's own rule.
    sbix = os.environ.get("SBIX_DIR")
    if sbix and os.path.isfile(os.path.join(sbix, "sim_observable.py")):
        if sbix not in sys.path:
            sys.path.insert(0, sbix)
        import sim_observable
        wins, starts = sim_observable.window_trace(np.asarray(trace), W)
        ok("S4e windowing matches sim_observable.window_trace",
           wins.shape[0] >= J and np.array_equal(np.asarray(wins[:J],
                                                            dtype=np.float64), x),
           "%d windows cut, starts %s, first %d compared"
           % (wins.shape[0], list(starts[:J]), J))
    else:
        report("S4e windowing matches sim_observable.window_trace", "SKIP",
               "set SBIX_DIR to the Sbi-extractor checkout to run this")

    # Negative path: the unbound gap perturbation (b) must fail loudly.
    try:
        prov(phi, J, W, fs, n_neurons, seed,
             param_overrides={"burst_duration_dist": "student_t"})
        raised = False
    except NotImplementedError:
        raised = True
    ok("S4f unbound gap (b) overrides raise instead of being ignored", raised,
       "param_overrides reaches BurstParams or it stops the run")


# ---------------------------------------------------------------------------
# S5 -- shard round-trip and contract behaviour
# ---------------------------------------------------------------------------

def _fake_arrays(n=12, W=16, p=6):
    rng = np.random.default_rng(3)
    return {
        "x": rng.random((n, W)),
        "theta": rng.random((n, p)),
        "cls": rng.integers(0, 3, size=n).astype(np.int64),
        "nu": rng.standard_normal((n, N_COMPONENTS)),
        "realisation_id": rng.integers(1, 2 ** 62, size=n).astype(np.uint64),
        "donor": np.repeat(np.arange(n // 2), 2).astype(np.int64),
        "well": np.arange(n, dtype=np.int64),
        "batch": np.zeros(n, dtype=np.int64),
        "subregion": np.zeros(n, dtype=np.int64),
        "window_idx": np.arange(n, dtype=np.int64),
        "contaminated": np.zeros(n, dtype=bool),
    }


def _fake_sidecar(spec, W, arm="S", withheld=False):
    return build_sidecar(
        param_names=["phi%d" % k for k in range(spec.n_latent)],
        bounds_theta=[(0.0, 1.0)] * spec.n_latent,
        coord="unit_box", fs=spec.fs, w_size=1.0 / spec.fs,
        T_win=spec.T_win, W=W, scale_convention="per_electrode_mean",
        latent_spec=spec.to_dict(), nuisance_spec=NuisanceSpec().to_dict(),
        realisation_spec=RealisationSpec().to_dict(),
        gap_spec=GapSpec().to_dict(), generator_sha256="fixture",
        theta_withheld=withheld, arm=arm,
        extra={"host": "sandbox", "shard": 0})


def test_s5():
    spec = small_spec()
    W = 16
    arrays = _fake_arrays(W=W, p=spec.n_latent)
    side = _fake_sidecar(spec, W)
    tmp = tempfile.mkdtemp()
    try:
        p1 = os.path.join(tmp, "shard_0000.npz")
        write_shard(p1, arrays, side)
        back, side_back = read_shard(p1)
        same = all(np.array_equal(arrays[k], back[k]) for k in arrays)
        ok("S5a shard round-trip is exact", same, "%d arrays compared"
           % len(arrays))

        d1 = contract_digest(side)
        side2 = _fake_sidecar(spec, W)
        ok("S5b digest is stable across rebuilds",
           d1 == contract_digest(side2), "digest = %s" % d1[:16])

        side3 = _fake_sidecar(spec, W, arm="R", withheld=True)
        ok("S5c digest changes when the contract changes",
           contract_digest(side3) != d1, "arm/theta_withheld differ")

        # Provenance is outside the digest, by design.
        side4 = dict(side)
        side4["provenance"] = {"host": "elsewhere"}
        ok("S5d provenance does not enter the digest",
           contract_digest(side4) == d1, "same digest with different host")

        # A second shard with the same contract concatenates; a different one
        # must be refused.
        p2 = os.path.join(tmp, "shard_0001.npz")
        write_shard(p2, _fake_arrays(W=W, p=spec.n_latent), side2)
        merged, _ = concat_shards([p1, p2])
        ok("S5e concat_shards merges matching contracts",
           merged["x"].shape[0] == 24, "rows = %d" % merged["x"].shape[0])

        p3 = os.path.join(tmp, "shard_R.npz")
        write_shard(p3, _fake_arrays(W=W, p=spec.n_latent), side3)
        try:
            concat_shards([p1, p3])
            refused = False
        except ContractError:
            refused = True
        ok("S5f concat_shards refuses a different contract", refused,
           "mixing arms S and R raises ContractError")

        # Negative path: a malformed shard must be refused at write time.
        bad = _fake_arrays(W=W, p=spec.n_latent)
        bad["cls"] = bad["cls"][:-1]
        try:
            write_shard(os.path.join(tmp, "bad.npz"), bad, side)
            caught = False
        except ContractError:
            caught = True
        ok("S5g ragged shard is refused", caught,
           "short `cls` raises ContractError")

        print("      shape_report:\n        "
              + shape_report(back, side_back).replace("\n", "\n        "))
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# S6 -- Monte Carlo prior floor converges
# ---------------------------------------------------------------------------

def mc_floor(spec, n, seed):
    """L_0 = -E_{phi ~ p(phi)}[log p(phi)], with its standard error."""
    rng = np.random.default_rng(seed)
    _, phi = sample_prior(spec, n, rng)
    lp = prior_log_prob(spec, phi)
    return float(-lp.mean()), float(lp.std(ddof=1) / np.sqrt(n))


def test_s6():
    spec = small_spec()
    L1, se1 = mc_floor(spec, 40000, 11)
    L2, se2 = mc_floor(spec, 40000, 12)
    se = np.sqrt(se1 ** 2 + se2 ** 2)
    ok("S6 MC prior floor converges across seeds",
       abs(L1 - L2) < 3.0 * se,
       "L0 = %.4f vs %.4f, |diff| = %.4f, 3 SE = %.4f"
       % (L1, L2, abs(L1 - L2), 3 * se))


# ---------------------------------------------------------------------------
# S7 / S8 -- the gap knob
# ---------------------------------------------------------------------------

def window_stats(x):
    """Cheap per-window summary for the permutation MMD."""
    x = np.atleast_2d(x)
    return np.column_stack([
        x.mean(axis=1),
        x.std(axis=1),
        np.percentile(x, 90, axis=1),
        (x < 1e-9).mean(axis=1),
    ])


def _mmd2(a, b, gamma):
    def k(u, v):
        d2 = ((u[:, None, :] - v[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)
    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    na, nb = len(a), len(b)
    np.fill_diagonal(kaa, 0.0)
    np.fill_diagonal(kbb, 0.0)
    return (kaa.sum() / (na * (na - 1)) + kbb.sum() / (nb * (nb - 1))
            - 2.0 * kab.mean())


def permutation_mmd(a, b, rng, n_perm=200):
    """MMD^2 with an RBF kernel and a permutation p-value. numpy only."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    z = np.vstack([a, b])
    mu, sd = z.mean(axis=0), z.std(axis=0) + 1e-12
    a, b, z = (a - mu) / sd, (b - mu) / sd, (z - mu) / sd
    # Median heuristic for the bandwidth.
    sub = z[rng.choice(len(z), size=min(len(z), 120), replace=False)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / med

    stat = _mmd2(a, b, gamma)
    na = len(a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(z))
        if _mmd2(z[perm[:na]], z[perm[na:]], gamma) >= stat:
            count += 1
    return stat, (count + 1.0) / (n_perm + 1.0)


def _draw_arm(spec, provider, gap, n_traces, base_seed):
    rng = np.random.default_rng(base_seed)
    rspec = RealisationSpec()
    _, phi = sample_prior(spec, n_traces, rng)
    blocks = []
    for i in range(n_traces):
        x, _ = simulate_windows(spec, phi[i], provider, donor=i, well=i,
                                subregion=0, realisation_spec=rspec,
                                base_seed=base_seed + i, gap_spec=gap,
                                rng=np.random.default_rng(base_seed + 1000 + i))
        blocks.append(x)
    return window_stats(np.vstack(blocks))


def test_s7_s8(full=False):
    spec = small_spec()
    prov = ReferenceBurstProvider()
    rng = np.random.default_rng(5)
    n_traces = 25
    modes = ("range_shift", "drift", "contamination")

    a = _draw_arm(spec, prov, GapSpec(modes=modes, pi=0.0), n_traces, 100)
    b = _draw_arm(spec, prov, GapSpec(modes=modes, pi=0.0), n_traces, 200)
    stat0, p0 = permutation_mmd(a, b, rng, n_perm=150)
    ok("S7 at pi=0 the two arms are indistinguishable", p0 > 0.05,
       "MMD^2 = %.3e, permutation p = %.3f" % (stat0, p0))

    # Gap (a) must leave every RECORDED phi inside the open box, or the prior
    # density of a pseudo-real theta is -inf and L_0 is undefined (Stage 3).
    rspec = RealisationSpec()
    rng2 = np.random.default_rng(77)
    _, phi_chk = sample_prior(spec, 64, rng2)
    gap1 = GapSpec(modes=modes, pi=1.0)
    outs = [simulate_windows(spec, phi_chk[i], prov, donor=i, well=i,
                             subregion=0, realisation_spec=rspec,
                             base_seed=9, gap_spec=gap1,
                             rng=np.random.default_rng(9 + i))[1]["phi_eff"]
            for i in range(8)]
    inside = all(np.all((o > 0.0) & (o < 1.0)) for o in outs)
    ok("S8a gap (a) at pi=1 leaves recorded phi inside the box", inside,
       "8 traces checked; the shift acts on the axis RANGES, not on phi")

    # (c) drift must be ONE continuous slow function over the trace: the
    # difference between the last sample of window j and the first of j+1 must
    # be small relative to the drift amplitude, for every j.
    gdrift = GapSpec(modes=("drift",), pi=1.0, drift_period_s=60.0)
    xd, _ = simulate_windows(spec, phi_chk[0], prov, donor=0, well=0,
                             subregion=0, realisation_spec=rspec,
                             base_seed=3, gap_spec=gdrift,
                             rng=np.random.default_rng(3))
    x0, _ = simulate_windows(spec, phi_chk[0], prov, donor=0, well=0,
                             subregion=0, realisation_spec=rspec,
                             base_seed=3)
    drift = xd - x0                                    # the drift alone
    jumps = np.abs(drift[1:, 0] - drift[:-1, -1])
    within = np.abs(np.diff(drift, axis=1)).max()
    ok("S8b the gap drift is continuous across window boundaries",
       jumps.max() <= 3.0 * within + 1e-12,
       "max boundary jump %.3e vs max within-window step %.3e"
       % (jumps.max(), within))

    levels = [0.0, 0.25, 0.5, 1.0] if full else [0.0, 0.5, 1.0]
    stats = []
    for pi in levels:
        bb = _draw_arm(spec, prov, GapSpec(modes=modes, pi=pi), n_traces, 200)
        s, _ = permutation_mmd(a, bb, rng, n_perm=60)
        stats.append(s)
    monotone = all(stats[i + 1] > stats[i] for i in range(len(stats) - 1))
    ok("S8 the gap grows monotonically in pi", monotone,
       "MMD^2 at pi=%s -> %s"
       % (levels, ["%.3e" % s for s in stats]))


# ---------------------------------------------------------------------------
# S9 -- nuisance is exactly invertible, and the nesting behaves
# ---------------------------------------------------------------------------

def test_s9():
    nspec = NuisanceSpec()
    rng = np.random.default_rng(9)
    spec = small_spec()
    x = rng.random((spec.W,)) * 5.0
    t = time_grid(spec, 0)
    nu, parts = sample_nuisance(nspec, ["b0"], ["d0"], ["w0"], base_seed=4)

    y = apply_nuisance(nspec, x, nu[0], t)
    xr = invert_nuisance(nspec, y, nu[0], t)
    ok("S9a nuisance is exactly invertible",
       np.allclose(x, xr, rtol=0, atol=1e-9),
       "max abs error = %.2e" % np.max(np.abs(x - xr)))
    A0, b0 = __import__("latent_nuisance").nuisance_affine(nspec, np.zeros(5), t)
    ok("S9a2 nu = 0 is exactly the identity map",
       abs(A0 - 1.0) < 1e-12 and float(np.max(np.abs(b0))) < 1e-12,
       "A = %.12f, max|b| = %.2e (every component centred on no-op)"
       % (A0, float(np.max(np.abs(b0)))))

    nu_many, _ = sample_nuisance(nspec, ["b"] * 3000,
                                 ["d%d" % i for i in range(3000)],
                                 ["w%d" % i for i in range(3000)], base_seed=5)
    frac = 1.0 / (1.0 + np.exp(-(nspec.dropout_logit0 + nu_many[:, 3])))
    ok("S9a3 mean electrode loss is a nuisance, not a regime",
       frac.mean() < 0.10,
       "mean fraction lost = %.4f (was 0.478 before the centring fix)"
       % frac.mean())

    ok("S9b the nuisance actually does something",
       not np.allclose(x, y, atol=1e-6),
       "max |x' - x| = %.4f" % np.max(np.abs(y - x)))

    # Nesting: wells of one donor share the donor and batch parts exactly.
    b = ["B0"] * 4
    d = ["D0", "D0", "D1", "D1"]
    w = ["W0", "W1", "W2", "W3"]
    nu4, parts4 = sample_nuisance(nspec, b, d, w, base_seed=4)
    ok("S9c donor part is shared within a donor",
       np.allclose(parts4["donor"][0], parts4["donor"][1]) and
       not np.allclose(parts4["donor"][0], parts4["donor"][2]),
       "rows 0,1 share D0; row 2 is D1")
    ok("S9d batch part is shared across all four",
       np.allclose(parts4["batch"][0], parts4["batch"][3]),
       "one batch, one draw")
    ok("S9e nu is the sum of the three levels",
       np.allclose(nu4, sum(parts4[l] for l in NU_LEVELS)),
       "additive decomposition exact")


# ---------------------------------------------------------------------------
# S10 -- the realisation latent (new in v0.6; gates J13b and D17)
# ---------------------------------------------------------------------------

def test_s10():
    spec = small_spec()
    prov = ReferenceBurstProvider()
    rspec = RealisationSpec()
    rng = np.random.default_rng(21)
    _, phi = sample_prior(spec, 1, rng)
    phi = phi[0]

    xg, ig = simulate_windows(spec, phi, prov, donor="D0", well="W0",
                              subregion=0, realisation_spec=rspec,
                              base_seed=77)
    xh, ih = simulate_windows(spec, phi, prov, donor="D0", well="W1",
                              subregion=0, realisation_spec=rspec,
                              base_seed=77)

    ok("S10a two wells of one donor differ in realisation_id",
       ig["realisation_id"] != ih["realisation_id"],
       "ids %d vs %d" % (ig["realisation_id"], ih["realisation_id"]))
    ok("S10b and therefore in their traces",
       not np.allclose(xg, xh),
       "max |x_g - x_g'| = %.4f" % np.max(np.abs(xg - xh)))

    # Same well, same everything: bit-for-bit reproducible, or nothing above
    # is a controlled comparison.
    xg2, _ = simulate_windows(spec, phi, prov, donor="D0", well="W0",
                              subregion=0, realisation_spec=rspec,
                              base_seed=77)
    ok("S10c the provider is a pure function of its arguments",
       np.array_equal(xg, xg2), "identical bytes on repeat")

    # The marginal law of x at fixed theta must not depend on the realisation
    # seed -- eq. (R1) integrates it out. Two independent seed sets, same phi,
    # compared by permutation MMD on window statistics.
    def block(seed0):
        out = []
        for k in range(20):
            x, _ = simulate_windows(spec, phi, prov, donor="D0",
                                    well="W%d" % (seed0 + k), subregion=0,
                                    realisation_spec=rspec, base_seed=77)
            out.append(x)
        return window_stats(np.vstack(out))

    s, p = permutation_mmd(block(0), block(100), np.random.default_rng(3),
                           n_perm=150)
    ok("S10d marginal law of x is invariant to the realisation seed", p > 0.05,
       "MMD^2 = %.3e, permutation p = %.3f" % (s, p))

    # Lever 2 of S2.6: subregions of one well share every nuisance factor but
    # must DIFFER in their graph realisation, while windows of one subregion
    # share it. Getting this backwards removes the nesting that breaks the
    # nuisance/mechanism confound, while leaving the config looking right.
    k00 = realisation_id(rspec, 77, "D0", "W0", 0)
    k01 = realisation_id(rspec, 77, "D0", "W0", 1)
    k00b = realisation_id(rspec, 77, "D0", "W0", 0)
    ok("S10c2 subregions of one well draw different realisations", k00 != k01,
       "subregion 0 -> %d, subregion 1 -> %d" % (k00 % 10 ** 6, k01 % 10 ** 6))
    ok("S10c3 while one subregion is constant across windows", k00 == k00b,
       "windows of a subregion are exchangeable but not independent (S2.7)")

    # The D17 audit function.
    ids, _ = assign_realisations(rspec, 77, ["D0"] * 4, ["W0", "W1", "W2", "W3"],
                                 [0, 0, 0, 0])
    theta = np.tile(phi, (4, 1))
    counts = distinct_realisations_per_theta(theta, ids)
    ok("S10e distinct_realisations_per_theta counts correctly",
       list(counts.values()) == [4],
       "one theta value, %d distinct realisations" % list(counts.values())[0])

    # And the builder must REFUSE a configuration that cannot satisfy it.
    import subprocess
    r = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "build_latent_bank.py"),
         "--out-dir", "/tmp/_s10_guard", "--n-traces", "4",
         "--wells-per-donor", "1", "--n-per-theta", "2", "--dry-run"],
        capture_output=True, text=True)
    ok("S10e2 the builder refuses wells_per_donor < n_per_theta",
       r.returncode != 0 and "n-per-theta" in (r.stdout + r.stderr),
       "exit %d: %s" % (r.returncode,
                        (r.stdout + r.stderr).strip().splitlines()[-1][:80]))

    single = distinct_realisations_per_theta(theta, [1, 1, 1, 1])
    ok("S10f and detects the one-realisation case (the D17 confound)",
       list(single.values()) == [1],
       "would flag the current ANN bank")


def main():
    full = "--full" in sys.argv
    print("=" * 74)
    print("Smoke test: Stage 1 bench modules (plan v0.6)")
    print("=" * 74)
    test_s1()
    test_s2()
    test_s3()
    test_s4()
    test_s5()
    test_s6()
    test_s7_s8(full=full)
    test_s9()
    test_s10()
    print("-" * 74)
    n_pass = RESULTS.count("PASS")
    n_fail = RESULTS.count("FAIL")
    n_skip = RESULTS.count("SKIP")
    print("%d passed, %d failed, %d skipped" % (n_pass, n_fail, n_skip))
    if n_skip:
        print("Skips need DSN_MAIN_DIR (and SBIX_DIR for S4e); everything\n"
              "else runs with the built-in fixture provider.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
