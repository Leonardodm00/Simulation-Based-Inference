#!/usr/bin/env python3
"""
smoke_test_bench_burst.py -- Stage B + C + D gate for bench_burst_generator.py.

Every test is a claim in STAGE_A_BENCH_GENERATOR_SPEC_v1.md, tested as
written. The tests that matter most are the REDUCTION properties (B2, B9): a
generator that passes on its own outputs but fails to reduce to the simpler
model it claims to contain is wrong in a way no distributional test on the
full model can see.

Run:
    python3 smoke_test_bench_burst.py          # ~30 s
    python3 smoke_test_bench_burst.py --quick  # ~5 s, looser tolerances

Pure ASCII, LF only.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy import stats as sps

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from bench_burst_generator import (                       # noqa: E402
    BenchBurstParams, EnvelopeStats, build_envelopes, draw_durations,
    draw_participation_probs, expected_mfr, fragment_envelopes,
    generate_onsets, generate_spike_times,
)

OK = []


def check(tag, name, cond, detail=""):
    OK.append(bool(cond))
    print("  %-4s %s  %s%s" % (tag, "PASS" if cond else "FAIL", name,
                               ("   [%s]" % detail) if detail else ""))


def base(**kw):
    d = dict(lambda_b=0.25, ibi_cv=0.5, d_med=0.25, sigma_d=0.6,
             lambda_burst=100.0, lambda_bg=0.03, n_neurons=20,
             duration_s=1080.0)
    d.update(kw)
    return BenchBurstParams(**d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    reps = 40 if a.quick else 200
    rng = np.random.default_rng(20260911)

    # ---- B1: eq. (2) -- CV and mean rate are orthogonal ------------------
    print("B1  eq.(2): ibi_cv and burst_rate orthogonal in expectation")
    # z-scores against the empirical SE over replicates, not a fixed
    # tolerance: at lambda_b = 0.10 a 20 ks trace holds only ~2000 IBIs and
    # the sample CV of an exponential has SE ~0.02 there, so a fixed 0.03
    # band fails by chance every ~20 runs. (Found the hard way.)
    nrep = 8 if a.quick else 24
    for cv in (0.25, 1.0):
        for lam in (0.10, 0.40):
            p = base(lambda_b=lam, ibi_cv=cv, duration_s=20000.0)
            cvs, rates = [], []
            for _ in range(nrep):
                on = generate_onsets(p, rng)
                ibi = np.diff(on)
                cvs.append(ibi.std() / ibi.mean())
                rates.append(on.size / p.duration_s)
            cvs, rates = np.array(cvs), np.array(rates)
            # sample-SE z with nrep replicates is a t(nrep-1); use its
            # 0.999 quantile so the false-fail rate is 0.2% per cell at any
            # nrep, instead of 1% at nrep=8 with a fixed 3.5.
            tcrit = sps.t.ppf(0.999, nrep - 1)
            z_cv = (cvs.mean() - cv) / (cvs.std(ddof=1) / np.sqrt(nrep))
            z_r = (rates.mean() - lam) / (rates.std(ddof=1) / np.sqrt(nrep))
            check("B1", "cv=%.2f lam=%.2f: CV %.4f (t=%+.2f), rate %.4f (t=%+.2f), crit %.2f"
                  % (cv, lam, cvs.mean(), z_cv, rates.mean(), z_r, tcrit),
                  abs(z_cv) < tcrit and abs(z_r) < tcrit)

    # ---- B2: reduction at CV=1 -- onsets are Poisson ---------------------
    print("B2  S3.1 reduction: CV_IBI = 1 gives exponential IBIs")
    p = base(lambda_b=0.30, ibi_cv=1.0, duration_s=30000.0)
    # The exact claim is eq. (2): k_gamma = 1 and theta = 1/lambda_b, which
    # makes the IBI law Exponential(lambda_b) BY DEFINITION. Test that
    # exactly; the KS below is only a loose shape sanity check on the
    # generated onsets (p > 1e-4), because a 0.01 threshold on a fixed seed
    # fails 1% of seeds by design.
    check("B2", "eq.(2) at cv=1: k_gamma=%.1f, theta=%.4f=1/lambda_b"
          % (p.k_gamma, p.theta_gamma),
          p.k_gamma == 1.0 and abs(p.theta_gamma - 1.0 / p.lambda_b) < 1e-12)
    ibi = np.diff(generate_onsets(p, rng))
    ks = sps.kstest(ibi, "expon", args=(0.0, 1.0 / p.lambda_b)).pvalue
    check("B2", "shape: KS vs Exponential p=%.3f > 1e-4" % ks, ks > 1e-4)
    ibi2 = np.diff(generate_onsets(base(lambda_b=0.30, ibi_cv=0.25,
                                        duration_s=30000.0), rng))
    ks2 = sps.kstest(ibi2, "expon", args=(0.0, 1.0 / 0.30)).pvalue
    check("B2", "and CV=0.25 is NOT exponential: p=%.2e" % ks2, ks2 < 1e-6)

    # ---- B3: eq. (3) burn-in -- count on [0,T) is unbiased ---------------
    print("B3  eq.(3): burn-in makes the count on [0,T) unbiased")
    for cv in (0.25, 1.0):
        p = base(lambda_b=0.20, ibi_cv=cv, duration_s=200.0)
        counts = np.array([generate_onsets(p, rng).size for _ in range(reps * 5)])
        want = p.lambda_b * p.duration_s
        se = counts.std() / np.sqrt(counts.size)
        z = (counts.mean() - want) / max(se, 1e-9)
        check("B3", "cv=%.2f: mean count %.2f vs %.1f (z=%.2f)"
              % (cv, counts.mean(), want, z), abs(z) < 3.5)

    # ---- B4: eq. (4) -- raw durations are lognormal(mu_d, sigma_d) -------
    print("B4  eq.(4): duration draws are lognormal with median d_med")
    # Moment tests with known SE, not a KS at p>0.01: KS p-values are
    # uniform under H0 (verified over 300 seeds), so a 0.01 threshold fails
    # a fixed-seed gate 1% of the time by design. The moments have SE
    # sigma/sqrt(n) and sigma/sqrt(2n); a 4-sigma band is a 6e-5 false-fail.
    p = base(d_med=0.25, sigma_d=0.6)
    nd = 50000
    d = draw_durations(nd, p, rng)
    ld = np.log(d)
    z_mu = (ld.mean() - p.mu_d) / (p.sigma_d / np.sqrt(nd))
    z_sd = (ld.std(ddof=1) - p.sigma_d) / (p.sigma_d / np.sqrt(2 * nd))
    check("B4", "log-mean %.4f vs mu_d %.4f (z=%+.2f)" % (ld.mean(), p.mu_d, z_mu),
          abs(z_mu) < 4.0)
    check("B4", "log-sd %.4f vs sigma_d %.4f (z=%+.2f)" % (ld.std(ddof=1), p.sigma_d, z_sd),
          abs(z_sd) < 4.0)
    ks = sps.kstest(ld, "norm", args=(p.mu_d, p.sigma_d)).pvalue
    check("B4", "shape: KS p=%.3f > 1e-4 (loose, shape-only)" % ks, ks > 1e-4)
    check("B4", "median %.4f vs d_med %.4f" % (np.median(d), p.d_med),
          abs(np.median(d) / p.d_med - 1) < 0.02)

    # ---- B5: envelopes are valid intervals ------------------------------
    print("B5  envelopes: non-overlapping, ordered, inside [0,T)")
    for pol in ("merge", "clamp"):
        p = base(lambda_b=0.40, ibi_cv=1.0, d_med=0.35, sigma_d=0.9,
                 overlap=pol)
        on = generate_onsets(p, rng)
        env, st = build_envelopes(on, draw_durations(on.size, p, rng),
                                  p.duration_s, pol)
        ok = (env.shape[1] == 2 and np.all(env[:, 0] < env[:, 1])
              and np.all(env[1:, 0] >= env[:-1, 1])
              and env[0, 0] >= 0 and env[-1, 1] <= p.duration_s)
        check("B5", "%s: %d onsets -> %d envelopes, affected %d"
              % (pol, st.n_onsets, st.n_envelopes, st.n_affected), ok)

    # ---- B6: overlap rate reproduces the spec's measured table -----------
    # The spec's S3.2 table is P(D > IBI) PAIRWISE, which is exactly the
    # CLAMP-affected rate. The MERGE rate is strictly higher because chained
    # merges extend the open envelope, so later onsets are tested against a
    # longer interval. Both are asserted here; the spec records this.
    print("B6  S3.2 table: clamp rate matches the table; merge rate is higher")
    table = [  # (lambda_b, cv, d_med, sigma_d, spec value, tol)
        (0.40, 1.00, 0.35, 0.90, 0.1718, 0.015),
        (0.10, 0.25, 0.35, 0.90, 0.0002, 0.003),
    ]
    for lam, cv, dm, sd, want, tol in table:
        p = base(lambda_b=lam, ibi_cv=cv, d_med=dm, sigma_d=sd,
                 duration_s=40000.0)
        on = generate_onsets(p, rng); dd = draw_durations(on.size, p, rng)
        _, sc = build_envelopes(on, dd, p.duration_s, "clamp")
        _, sm = build_envelopes(on, dd, p.duration_s, "merge")
        check("B6", "lam=%.2f cv=%.2f: clamp %.4f vs spec %.4f; merge %.4f"
              % (lam, cv, sc.affected_rate, want, sm.affected_rate),
              abs(sc.affected_rate - want) < tol
              and sm.affected_rate >= sc.affected_rate - 1e-9)

    # ---- B7: merge preserves the draw; clamp shortens it -----------------
    print("B7  O-5(b) vs (a): merge keeps total drawn time, clamp loses it")
    p_m = base(lambda_b=0.40, ibi_cv=1.0, d_med=0.35, sigma_d=0.9,
               duration_s=20000.0, overlap="merge")
    on = generate_onsets(p_m, rng); dd = draw_durations(on.size, p_m, rng)
    _, sm = build_envelopes(on, dd, p_m.duration_s, "merge")
    _, sc = build_envelopes(on, dd, p_m.duration_s, "clamp")
    check("B7", "merge: envelope/drawn = %.3f (<=1 only via overlap union)"
          % (sm.total_envelope_s / sm.total_drawn_s),
          0.85 < sm.total_envelope_s / sm.total_drawn_s <= 1.0 + 1e-12)
    check("B7", "clamp: envelope/drawn = %.3f, strictly less than merge"
          % (sc.total_envelope_s / sc.total_drawn_s),
          sc.total_envelope_s < sm.total_envelope_s)
    check("B7", "merge reduces count (%d -> %d), clamp keeps it (%d)"
          % (sm.n_onsets, sm.n_envelopes, sc.n_envelopes),
          sm.n_envelopes < sm.n_onsets and sc.n_envelopes == sc.n_onsets)

    # ---- B8: eq. (14) MFR identity ---------------------------------------
    print("B8  eq.(14): empirical MFR vs the identity (merge effect reported)")
    for lam, cv in ((0.10, 0.25), (0.40, 1.00)):
        p = base(lambda_b=lam, ibi_cv=cv, d_med=0.25, sigma_d=0.6,
                 n_neurons=30, duration_s=2000.0)
        tot = 0.0; cov = 0.0
        for _ in range(max(2, reps // 20)):
            sp, st = generate_spike_times(p, rng)
            tot += sum(s.size for s in sp) / (p.n_neurons * p.duration_s)
            cov += st.total_envelope_s / p.duration_s
        n = max(2, reps // 20)
        emp = tot / n
        want = expected_mfr(p)
        # what the identity should give once overlap is accounted for
        want_cov = p.lambda_bg + p.lambda_burst * (cov / n)
        check("B8", "lam=%.2f cv=%.2f: emp %.3f, eq.(14) %.3f, overlap-"
              "corrected %.3f" % (lam, cv, emp, want, want_cov),
              abs(emp / want_cov - 1) < 0.05)

    # ---- B9: Stage B guards refuse Stage C/D parameters ------------------
    print("B9  no NotImplementedError guards remain: every axis value accepted")
    for kw in (dict(n_frag_mean=2.0), dict(duty=0.5),
               dict(p_mean=0.6, kappa=4.0), dict(p_mean=0.8, kappa=3.0),
               dict(n_frag_mean=4.0, duty=0.35, p_mean=0.45, kappa=3.0)):
        try:
            base(**kw); check("B9", "%r accepted" % kw, True)
        except NotImplementedError:
            check("B9", "%r refused (WRONG, all stages exist)" % kw, False)

    # =====================================================================
    # Stage C -- participation, spec S3.4
    # =====================================================================

    # ---- C1: eq. (9) moments of p_i --------------------------------------
    print("C1  eq.(9): p_i has mean p_bar and var p_bar(1-p_bar)/(kappa+1)")
    for pbar, kap in ((0.45, 3.0), (0.65, 8.0), (0.80, 15.0)):
        p = base(p_mean=pbar, kappa=kap, n_neurons=200000)
        pi = draw_participation_probs(p, rng)
        want_var = pbar * (1 - pbar) / (kap + 1)
        # SE of the sample mean is sqrt(var/N); SE of the sample variance
        # ~ sqrt(2/N) * var for a loose bound (Beta is not normal, so 5 sigma)
        z_m = (pi.mean() - pbar) / np.sqrt(want_var / pi.size)
        z_v = (pi.var(ddof=1) - want_var) / (want_var * np.sqrt(2.0 / pi.size))
        check("C1", "pbar=%.2f kappa=%.0f: mean %.4f (z=%+.2f), var %.5f vs %.5f (z=%+.2f)"
              % (pbar, kap, pi.mean(), z_m, pi.var(ddof=1), want_var, z_v),
              abs(z_m) < 4.0 and abs(z_v) < 5.0)
        check("C1", "  all p_i strictly inside (0,1)",
              np.all(pi > 0) and np.all(pi < 1))

    # ---- C2: eq. (10) gating -- realised recruitment tracks p_i ----------
    print("C2  eq.(10): per-neuron recruitment fraction tracks p_i")
    p = base(p_mean=0.6, kappa=4.0, n_neurons=400, lambda_b=0.40,
             ibi_cv=0.25, duration_s=4000.0)
    sp, st = generate_spike_times(p, rng)
    pi = st.extra["p_i"]
    # recover per-neuron recruitment from spike presence per envelope is
    # indirect; instead re-run the gating function directly for a clean test
    from bench_burst_generator import _participation
    Z, pi2 = _participation(p, 2000, np.random.default_rng(99))
    frac = Z.mean(axis=1)
    slope = np.polyfit(pi2, frac, 1)[0]
    resid = np.abs(frac - pi2)
    check("C2", "slope of frac vs p_i = %.3f (want 1), mean |resid| = %.4f (Bernoulli SE at 2000 = ~0.01)"
          % (slope, resid.mean()), abs(slope - 1) < 0.03 and resid.mean() < 0.015)
    check("C2", "overall recruit frac %.4f vs p_bar %.2f" % (st.extra["recruit_frac"], p.p_mean),
          abs(st.extra["recruit_frac"] - p.p_mean) < 0.05)

    # ---- C3: kappa -> inf reduction -------------------------------------
    print("C3  S3.4 reduction: kappa=inf gives p_i == p_bar exactly, no RNG use")
    p = base(p_mean=0.7, kappa=float("inf"), n_neurons=50)
    r = np.random.default_rng(5); s0 = r.bit_generator.state
    pi = draw_participation_probs(p, r)
    check("C3", "p_i all == 0.7", np.all(pi == 0.7))
    check("C3", "RNG state untouched", r.bit_generator.state == s0)

    # ---- C4: Stage B reduction is EXACT for a fixed seed -----------------
    print("C4  Stage B reduction: p_mean=1, kappa=inf reproduces Stage B bit-for-bit")
    pB = base(n_neurons=10, duration_s=300.0)                     # Stage B values
    pC = base(n_neurons=10, duration_s=300.0, p_mean=1.0, kappa=float("inf"))
    a, _ = generate_spike_times(pB, np.random.default_rng(11))
    b, _ = generate_spike_times(pC, np.random.default_rng(11))
    check("C4", "identical spike times", all(np.array_equal(x, y) for x, y in zip(a, b)))
    # and a finite kappa at p_mean=1 is the same degenerate point
    pD = base(n_neurons=10, duration_s=300.0, p_mean=1.0, kappa=4.0)
    c, stc = generate_spike_times(pD, np.random.default_rng(11))
    check("C4", "p_mean=1 with finite kappa: still all-recruited, identical",
          all(np.array_equal(x, y) for x, y in zip(a, c)) and np.all(stc.extra["p_i"] == 1.0))

    # ---- C5: eq. (14) with p_bar < 1 -------------------------------------
    print("C5  eq.(14): MFR identity with the participation factor")
    p = base(p_mean=0.55, kappa=6.0, n_neurons=60, lambda_b=0.30,
             ibi_cv=0.5, duration_s=2000.0)
    tot = 0.0; cov = 0.0; nr = max(2, reps // 20)
    for _ in range(nr):
        sp, st = generate_spike_times(p, rng)
        tot += sum(x.size for x in sp) / (p.n_neurons * p.duration_s)
        cov += st.total_envelope_s / p.duration_s
    emp = tot / nr
    want_cov = p.lambda_bg + p.lambda_burst * p.p_mean * (cov / nr)
    check("C5", "emp %.3f vs overlap-corrected identity %.3f (eq.14 naive %.3f)"
          % (emp, want_cov, expected_mfr(p)), abs(emp / want_cov - 1) < 0.06)

    # ---- C6: O-1 -- the J-shaped regime is real and in scope -------------
    print("C6  O-1: pbar=0.80, kappa=3 is J-shaped (mass at the top), kappa=15 is not")
    lo = draw_participation_probs(base(p_mean=0.8, kappa=3.0, n_neurons=100000), rng)
    hi = draw_participation_probs(base(p_mean=0.8, kappa=15.0, n_neurons=100000), rng)
    f_lo, f_hi = (lo > 0.97).mean(), (hi > 0.97).mean()
    check("C6", "P(p_i > 0.97): kappa=3 -> %.3f, kappa=15 -> %.3f" % (f_lo, f_hi),
          f_lo > 5 * max(f_hi, 1e-6) and f_lo > 0.05)

    # ---- C7: parameter validation for the new axes ----------------------
    print("C7  invalid participation parameters rejected")
    for kw in (dict(p_mean=0.0), dict(p_mean=1.2), dict(kappa=0.0), dict(kappa=-1.0)):
        try:
            base(**kw); check("C7", "%r accepted (WRONG)" % kw, False)
        except ValueError:
            check("C7", "%r rejected" % kw, True)

    # =====================================================================
    # Stage D -- fragmentation, spec S3.3
    # =====================================================================
    def envs(nf, duty, lam=0.30, cv=0.5, T=20000.0, seed=None):
        p = base(n_frag_mean=nf, duty=duty, lambda_b=lam, ibi_cv=cv, duration_s=T)
        r = np.random.default_rng(seed) if seed is not None else rng
        on = generate_onsets(p, r)
        env, _ = build_envelopes(on, draw_durations(on.size, p, r), T, "merge")
        return p, env, r

    # ---- D1: eq. (6) -- E[m] = n_frag_mean, m >= 1 ----------------------
    print("D1  eq.(6): fragment count per envelope has mean n_frag_mean, min 1")
    for nf in (1.5, 2.5, 4.0):
        p, env, r = envs(nf, 0.6)
        frags, owner, fs = fragment_envelopes(env, p, r)
        m = np.bincount(owner, minlength=env.shape[0])
        se = m.std(ddof=1) / np.sqrt(m.size)
        z = (m.mean() - nf) / se
        check("D1", "nf=%.1f: E[m]=%.3f (z=%+.2f), min=%d, P(m=1)=%.3f vs exp(-(nf-1))=%.3f"
              % (nf, m.mean(), z, m.min(), (m == 1).mean(), p.p_single_fragment),
              abs(z) < 4.0 and m.min() >= 1
              and abs((m == 1).mean() - p.p_single_fragment) < 0.02)

    # ---- D2: eqs. (7)-(8) -- fragments tile the envelope exactly -------
    print("D2  eqs.(7)-(8): fragments are inside, ordered, non-overlapping, tile to D")
    for nf, duty in ((2.5, 0.35), (4.0, 0.7), (3.0, 1.0)):
        p, env, r = envs(nf, duty)
        frags, owner, fs = fragment_envelopes(env, p, r)
        ok = True
        for j in range(env.shape[0]):
            f = frags[owner == j]
            if f.shape[0] == 0:
                ok = False; break
            D = env[j, 1] - env[j, 0]
            m = f.shape[0]
            inside = f[0, 0] >= env[j, 0] - 1e-9 and f[-1, 1] <= env[j, 1] + 1e-9
            ordered = np.all(f[1:, 0] >= f[:-1, 1] - 1e-12)
            pos = np.all(f[:, 1] > f[:, 0])
            want_active = D if m == 1 else duty * D
            active_ok = abs(np.sum(f[:, 1] - f[:, 0]) - want_active) < 1e-9 * max(1.0, D)
            # first fragment starts at t0, last ends at t0 + D (eq. 8 with r=m-1)
            ends_ok = (abs(f[0, 0] - env[j, 0]) < 1e-9) and (abs(f[-1, 1] - env[j, 1]) < 1e-9)
            if not (inside and ordered and pos and active_ok and ends_ok):
                ok = False; break
        check("D2", "nf=%.1f duty=%.2f: %d envelopes -> %d fragments, all tile exactly"
              % (nf, duty, env.shape[0], frags.shape[0]), ok)

    # ---- D3: reduction at n_frag_mean = 1 -- exact, no RNG -------------
    print("D3  S3.3 reduction: n_frag_mean=1 gives frags == env, RNG untouched")
    p, env, r = envs(1.0, 0.5, seed=3)
    s0 = r.bit_generator.state
    frags, owner, fs = fragment_envelopes(env, p, r)
    check("D3", "frags identical to envelopes", np.array_equal(frags, env))
    check("D3", "RNG state untouched", r.bit_generator.state == s0)
    check("D3", "duty=0.5 had no effect (m=1 ignores duty)", fs.active_fraction == 1.0)

    # ---- D4: Stage C reduction bit-for-bit through the full generator ---
    print("D4  full-trace reduction: n_frag_mean=1 reproduces Stage C output exactly")
    pC = base(n_neurons=12, duration_s=400.0, p_mean=0.6, kappa=5.0)
    pD = base(n_neurons=12, duration_s=400.0, p_mean=0.6, kappa=5.0,
              n_frag_mean=1.0, duty=0.4)
    a, _ = generate_spike_times(pC, np.random.default_rng(21))
    b, _ = generate_spike_times(pD, np.random.default_rng(21))
    check("D4", "identical spike times, duty=0.4 inert at n_f=1",
          all(np.array_equal(x, y) for x, y in zip(a, b)))
    pB = base(n_neurons=12, duration_s=400.0)
    pAll = base(n_neurons=12, duration_s=400.0, p_mean=1.0, kappa=float("inf"),
                n_frag_mean=1.0, duty=1.0)
    c, _ = generate_spike_times(pB, np.random.default_rng(22))
    d, _ = generate_spike_times(pAll, np.random.default_rng(22))
    check("D4", "and all four reductions together reproduce Stage B exactly",
          all(np.array_equal(x, y) for x, y in zip(c, d)))

    # ---- D5: realised active fraction -> P(m=1) + duty(1-P(m=1)) --------
    print("D5  O-6: active fraction is P(m=1) + duty*(1-P(m=1)), not duty")
    for nf, duty in ((1.5, 0.5), (4.0, 0.5)):
        p, env, r = envs(nf, duty, T=60000.0)
        frags, owner, fs = fragment_envelopes(env, p, r)
        want = p.p_single_fragment + duty * (1 - p.p_single_fragment)
        check("D5", "nf=%.1f duty=%.1f: realised %.4f vs %.4f (naive duty would say %.1f)"
              % (nf, duty, fs.active_fraction, want, duty),
              abs(fs.active_fraction - want) < 0.01)

    # ---- D6: eq. (14) with fragmentation + participation together -------
    print("D6  eq.(14): MFR identity with all ten axes active")
    p = base(n_frag_mean=3.0, duty=0.6, p_mean=0.55, kappa=6.0, n_neurons=60,
             lambda_b=0.30, ibi_cv=0.5, duration_s=2000.0)
    tot = 0.0; act = 0.0; nr = max(2, reps // 20)
    for _ in range(nr):
        sp, st = generate_spike_times(p, rng)
        tot += sum(x.size for x in sp) / (p.n_neurons * p.duration_s)
        act += st.extra["fragments"].total_fragment_s / p.duration_s
    emp = tot / nr
    want_exact = p.lambda_bg + p.lambda_burst * p.p_mean * (act / nr)
    check("D6", "emp %.3f vs realised-active identity %.3f (eq.14 refined %.3f)"
          % (emp, want_exact, expected_mfr(p)), abs(emp / want_exact - 1) < 0.06)

    # ---- D7: validation of the new axes ---------------------------------
    print("D7  invalid fragmentation parameters rejected")
    for kw in (dict(n_frag_mean=0.5), dict(duty=0.0), dict(duty=1.5)):
        try:
            base(**kw); check("D7", "%r accepted (WRONG)" % kw, False)
        except ValueError:
            check("D7", "%r rejected" % kw, True)

    # ---- B10: determinism and output contract ---------------------------
    print("B10 determinism + interface contract")
    p = base(n_neurons=8, duration_s=300.0)
    s1, _ = generate_spike_times(p, np.random.default_rng(7))
    s2, _ = generate_spike_times(p, np.random.default_rng(7))
    s3, _ = generate_spike_times(p, np.random.default_rng(8))
    check("B10", "same seed -> identical",
          all(np.array_equal(x, y) for x, y in zip(s1, s2)))
    check("B10", "different seed -> different",
          not all(np.array_equal(x, y) for x, y in zip(s1, s3)))
    check("B10", "list of n_neurons float64 sorted arrays in [0,T)",
          len(s1) == p.n_neurons
          and all(x.dtype == np.float64 and np.all(np.diff(x) >= 0)
                  and (x.size == 0 or (x[0] >= 0 and x[-1] < p.duration_s))
                  for x in s1))

    # ---- B11: parameter validation --------------------------------------
    print("B11 invalid parameters are rejected")
    for kw in (dict(lambda_b=0.0), dict(ibi_cv=0.0), dict(d_med=-1.0),
               dict(overlap="truncate")):
        try:
            base(**kw); check("B11", "%r accepted (WRONG)" % kw, False)
        except (ValueError, NotImplementedError):
            check("B11", "%r rejected" % kw, True)

    print("")
    print("SMOKE TEST: %s  (%d/%d)" % ("PASS" if all(OK) else "FAIL",
                                       sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    sys.exit(main())
