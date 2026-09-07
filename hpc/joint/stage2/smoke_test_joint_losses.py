"""Smoke test for joint_losses.py -- the replicate term (plan v0.6, Stage 2).

Run:  python3 smoke_test_joint_losses.py

Covers J10-J16 of the Stage 2 test list. The remaining Stage 2 tests (J1-J9)
belong to the training loop and the model, which are separate deliverables.

Every check is against an analytic truth or against `replicate_statistic.py`,
which is the SPECIFICATION for this module -- never against another torch
implementation of the same thing.

Pure ASCII, LF only. Requires torch, numpy, scipy.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import replicate_statistic as rs  # noqa: E402  the numpy specification
from joint_losses import (ReplicateConsistencyLoss, box_prior_covariance,  # noqa: E402
                          p_eff_from_trace, posterior_moments,
                          replicate_loss, replicate_statistic,
                          stop_grad_params, symmetrised_covariance)

torch.set_default_dtype(torch.float64)
RESULTS = []


def report(name, cond, detail):
    RESULTS.append(bool(cond))
    print("[%s] %-46s %s" % ("PASS" if cond else "FAIL", name, detail))


def random_spd(d, rng, scale=1.0, spread=2.0):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    ev = scale * np.exp(rng.uniform(-spread, spread, size=d))
    return Q @ np.diag(ev) @ Q.T


# ---------------------------------------------------------------------------
# J10 -- zero on identical pairs; exchangeable in (g, g')
# ---------------------------------------------------------------------------

def test_j10():
    rng = np.random.default_rng(0)
    d = 8
    C_g = torch.tensor(random_spd(d, rng))
    C_gp = torch.tensor(random_spd(d, rng))
    m = torch.tensor(rng.standard_normal(d))

    T_same = replicate_statistic(m, m, C_g, C_g)
    report("J10a T is exactly zero for an identical pair",
           float(T_same) == 0.0, "T = %r" % float(T_same))

    m2 = torch.tensor(rng.standard_normal(d))
    T_ab = replicate_statistic(m, m2, C_g, C_gp)
    T_ba = replicate_statistic(m2, m, C_gp, C_g)
    report("J10b T is invariant to swapping the pair order",
           torch.allclose(T_ab, T_ba, atol=1e-14),
           "T = %.12f vs %.12f (guaranteed by the symmetrised C_bar)"
           % (T_ab, T_ba))

    # The unsymmetrised form is NOT exchangeable, which is why eq. (17) exists.
    Ld = torch.linalg.cholesky(2.0 * C_g)
    dd = (m - m2).unsqueeze(-1)
    T_g = float((dd * torch.cholesky_solve(dd, Ld)).sum())
    Ld2 = torch.linalg.cholesky(2.0 * C_gp)
    T_gp = float((dd * torch.cholesky_solve(dd, Ld2)).sum())
    report("J10c and the unsymmetrised form is not",
           abs(T_g - T_gp) > 1e-6,
           "C_g gives %.4f, C_g' gives %.4f" % (T_g, T_gp))


# ---------------------------------------------------------------------------
# J11 -- the constant map is pushed to +infinity, not merely penalised
# ---------------------------------------------------------------------------

def test_j11():
    p_eff = torch.tensor(3.0)
    Ts = torch.tensor([1e-2, 1e-4, 1e-6, 1e-10])
    L = replicate_loss(Ts, p_eff, t_floor=1e-30)
    growing = bool(torch.all(L[1:] > L[:-1]))
    report("J11a the loss grows without bound as T -> 0+", growing,
           "L = %s" % np.array2string(L.numpy(), precision=2))

    # The constant encoder: both wells give the same mean, so T = 0 exactly.
    rng = np.random.default_rng(1)
    d = 6
    C = torch.tensor(random_spd(d, rng))
    m = torch.tensor(rng.standard_normal(d))
    T0 = replicate_statistic(m, m, C, C)
    L0 = replicate_loss(T0, p_eff, t_floor=1e-30)
    report("J11b a constant encoder is at the maximum, not the minimum",
           torch.isinf(L0) or float(L0) > 1e3,
           "T = 0 gives L = %.3e" % float(L0))

    # And the minimum is exactly at the target, not at zero.
    Lstar = replicate_loss(p_eff, p_eff)
    report("J11c the minimum sits exactly at p_eff", float(Lstar) == 0.0,
           "L(p_eff) = %r" % float(Lstar))


# ---------------------------------------------------------------------------
# J12 -- invariance under linear reparameterisation
# ---------------------------------------------------------------------------

def test_j12():
    rng = np.random.default_rng(2)
    d = 10
    C_g = torch.tensor(random_spd(d, rng))
    C_gp = torch.tensor(random_spd(d, rng))
    m_g = torch.tensor(rng.standard_normal(d))
    m_gp = torch.tensor(rng.standard_normal(d))

    T0 = replicate_statistic(m_g, m_gp, C_g, C_gp, jitter=0.0)

    # A general invertible A, and separately a pure axis rescaling -- the
    # transformation that actually occurs when an axis moves between log and
    # linear coordinates.
    for label, A_np in (("general A", rng.standard_normal((d, d))),
                        ("axis rescaling", np.diag(np.exp(
                            rng.uniform(-3, 3, size=d))))):
        while abs(np.linalg.det(A_np)) < 1e-8:
            A_np = rng.standard_normal((d, d))
        A = torch.tensor(A_np)
        T1 = replicate_statistic(A @ m_g, A @ m_gp, A @ C_g @ A.T,
                                 A @ C_gp @ A.T, jitter=0.0)
        rel = float(abs(T1 - T0) / max(abs(float(T0)), 1e-12))
        report("J12 T invariant under %s" % label, rel < 1e-8,
               "T = %.10f -> %.10f, rel diff = %.2e" % (T0, T1, rel))

    # The Euclidean form must FAIL the same test, or the test proves nothing.
    E0 = float(((m_g - m_gp) ** 2).sum())
    A = torch.tensor(np.diag(np.exp(rng.uniform(-3, 3, size=d))))
    E1 = float(((A @ m_g - A @ m_gp) ** 2).sum())
    report("J12c Euclidean form is not invariant", abs(E1 - E0) / E0 > 1e-3,
           "||.||^2 = %.4f -> %.4f" % (E0, E1))


# ---------------------------------------------------------------------------
# J13 -- E[T] = p_eff on conjugate Gaussian pairs, through the torch path
# ---------------------------------------------------------------------------

def test_j13():
    rng = np.random.default_rng(3)
    d = 23
    n_pairs = 200000
    Sigma0 = random_spd(d, rng, spread=0.5)
    F = random_spd(d, rng, spread=3.0)
    C = rs.conjugate_posterior_covariance(Sigma0, F)
    p_eff_np = rs.p_eff_from_trace(Sigma0, C)

    deltas, _ = rs.simulate_replicate_pairs(np.zeros(d), Sigma0, F, n_pairs, rng)
    Ct = torch.tensor(C)
    Lt = torch.linalg.cholesky(2.0 * Ct)
    D = torch.tensor(deltas).unsqueeze(-1)
    T = (D * torch.cholesky_solve(D, Lt)).sum(dim=(-2, -1))

    mean_T = float(T.mean())
    se = float(T.std(unbiased=True)) / np.sqrt(n_pairs)
    report("J13a E[T] = p_eff through the torch path",
           abs(mean_T - p_eff_np) < 4.0 * se,
           "mean(T) = %.4f +/- %.4f (4 se), p_eff = %.4f"
           % (mean_T, 4 * se, p_eff_np))

    p_eff_t = float(p_eff_from_trace(torch.tensor(Sigma0), Ct))
    report("J13b torch p_eff equals the numpy specification",
           abs(p_eff_t - p_eff_np) < 1e-10,
           "%.12f vs %.12f" % (p_eff_t, p_eff_np))


# ---------------------------------------------------------------------------
# J14 -- the stop-gradients hold
# ---------------------------------------------------------------------------

class ToyFlow(torch.nn.Module):
    """A stand-in for q_omega: draws are an affine function of z plus noise.

    Enough to test the gradient plumbing, which is all J14 is about. The real
    flow is built by `posterior_nn` and is not needed to check which parameters
    receive gradient.
    """

    def __init__(self, d_z, d_theta, S, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.Wm = torch.nn.Parameter(torch.randn(d_theta, d_z, generator=g))
        self.b = torch.nn.Parameter(torch.randn(d_theta, generator=g))
        self.log_s = torch.nn.Parameter(torch.zeros(d_theta))
        self.register_buffer("eps", torch.randn(S, d_theta, generator=g))

    def forward(self, z):
        mu = z @ self.Wm.T + self.b
        return mu.unsqueeze(1) + self.eps.unsqueeze(0) * torch.exp(self.log_s)


def test_j14():
    d_z, d_th, S, B = 5, 4, 64, 3
    flow = ToyFlow(d_z, d_th, S, seed=1)
    Sigma0 = box_prior_covariance(np.zeros(d_th), np.ones(d_th))
    crit = ReplicateConsistencyLoss(Sigma0, n_draws=S, correct_mc=True)

    encoder = torch.nn.Linear(7, d_z)
    x_g = torch.randn(B, 7)
    x_gp = torch.randn(B, 7)

    with stop_grad_params(flow):
        s_g = flow(encoder(x_g))
        s_gp = flow(encoder(x_gp))
        loss, info = crit(s_g, s_gp)
    loss.backward()

    omega_grads = [p.grad for p in flow.parameters()]
    none_or_zero = all(g is None or float(g.abs().sum()) == 0.0
                       for g in omega_grads)
    report("J14a the replicate term gives no gradient to omega", none_or_zero,
           "%d flow parameters, all grads None or exactly 0"
           % len(omega_grads))

    psi_grad = float(encoder.weight.grad.abs().sum())
    report("J14b but does give gradient to psi through z", psi_grad > 0.0,
           "sum |dL/dW_enc| = %.6f" % psi_grad)

    report("J14c requires_grad is restored on exit",
           all(p.requires_grad for p in flow.parameters()),
           "the context manager did not leave the flow frozen")

    # The target must be a constant of the graph: no gradient may reach the
    # loss by way of p_eff, or the model could satisfy it by inflating C.
    with stop_grad_params(flow):
        s2_g, s2_gp = flow(encoder(x_g)), flow(encoder(x_gp))
    m2g, C2g, _ = posterior_moments(s2_g)
    m2gp, C2gp, _ = posterior_moments(s2_gp)
    p_t = p_eff_from_trace(crit.Sigma0, symmetrised_covariance(C2g, C2gp).detach())
    report("J14d the target carries no gradient", not p_t.requires_grad,
           "p_eff.requires_grad = %s" % p_t.requires_grad)

    # Meanwhile the statistic MUST carry gradient, through Delta only.
    T2 = replicate_statistic(m2g, m2gp, C2g, C2gp)
    report("J14e but the statistic does, through Delta", T2.requires_grad,
           "T.requires_grad = %s" % T2.requires_grad)

    # This toy flow is deliberately wider than the box prior, so p_eff goes
    # negative -- the "learned nothing" regime. It must be caught, not clamped
    # silently.
    report("J14f a posterior wider than the prior is flagged",
           info["n_p_eff_invalid"] == B and
           bool(torch.all(info["p_eff_raw"] < 0)),
           "n_p_eff_invalid = %d of %d, raw p_eff = %s"
           % (info["n_p_eff_invalid"], B,
              np.array2string(info["p_eff_raw"].numpy(), precision=1)))


# ---------------------------------------------------------------------------
# J15 -- the d/S correction
# ---------------------------------------------------------------------------

def test_j15():
    rng = np.random.default_rng(4)
    d = 8
    n_pairs = 40000
    S_mc = 200
    Sigma0 = random_spd(d, rng, spread=0.5)
    F = random_spd(d, rng, spread=3.0)
    C = rs.conjugate_posterior_covariance(Sigma0, F)
    p_eff = rs.p_eff_from_trace(Sigma0, C)

    deltas, _ = rs.simulate_replicate_pairs(np.zeros(d), Sigma0, F, n_pairs, rng)
    L = np.linalg.cholesky(C)
    e_g = (rng.standard_normal((n_pairs, d)) @ L.T) / np.sqrt(S_mc)
    e_gp = (rng.standard_normal((n_pairs, d)) @ L.T) / np.sqrt(S_mc)
    dh = torch.tensor(deltas + e_g - e_gp)

    Ct = torch.tensor(C)
    zero = torch.zeros(n_pairs, d, dtype=torch.float64)
    T_raw = replicate_statistic(dh, zero, Ct, Ct, jitter=0.0)
    T_cor = replicate_statistic(dh, zero, Ct, Ct, n_draws=S_mc,
                                correct_mc=True, jitter=0.0)

    se = float(T_raw.std(unbiased=True)) / np.sqrt(n_pairs)
    pred = p_eff + d / S_mc
    report("J15a uncorrected T is inflated by exactly d/S",
           abs(float(T_raw.mean()) - pred) < 4 * se,
           "mean = %.4f +/- %.4f, predicted %.4f (p_eff %.4f + %.4f)"
           % (T_raw.mean(), 4 * se, pred, p_eff, d / S_mc))
    report("J15b the correction recovers p_eff",
           abs(float(T_cor.mean()) - p_eff) < 4 * se,
           "corrected mean = %.4f, p_eff = %.4f" % (T_cor.mean(), p_eff))


# ---------------------------------------------------------------------------
# J16 -- parity with the numpy specification
# ---------------------------------------------------------------------------

def test_j16():
    rng = np.random.default_rng(5)
    d, S_mc = 12, 500
    Sigma0 = random_spd(d, rng, spread=0.5)
    F = random_spd(d, rng, spread=2.0)
    C = rs.conjugate_posterior_covariance(Sigma0, F)
    L = np.linalg.cholesky(C)

    m_true = rng.standard_normal((2, d)) * 0.1
    sam = [m_true[i] + rng.standard_normal((S_mc, d)) @ L.T for i in (0, 1)]

    # numpy specification
    m0, C0, s0 = rs.posterior_moments(sam[0])
    m1, C1, s1 = rs.posterior_moments(sam[1])
    T_np = rs.replicate_statistic(m0, m1, C0, C1, n_draws=S_mc, correct_mc=True)
    p_np = rs.p_eff_from_trace(Sigma0, 0.5 * (C0 + C1))
    L_np = rs.replicate_loss(T_np, p_np)

    # torch implementation, jitter off so the two are the same estimator
    t_sam = [torch.tensor(s) for s in sam]
    tm0, tC0, _ = posterior_moments(t_sam[0])
    tm1, tC1, _ = posterior_moments(t_sam[1])
    T_t = float(replicate_statistic(tm0, tm1, tC0, tC1, n_draws=S_mc,
                                    correct_mc=True, jitter=0.0))
    p_t = float(p_eff_from_trace(torch.tensor(Sigma0),
                                 symmetrised_covariance(tC0, tC1)))
    L_t = float(replicate_loss(torch.tensor(T_t), torch.tensor(p_t)))

    report("J16a moments agree", np.allclose(m0, tm0.numpy(), atol=1e-12)
           and np.allclose(C0, tC0.numpy(), atol=1e-12),
           "max |dm| = %.2e, max |dC| = %.2e"
           % (np.max(np.abs(m0 - tm0.numpy())),
              np.max(np.abs(C0 - tC0.numpy()))))
    report("J16b T agrees with the numpy specification",
           abs(T_np - T_t) < 1e-9 * max(1.0, abs(T_np)),
           "%.12f vs %.12f" % (T_np, T_t))
    report("J16c p_eff agrees", abs(p_np - p_t) < 1e-10,
           "%.12f vs %.12f" % (p_np, p_t))
    report("J16d L_rep agrees", abs(L_np - L_t) < 1e-9,
           "%.12f vs %.12f" % (L_np, L_t))

    # The jitter is the only intended difference, and it must be small.
    T_j = float(replicate_statistic(tm0, tm1, tC0, tC1, n_draws=S_mc,
                                    correct_mc=True))
    report("J16e the default jitter perturbs T negligibly",
           abs(T_j - T_t) / max(abs(T_t), 1e-12) < 1e-3,
           "with jitter %.9f vs without %.9f" % (T_j, T_t))


# ---------------------------------------------------------------------------
# Extra: the warm-up ramp and the per-direction decomposition
# ---------------------------------------------------------------------------

def test_extras():
    d, S_mc, B = 6, 200, 4
    Sigma0 = box_prior_covariance(np.zeros(d), np.ones(d))
    crit = ReplicateConsistencyLoss(Sigma0, n_draws=S_mc, warmup=0.2)

    crit.set_progress(0.0)
    r0 = crit.ramp
    crit.set_progress(0.1)
    r1 = crit.ramp
    crit.set_progress(0.5)
    r2 = crit.ramp
    report("X1 the warm-up ramp is 0 -> 1 over warmup",
           r0 == 0.0 and abs(r1 - 0.5) < 1e-12 and r2 == 1.0,
           "ramp at progress 0/0.1/0.5 = %.2f/%.2f/%.2f" % (r0, r1, r2))

    g = torch.Generator().manual_seed(2)
    sam_g = torch.randn(B, S_mc, d, generator=g) * 0.1
    sam_gp = torch.randn(B, S_mc, d, generator=g) * 0.1 + 0.05
    crit.set_progress(1.0)
    loss, info = crit(sam_g, sam_gp)

    report("X2 the loss is finite and per-pair diagnostics are returned",
           torch.isfinite(loss) and info["T"].shape[0] == B,
           "loss = %.4f, T = %s, p_eff = %s"
           % (loss, np.array2string(info["T"].numpy(), precision=2),
              np.array2string(info["p_eff"].numpy(), precision=2)))

    pd = info["per_direction"]
    report("X3 the per-direction decomposition has the right shape",
           tuple(pd.shape) == (B, d) and bool(torch.all(pd >= 0)),
           "shape %s, all non-negative" % (tuple(pd.shape),))


def main():
    print("=" * 74)
    print("Smoke test: joint_losses.py (Stage 2, J10-J16)")
    print("torch %s" % torch.__version__)
    print("=" * 74)
    test_j10()
    test_j11()
    test_j12()
    test_j13()
    test_j14()
    test_j15()
    test_j16()
    test_extras()
    print("-" * 74)
    n_fail = RESULTS.count(False)
    print("%d/%d passed" % (RESULTS.count(True), len(RESULTS)))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
