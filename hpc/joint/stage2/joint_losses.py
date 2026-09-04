"""The replicate-consistency term of eq. (2), in torch (plan v0.6, S2.5).

This module is the TRAINING-TIME implementation of what `replicate_statistic.py`
specifies in numpy. The numpy module is the specification; this one must agree
with it bit-for-bit in float64 (smoke test J16). Where they disagree, the numpy
module is right.

What is computed, per same-donor well pair (g, g')
--------------------------------------------------
    C_bar   = (C_g + C_g') / 2                                      eq. (17)
    Delta   = m_g - m_g'
    T       = Delta^T (2 C_bar)^{-1} Delta                          eq. (3)
    T_hat   = T - d_theta / S_mc            (finite-draw correction, eq. 3h)
    p_eff   = d_theta - tr(Sigma_0^{-1} C_bar)                      eq. (9)
    L_rep   = (log T_hat - log p_eff)^2                             eq. (3c)

Three gradient decisions, each blocking a different escape route (S2.5e). They
are implemented here rather than left to the training loop, because a loss that
relies on its caller to be careful is a loss that will eventually be called
carelessly.

  1. **Stop-gradient on omega.** `stop_grad_params` switches the flow's
     parameters to requires_grad=False for the duration of the replicate
     forward pass. Gradient still reaches psi THROUGH z, which is the wanted
     route; it cannot reach omega, which is the G1-failure route (make the flow
     ignore its conditioner).
  2. **Stop-gradient through C_bar.** The metric and the target are both built
     from `C_bar.detach()`, so the optimiser cannot reduce the loss by
     inflating the posterior. Note the consequence, which is deliberate: within
     an epoch p_eff is a CONSTANT, so the self-limiting effect of S2.5e (a
     wider C_bar raises p_eff too) acts only across epochs, when C_bar is
     recomputed. That is the intended behaviour and not an oversight.
  3. **The target is not zero.** eq. (3c) diverges as T -> 0+, so the constant
     map -- the global minimum of every pure invariance objective -- is pushed
     to +infinity rather than merely penalised.

Numerical choices, stated rather than buried
--------------------------------------------
* The quadratic form is evaluated by a Cholesky solve on 2*C_bar, never by
  forming an inverse.
* A jitter of `jitter * mean(diag(C_bar))` is added to the diagonal before the
  factorisation. Sample covariances from S_mc draws in d dimensions are
  singular for S_mc <= d and ill-conditioned well above that; without the
  jitter the first collapsing arm crashes the run instead of scoring badly.
  The jitter is relative, so it does not silently change the scale.
* `T_hat` is clamped below at `t_floor` before the log. The clamp is a
  numerical guard, not a modelling choice: at the default it is far below any
  T a non-degenerate posterior produces, and J11 checks that the loss still
  grows without bound as the clamp is approached.

Pure ASCII, LF only.
"""

import contextlib

import torch

__all__ = [
    "posterior_moments",
    "symmetrised_covariance",
    "replicate_statistic",
    "p_eff_from_trace",
    "replicate_loss",
    "ReplicateConsistencyLoss",
    "stop_grad_params",
    "box_prior_covariance",
]

DEFAULT_JITTER = 1e-6
DEFAULT_T_FLOOR = 1e-8
# Below this, p_eff is treated as undefined rather than as a target (see
# ReplicateConsistencyLoss.forward).
DEFAULT_P_EFF_MIN = 1e-3


# ---------------------------------------------------------------------------
# Gradient plumbing
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def stop_grad_params(target):
    """Temporarily set requires_grad=False on the given parameters.

    `target` may be an nn.Module or an ITERABLE OF PARAMETERS.

    [CORRECTION, found in Stage 3] Passing the whole estimator here is wrong
    in the real model and was wrong in `joint_train` until this was caught.
    sbi holds the embedding net INSIDE the estimator, so
    `stop_grad_params(model.estimator)` freezes psi as well as omega, and the
    replicate term then reaches nothing at all. The symptom was silent: arm A5
    produced held-out NLL identical to A1's to four decimals, which reads as
    "the replicate term does nothing much" rather than as "the replicate term
    is disconnected". Pass `model.flow_parameters()`.

    Restores the previous flags on exit, including on an exception, so a failed
    step cannot leave the flow frozen for the rest of training.
    """
    params = (list(target.parameters()) if hasattr(target, "parameters")
              else list(target))
    saved = []
    for p in params:
        saved.append((p, p.requires_grad))
        p.requires_grad_(False)
    try:
        yield params
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def posterior_moments(samples):
    """Sample mean and covariance of posterior draws.

    Parameters
    ----------
    samples : (S, d) or (B, S, d) tensor of draws from q_omega(theta | z).

    Returns
    -------
    m : (d,) or (B, d)
    C : (d, d) or (B, d, d), unbiased (ddof = 1)
    S : int, the number of draws
    """
    if samples.dim() == 2:
        squeeze = True
        samples = samples.unsqueeze(0)
    elif samples.dim() == 3:
        squeeze = False
    else:
        raise ValueError("samples must be (S, d) or (B, S, d)")

    S = samples.shape[1]
    if S < 2:
        raise ValueError("need at least 2 draws to form a covariance")
    m = samples.mean(dim=1)
    centred = samples - m.unsqueeze(1)
    C = centred.transpose(1, 2) @ centred / (S - 1)
    if squeeze:
        return m[0], C[0], S
    return m, C, S


def symmetrised_covariance(C_g, C_gp):
    """C_bar = (C_g + C_g')/2, symmetrised in the matrix sense as well.

    The explicit (X + X^T)/2 matters: a sample covariance built by matmul is
    symmetric only up to floating point, and `torch.linalg.cholesky` is not
    obliged to be forgiving about that.
    """
    C = 0.5 * (C_g + C_gp)
    return 0.5 * (C + C.transpose(-1, -2))


def box_prior_covariance(lower, upper, dtype=torch.float64, device=None):
    """Sigma_0 = diag((U - L)^2 / 12) for a uniform box prior.

    NOTE this is the box's second moment, not a Gaussian covariance. Using it
    in eq. (9) is a stated approximation (plan v0.6, S9), separate from the
    conjugacy one.
    """
    lower = torch.as_tensor(lower, dtype=dtype, device=device)
    upper = torch.as_tensor(upper, dtype=dtype, device=device)
    return torch.diag((upper - lower) ** 2 / 12.0)


# ---------------------------------------------------------------------------
# The statistic and its target
# ---------------------------------------------------------------------------

def _chol_of_twice(C_bar, jitter):
    d = C_bar.shape[-1]
    eye = torch.eye(d, dtype=C_bar.dtype, device=C_bar.device)
    scale = torch.diagonal(C_bar, dim1=-2, dim2=-1).mean(dim=-1)
    scale = scale.reshape(*scale.shape, 1, 1) if scale.dim() else scale
    A = 2.0 * (C_bar + jitter * scale * eye)
    return torch.linalg.cholesky(A)


def replicate_statistic(m_g, m_gp, C_g, C_gp, n_draws=None, correct_mc=False,
                        jitter=DEFAULT_JITTER, detach_metric=True):
    """T = Delta^T (2 C_bar)^{-1} Delta, optionally minus d/S.

    Batched over a leading pair dimension if the inputs carry one.

    `detach_metric=True` implements decision 2 of the module docstring: the
    metric carries no gradient, so the only route to a lower loss is a smaller
    Delta.
    """
    delta = (m_g - m_gp).unsqueeze(-1)                       # (..., d, 1)
    C_bar = symmetrised_covariance(C_g, C_gp)
    if detach_metric:
        C_bar = C_bar.detach()
    L = _chol_of_twice(C_bar, jitter)
    z = torch.cholesky_solve(delta, L)
    T = (delta * z).sum(dim=(-2, -1))

    if correct_mc:
        if n_draws is None:
            raise ValueError("correct_mc=True requires n_draws")
        d = m_g.shape[-1]
        T = T - float(d) / float(n_draws)
    return T


def p_eff_from_trace(Sigma0, C_bar):
    """p_eff = d - tr(Sigma_0^{-1} C_bar), eq. (9).

    Sigma_0 is a constant of the problem and C_bar arrives detached, so p_eff
    carries no gradient by construction.
    """
    d = C_bar.shape[-1]
    L0 = torch.linalg.cholesky(Sigma0)
    X = torch.cholesky_solve(C_bar, L0)                      # Sigma_0^{-1} C_bar
    tr = torch.diagonal(X, dim1=-2, dim2=-1).sum(dim=-1)
    return float(d) - tr


def replicate_loss(T, p_eff, t_floor=DEFAULT_T_FLOOR):
    """L_rep = (log T - log p_eff)^2, eq. (3c). Two-sided about the target."""
    Tc = torch.clamp(T, min=t_floor)
    pc = torch.clamp(p_eff, min=t_floor)
    return (torch.log(Tc) - torch.log(pc)) ** 2


# ---------------------------------------------------------------------------
# The module the training loop actually calls
# ---------------------------------------------------------------------------

class ReplicateConsistencyLoss(torch.nn.Module):
    """eq. (3c) with all three gradient decisions applied.

    Parameters
    ----------
    Sigma0 : (d, d) tensor
        Prior covariance. Registered as a buffer, so it moves with the module
        and is saved in the checkpoint -- the target is not reproducible
        without it.
    n_draws : int
        S_mc, the posterior draws per well.
    correct_mc : bool
        Apply the d/S correction of eq. (3h). Default True: at S_mc = 100 the
        inflation is about 10% of a p_eff of 3, which is not negligible.
    warmup : float in [0, 1)
        Fraction of training before the term ramps in. `set_progress` drives
        it. The ramp exists because at initialisation an untrained flow gives
        T near zero, i.e. the term starts on the COLLAPSE side of its target
        and would otherwise oppose the NPE term from step one (S2.5e).
    """

    def __init__(self, Sigma0, n_draws, correct_mc=True, warmup=0.0,
                 jitter=DEFAULT_JITTER, t_floor=DEFAULT_T_FLOOR,
                 p_eff_min=DEFAULT_P_EFF_MIN):
        super().__init__()
        Sigma0 = torch.as_tensor(Sigma0)
        if Sigma0.dim() != 2 or Sigma0.shape[0] != Sigma0.shape[1]:
            raise ValueError("Sigma0 must be square")
        self.register_buffer("Sigma0", Sigma0)
        self.n_draws = int(n_draws)
        self.correct_mc = bool(correct_mc)
        self.warmup = float(warmup)
        self.jitter = float(jitter)
        self.t_floor = float(t_floor)
        self.p_eff_min = float(p_eff_min)
        self._progress = 1.0

    def set_progress(self, progress):
        """`progress` in [0, 1]: fraction of planned training elapsed."""
        self._progress = float(progress)

    @property
    def ramp(self):
        if self.warmup <= 0.0:
            return 1.0
        if self._progress >= self.warmup:
            return 1.0
        return max(0.0, self._progress / self.warmup)

    def forward(self, samples_g, samples_gp):
        """samples_* : (B, S, d) draws for the two wells of each of B pairs.

        Returns
        -------
        loss : scalar tensor, the ramped mean over pairs
        info : dict of detached diagnostics -- T, p_eff, and the per-direction
            decomposition normalised by (1 - c_j), which is the falsifier for
            the shared-theta assumption (P16).
        """
        m_g, C_g, S1 = posterior_moments(samples_g)
        m_gp, C_gp, S2 = posterior_moments(samples_gp)
        if S1 != S2:
            raise ValueError("the two wells must be drawn with the same S_mc")

        T = replicate_statistic(m_g, m_gp, C_g, C_gp, n_draws=S1,
                                correct_mc=self.correct_mc, jitter=self.jitter)
        C_bar = symmetrised_covariance(C_g, C_gp).detach()
        p_eff_raw = p_eff_from_trace(self.Sigma0.to(C_bar.dtype), C_bar)

        # p_eff <= 0 means tr(Sigma_0^{-1} C_bar) >= d: the posterior is at
        # least as wide as the prior in the trace sense, so the data has
        # constrained NO effective direction. The target is then undefined --
        # there is no "right amount of disagreement" for an estimator that has
        # learned nothing. This is a G1 failure, and it is EXPECTED before the
        # warm-up ends, which is one more reason the ramp exists. Clamp so the
        # step still runs, and count it: a count that stays non-zero after the
        # ramp completes is a finding, not a numerical detail.
        p_eff = torch.clamp(p_eff_raw, min=self.p_eff_min)
        n_invalid = int((p_eff_raw <= 0.0).sum()) if p_eff_raw.dim() \
            else int(float(p_eff_raw) <= 0.0)

        per_pair = replicate_loss(T, p_eff, t_floor=self.t_floor)
        loss = self.ramp * per_pair.mean()

        info = {
            "T": T.detach(),
            "p_eff": p_eff.detach(),
            "p_eff_raw": p_eff_raw.detach(),
            "n_p_eff_invalid": n_invalid,
            "ramp": self.ramp,
            "n_draws": S1,
            "per_direction": self.per_direction(m_g - m_gp, C_bar),
        }
        return loss, info

    def per_direction(self, delta, C_bar):
        """Per-direction terms of T, each normalised by its own null share.

        In the generalised eigenbasis of (C_bar^{-1} - Sigma_0^{-1}, Sigma_0^{-1})
        the j-th term of T has expectation (1 - c_j) under H_0, so dividing by
        it makes the 26 directions comparable. A direction that sits far above
        1 across donors is evidence against H_0(i) -- the shared-theta premise
        -- rather than against calibration (P16).
        """
        with torch.no_grad():
            d = C_bar.shape[-1]
            Sigma0 = self.Sigma0.to(C_bar.dtype)
            # Generalised eigenproblem C_bar^{-1} v = (1 + lambda) Sigma_0^{-1} v
            # solved by symmetric whitening, which avoids inverting C_bar.
            L0 = torch.linalg.cholesky(Sigma0)
            # A = L0^T C_bar^{-1} L0 has eigenvalues 1 + lambda_j.
            eye = torch.eye(d, dtype=C_bar.dtype, device=C_bar.device)
            scale = torch.diagonal(C_bar, dim1=-2, dim2=-1).mean(dim=-1)
            scale = scale.reshape(*scale.shape, 1, 1) if scale.dim() else scale
            Lc = torch.linalg.cholesky(C_bar + self.jitter * scale * eye)
            M = torch.cholesky_solve(L0 if L0.dim() == C_bar.dim()
                                     else L0.expand_as(C_bar), Lc)
            A = L0.transpose(-1, -2) @ M
            A = 0.5 * (A + A.transpose(-1, -2))
            evals, evecs = torch.linalg.eigh(A)
            lam = torch.clamp(evals - 1.0, min=0.0)
            share = lam / (1.0 + lam)                    # 1 - c_j

            # Delta expressed in the same basis, in prior-whitened units.
            u = torch.linalg.solve_triangular(
                L0, delta.unsqueeze(-1), upper=False).squeeze(-1)
            coeff = (u.unsqueeze(-2) @ evecs).squeeze(-2)
            c_j = 1.0 / (1.0 + lam)
            term = coeff ** 2 / (2.0 * c_j)
            return term / torch.clamp(share, min=1e-12)
