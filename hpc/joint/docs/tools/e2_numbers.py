#!/usr/bin/env python3
"""e2_numbers.py -- every [RAN] number of E2_NPE_AND_FLOWS.md, recomputed.

Torch-free: numpy and scipy only. E2 derives two identities -- the
decomposition of the expected negative log-likelihood, eq. (E2.2), and the
gain identity, eq. (E2.5) -- and quotes a few numbers that follow from them
on the project's own objects. This script checks the identities on two toys
whose every term is known in closed form, and computes the numbers.

Sources, kept apart as in e1_numbers.py:

  * [KB] values are CONSTANTS below, each with the project document it is
    quoted from (the r2 bank's row counts).
  * [REPO] values are READ from the repository: the bench prior is built by
    the repository's own `LatentSBISpec`, `simplex_centres`, `sample_prior`
    and `prior_log_prob` (hpc/joint/stage1/latent_sbi_simulator.py) and the
    `bench` provider's axis tuples (hpc/joint/stage1/bench_burst_provider.py),
    imported from the working tree after checking that hpc/joint/stage1 is
    byte-identical to the freeze commit 834eb41; the DSN's class-centre
    function is read from the freeze with `git show` and executed alone.

Blocks:
  B1  a linear-Gaussian toy on the real line: eq. (E2.2) and eq. (E2.5), both
      cases of the reference, closed form against Monte Carlo
  B2  a toy on the unit interval, theta ~ U(0, 1) and z ~ Bernoulli(theta):
      negative differential entropies, an exact posterior whose loss lies
      below its KL, and a filtered bank on which a z-ignorant flow beats the
      box floor while z carries nothing
  B3  the r2 bank: the bound -ln Pr(pass) on what a z-ignorant flow can gain
      against the box floor
  B4  the bench prior of the `bench` provider at the bank job's defaults: its
      differential entropy (the expectation of the runner's L_0), the gain a
      box floor would hand a z-ignorant flow, an identity flow's expected
      loss, and the class-centre geometry (finding F-ba)
  B5  the anchor of P4 eq. (P4.10), recomputed as a cross-check

Run from hpc/joint/docs/tools (needs git and the repository's history):

    python e2_numbers.py

Pure ASCII, LF only.
"""

from __future__ import annotations

import ast
import math
import os
import subprocess
import sys

sys.dont_write_bytecode = True      # importing the stage1 modules must not leave caches

import numpy as np
from scipy import integrate, stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))   # repo root
_STAGE1 = os.path.join(_ROOT, "hpc", "joint", "stage1")

FREEZE = "834eb41"
LBG = "hpc/dsn/latent_burst_generator.py"

# --------------------------------------------------------------------------- #
# [KB] constants, each with its source
# --------------------------------------------------------------------------- #
SIM_ROWS_RAW = 86251      # SBI_PIPELINE.md sec. 5, 6 (54 shards of rho1300v3, r2 export)
SIM_ROWS_MFR = 29616      # SBI_PIPELINE.md sec. 6 (MFR >= 0.1 Hz/electrode)
D_THETA = 26              # SBI_PIPELINE.md sec. 3 (the DUP15HD running example)

# bank job defaults of build_latent_bank.py (lines 67-68 at the freeze)
N_CLASSES = 3
TAU_OV = 0.10

N_MC = 1_000_000          # Monte Carlo rows for the toys
N_BENCH = 2_000_000       # Monte Carlo rows for the bench prior


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", _ROOT] + list(args), check=False,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def kl_normal(m1, s1, m2, s2):
    """KL( N(m1, s1^2) || N(m2, s2^2) ), elementwise, nats."""
    return np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2.0 * s2 ** 2) - 0.5


# --------------------------------------------------------------------------- #
# B1  linear-Gaussian toy
# --------------------------------------------------------------------------- #
def block_b1() -> None:
    print("== B1 linear-Gaussian toy: theta ~ N(0, 1), z = theta + sigma eps")
    sig = 0.5                         # noise scale of the summary
    m = 1.0 / (1.0 + sig ** 2)        # posterior mean slope
    s = math.sqrt(sig ** 2 / (1.0 + sig ** 2))   # posterior sd
    a, b = 0.6, 0.6                   # a deliberately wrong estimator q = N(a z, b^2)
    tau_ref = 2.0                     # a reference density N(0, tau_ref^2) != p_ev(theta)
    print("sigma = %.2f; posterior N(%.2f z, %.4f); estimator q = N(%.2f z, %.4f);"
          " second reference N(0, %.1f^2)" % (sig, m, s * s, a, b * b, tau_ref))

    # closed forms
    ez2 = 1.0 + sig ** 2
    h_cond = 0.5 * math.log(2 * math.pi * math.e * s * s)
    ekl = math.log(b / s) + (s * s + (m - a) ** 2 * ez2) / (2 * b * b) - 0.5
    loss = 0.5 * math.log(2 * math.pi * b * b) + (1 - 2 * a + a * a * ez2) / (2 * b * b)
    mi = 0.5 * math.log(1.0 + 1.0 / sig ** 2)
    h_theta = 0.5 * math.log(2 * math.pi * math.e)
    l0_ref2 = 0.5 * math.log(2 * math.pi * tau_ref ** 2) + 1.0 / (2 * tau_ref ** 2)
    kl_marg = math.log(tau_ref) + 1.0 / (2 * tau_ref ** 2) - 0.5

    # Monte Carlo
    rng = np.random.default_rng(20261004)
    th = rng.standard_normal(N_MC)
    z = th + sig * rng.standard_normal(N_MC)
    lq = stats.norm.logpdf(th, loc=a * z, scale=b)
    loss_mc = -lq.mean()
    loss_se = lq.std(ddof=1) / math.sqrt(N_MC)
    ekl_mc = kl_normal(m * z, s, a * z, b).mean()
    l0_mc = -stats.norm.logpdf(th).mean()
    l0r_mc = -stats.norm.logpdf(th, scale=tau_ref).mean()
    lq0 = stats.norm.logpdf(th)                 # z-ignorant estimator q = p_ev(theta)

    print("E H[theta|z]           closed %.6f" % h_cond)
    print("E KL(post || q)        closed %.6f   MC %.6f" % (ekl, ekl_mc))
    print("E[-log q] (the loss)   closed %.6f   MC %.6f +- %.6f" % (loss, loss_mc, loss_se))
    print("check (E2.2): loss - (E H + E KL) = %.2e (closed)" % (loss - (h_cond + ekl)))
    assert abs(loss - (h_cond + ekl)) < 1e-12
    assert abs(loss_mc - loss) < 5 * loss_se
    print("I(theta; z) = 0.5 ln(1 + 1/sigma^2) = %.6f;  H[theta] = %.6f" % (mi, h_theta))
    d1 = h_theta - loss
    print("case 1, reference = p_ev(theta): Delta = L0 - loss = %.6f;  I - E KL = %.6f"
          % (d1, mi - ekl))
    assert abs(d1 - (mi - ekl)) < 1e-12
    print("         MC: L0 = %.6f, Delta = %.6f" % (l0_mc, l0_mc - loss_mc))
    d2 = l0_ref2 - loss
    print("case 2, reference = N(0, %.1f^2): Delta = %.6f;  I + KL(p_ev||ref) - E KL = %.6f"
          " (KL(p_ev||ref) = %.6f)" % (tau_ref, d2, mi + kl_marg - ekl, kl_marg))
    assert abs(d2 - (mi + kl_marg - ekl)) < 1e-12
    print("         MC: L0 = %.6f, Delta = %.6f" % (l0r_mc, l0r_mc - loss_mc))
    print("z-ignorant q = p_ev(theta): case 1 Delta = %.6f (MC %.6f); case 2 Delta = %.6f"
          " = KL(p_ev||ref) (MC %.6f)"
          % (0.0, l0_mc + lq0.mean(), kl_marg, l0r_mc + lq0.mean()))
    print("exact posterior q: E KL = 0, Delta (case 1) = I = %.6f" % mi)


# --------------------------------------------------------------------------- #
# B2  unit-interval toy
# --------------------------------------------------------------------------- #
def block_b2() -> None:
    print("\n== B2 unit-interval toy: theta ~ U(0, 1), z | theta ~ Bernoulli(theta)")
    h_theta = 0.0                                  # uniform on (0, 1)
    h_post = float(stats.beta(2, 1).entropy())     # p(theta | z=1) = 2 theta; z=0 mirror
    mi = h_theta - h_post
    print("H[theta] = %.6f;  H[theta | z] = H(Beta(2,1)) = ln 2 ... = %.6f (closed 1/2 - ln 2 = %.6f)"
          % (h_theta, h_post, 0.5 - math.log(2.0)))
    assert abs(h_post - (0.5 - math.log(2.0))) < 1e-12
    print("I(theta; z) = %.6f nats" % mi)

    rng = np.random.default_rng(20261005)
    th = rng.uniform(size=N_MC)
    z = (rng.uniform(size=N_MC) < th).astype(np.int8)
    lq_post = np.where(z == 1, np.log(2 * th), np.log(2 * (1 - th)))
    loss_post = -lq_post.mean()
    se = lq_post.std(ddof=1) / math.sqrt(N_MC)
    print("exact posterior q: loss = E H[theta|z] = %.6f (MC %.6f +- %.6f); E KL = 0;"
          " box-floor gain = %.6f = I" % (h_post, loss_post, se, -h_post))
    assert abs(loss_post - h_post) < 5 * se
    print("   the loss (%.4f) lies BELOW its expected KL (0): on the unit interval E H <= 0"
          % h_post)
    print("prior q (uniform): loss = 0, E KL = %.6f = I, box-floor gain = 0" % mi)

    # the filtered bank: keep the rows with z = 1, Pr(pass | theta) = theta
    keep = z == 1
    p_pass = keep.mean()
    th_f = th[keep]
    kl_f = math.log(2.0) - 0.5                     # KL(2 theta || U(0,1))
    kl_f_mc = np.log(2 * th_f).mean()
    bound = -math.log(0.5)
    slack = -0.5                                   # E_{2 theta}[ln theta]
    print("filtered bank (keep z = 1): Pr(pass) = 1/2 (MC %.4f); theta-marginal 2 theta"
          % p_pass)
    print("   KL(p(theta|pass) || U) = ln 2 - 1/2 = %.6f (MC %.6f);  bound -ln Pr(pass) = %.6f;"
          "  slack E[ln Pr(pass|theta)] = %.6f" % (kl_f, kl_f_mc, bound, slack))
    assert abs(kl_f - (bound + slack)) < 1e-12
    print("   a z-ignorant flow that learned 2 theta gains %.6f nats/row over the box floor,"
          " while I(theta; z | pass) = 0 (z = 1 on every kept row)" % kl_f)


# --------------------------------------------------------------------------- #
# B3  the r2 bank
# --------------------------------------------------------------------------- #
def block_b3() -> None:
    print("\n== B3 the r2 bank: what a z-ignorant flow can gain against the box floor")
    kept = SIM_ROWS_MFR / SIM_ROWS_RAW
    print("Pr(pass) estimated by the kept fraction %d / %d = %.6f" % (SIM_ROWS_MFR, SIM_ROWS_RAW, kept))
    print("KL(p_sim(theta | pass) || p_Theta) <= -ln Pr(pass) = %.6f nats/row" % (-math.log(kept)))
    print("box floor on the unit cube: L0 = sum_k ln(1 - 0) = 0 exactly, at d_theta = %d" % D_THETA)


# --------------------------------------------------------------------------- #
# B4  the bench prior
# --------------------------------------------------------------------------- #
def _dsn_class_centres(n_classes: int, n_axes: int) -> tuple:
    """Execute the DSN's `_class_center_vectors` as read at the freeze, alone."""
    out = _git("show", "%s:%s" % (FREEZE, LBG))
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode())
    src = out.stdout.decode("utf-8")
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == "_class_center_vectors"][0]
    seg = ast.get_source_segment(src, fn)
    ns = {"np": np}
    exec(compile(seg, LBG, "exec"), ns)      # noqa: S102 -- the repository's own code
    return ns["_class_center_vectors"](n_classes, n_axes, "simplex"), fn.lineno, fn.end_lineno


def _geometry(c: np.ndarray, tau: float) -> tuple:
    cc = c - c.mean(axis=0, keepdims=True)
    nrm = np.linalg.norm(cc, axis=1)
    cos = (cc @ cc.T) / np.outer(nrm, nrm)
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    iu = np.triu_indices(c.shape[0], 1)
    return cos[iu], d[iu], d[iu] / tau


def block_b4() -> None:
    print("\n== B4 the bench prior (provider `bench`, bank job defaults C = %d, tau_ov = %.2f)"
          % (N_CLASSES, TAU_OV))
    if _git("diff", "--quiet", FREEZE, "--", "hpc/joint/stage1").returncode != 0:
        raise SystemExit("hpc/joint/stage1 differs from %s; refusing to import it" % FREEZE)
    print("hpc/joint/stage1 is identical to %s (git diff --quiet): importing it" % FREEZE)
    sys.path.insert(0, _STAGE1)
    import bench_burst_provider as bbp            # noqa: E402
    import latent_sbi_simulator as lss            # noqa: E402

    centres = lss.simplex_centres(N_CLASSES, len(bbp.BENCH_LABEL_IDX))
    spec = lss.LatentSBISpec(n_latent=len(bbp.BENCH_AXES),
                             label_idx=bbp.BENCH_LABEL_IDX,
                             class_centres=centres, tau_ov=TAU_OV)
    print("d_theta = %d axes; label axes (code indices) %r; free axes %r"
          % (spec.n_latent, spec.label_idx, spec.free_idx))

    rng = np.random.default_rng(20261006)
    cls_draw, phi = lss.sample_prior(spec, N_BENCH, rng)
    lp = lss.prior_log_prob(spec, phi)
    assert np.all(np.isfinite(lp))
    h_mc = float(-lp.mean())
    h_se = float(lp.std(ddof=1) / math.sqrt(N_BENCH))
    # the same entropy from its decomposition H[theta] = ln C + mean_c H[theta | c] - H[c | theta]:
    # the component entropies in closed form (scipy), the overlap term H[c | theta] by Monte Carlo
    h_tn = 0.0
    for cls in range(N_CLASSES):
        for j in range(len(spec.label_idx)):
            mu = centres[cls, j]
            h_tn += float(stats.truncnorm((0 - mu) / TAU_OV, (1 - mu) / TAU_OV,
                                          loc=mu, scale=TAU_OV).entropy()) / N_CLASSES
    post_c = lss.class_posterior(spec, phi)
    h_c_given = float(-np.mean(np.log(post_c[np.arange(N_BENCH), cls_draw])))
    h_dec = math.log(N_CLASSES) + h_tn - h_c_given
    print("H[p_Theta] (Monte Carlo, %d draws) = %.4f +- %.4f nats" % (N_BENCH, h_mc, h_se))
    print("   decomposition ln C + mean_c H[theta|c] - H[c|theta] = %.4f + %.4f - %.4f = %.4f"
          % (math.log(N_CLASSES), h_tn, h_c_given, h_dec))
    assert abs(h_mc - h_dec) < 5 * h_se
    print("runner's L_0 on a prior-faithful bench split: an estimate of H[p_Theta] = %.2f" % h_dec)
    print("a box floor (L_0 = 0 on the unit cube) would hand a z-ignorant flow"
          " KL(p_Theta || U) = -H = %.2f nats/row" % (-h_dec))

    # an identity flow (every spline the identity, P4 eq. (P4.10)) scored on the bench prior
    lg = np.log(phi) - np.log1p(-phi)                     # logit
    ell_id = (-stats.norm.logpdf(lg) + np.log(phi) + np.log1p(-phi)).sum(axis=1)
    print("identity flow on the bench prior: E[ell_i] = %.4f +- %.4f nats/row;"
          " excess over the entropy = KL(p_Theta || q_id) = %.2f"
          % (ell_id.mean(), ell_id.std(ddof=1) / math.sqrt(N_BENCH), ell_id.mean() - h_dec))

    # class-centre geometry (finding F-ba)
    cos_b, d_b, dt_b = _geometry(centres, TAU_OV)
    print("bench `simplex_centres(%d, %d)`: centred cosines %s (regular simplex: %.4f each)"
          % (N_CLASSES, len(spec.label_idx), np.round(cos_b, 4).tolist(), -1.0 / (N_CLASSES - 1)))
    print("   pairwise distances %s = %s tau_ov" % (np.round(d_b, 4).tolist(), np.round(dt_b, 2).tolist()))
    dsn, l0, l1 = _dsn_class_centres(N_CLASSES, len(spec.label_idx))
    cos_d, d_d, dt_d = _geometry(dsn, TAU_OV)
    print("DSN `_class_center_vectors(%d, %d, 'simplex')` (%s:%d-%d at %s): centred cosines %s"
          % (N_CLASSES, len(spec.label_idx), LBG, l0, l1, FREEZE, np.round(cos_d, 4).tolist()))
    print("   pairwise distances %s" % np.round(d_d, 4).tolist())
    for C, k in ((3, 2), (4, 3), (5, 4)):
        cb, _db, _ = _geometry(lss.simplex_centres(C, k), TAU_OV)
        print("   bench simplex_centres(%d, %d): cosines range [%.4f, %.4f], regular %.4f"
              % (C, k, cb.min(), cb.max(), -1.0 / (C - 1)))


# --------------------------------------------------------------------------- #
# B5  the anchor of P4 eq. (P4.10)
# --------------------------------------------------------------------------- #
def block_b5() -> None:
    print("\n== B5 the anchor of P4 eq. (P4.10), recomputed")
    f = lambda t: stats.logistic.pdf(t) * (stats.logistic.logpdf(t) - stats.norm.logpdf(t))
    kl, err = integrate.quad(f, -60, 60, limit=400)
    box, _ = integrate.quad(lambda u: math.log(u * (1 - u)), 0, 1)
    print("KL(standard logistic || standard normal) = %.4f per axis (quad error %.1e)" % (kl, err))
    print("   x %d = %.2f nats/row; x 10 = %.2f" % (D_THETA, D_THETA * kl, 10 * kl))
    print("E_U[ln(theta (1 - theta))] = %.4f per axis; x %d = %.1f" % (box, D_THETA, D_THETA * box))


def main() -> int:
    block_b1()
    block_b2()
    block_b3()
    block_b4()
    block_b5()
    return 0


if __name__ == "__main__":
    sys.exit(main())
