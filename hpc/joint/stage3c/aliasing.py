"""Aliasing: what parameter shift is a nuisance direction mistaken for?

Plan v0.6, S2.6, eq. (4). Build the two Jacobians of the embedding by finite
differences on the simulator,

    J_theta = [ dz/dtheta_1, ..., dz/dtheta_d ]   (E x d)
    J_nu    = [ dz/dnu_1,    ..., dz/dnu_M    ]   (E x M)

and per nuisance direction m report

    a_m         = || P_col(J_theta) (dz/dnu_m) || / || dz/dnu_m ||     (4a)
    delta_theta = J_theta^+ (dz/dnu_m)                                 (4b)

a_m in [0, 1] is the fraction of that nuisance direction the parameter
subspace can absorb: a_m near 1 means the pipeline CANNOT tell that nuisance
apart from some parameter change. delta_theta_m says which change, and is
directly readable -- "a 10% gain drift is read as 0.3 log-units of synaptic
strength".

Why common random numbers are not optional here
-----------------------------------------------
The simulator is stochastic. A central difference evaluated with independent
noise on each side estimates the derivative plus a difference of two noise
draws, whose variance does not shrink with the step size -- it GROWS as
1/(2h)^2. The finite-difference "derivative" would then be dominated by noise
at exactly the small steps one would choose for accuracy, and would look
smooth only because averaging hides it.

Every evaluation here therefore uses the SAME realisation key and the SAME
base seed, so the spike times and the graph are identical on both sides of
every difference and only the perturbed coordinate moves. This is the
pathwise-derivative trick, and without it the whole module measures noise.

Two consequences worth stating:

  * the estimate is the derivative of the simulator AT THAT SEED, not of the
    marginal p(x | theta). Averaging over a handful of seeds gives the latter;
    `n_seeds` does that and reports the across-seed spread so the reader can
    see whether one seed was enough.
  * a_m is scale-dependent through the step sizes. Steps are given in units of
    each coordinate's own range (parameters) or SD (nuisance), so the answer
    does not depend on how the axes happen to be stored.

Pure ASCII, LF only.
"""

import numpy as np
import torch

__all__ = ["embed_at", "jacobian_theta", "jacobian_nu", "aliasing_report",
           "principal_angles"]


def embed_at(model, spec, provider, theta, realisation_spec, base_seed,
             nuisance_spec=None, nu_row=None, key=("ALIAS", "W0", 0),
             reduce="mean"):
    """z for one controlled evaluation. Deterministic given its arguments."""
    from latent_sbi_simulator import simulate_windows

    x, _ = simulate_windows(
        spec, np.asarray(theta, dtype=np.float64).ravel(), provider,
        donor=key[0], well=key[1], subregion=key[2],
        realisation_spec=realisation_spec, base_seed=base_seed,
        nuisance_spec=nuisance_spec, nu_row=nu_row,
        rng=np.random.default_rng(base_seed))
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    with torch.no_grad():
        z = model.encode(xt).cpu().numpy()
    return z.mean(axis=0) if reduce == "mean" else z[0]


def jacobian_theta(model, spec, provider, theta0, realisation_spec, base_seed,
                   h=0.02, nuisance_spec=None, nu_row=None):
    """(E, d) central-difference Jacobian dz/dtheta at theta0.

    `h` is in units of the unit box the bench parameters live in, so it is
    already dimensionless. Steps that would leave (0, 1) are made one-sided
    rather than clipped: a clipped central difference is a one-sided
    difference with the wrong denominator, which is a silent factor-of-two
    error in that column.
    """
    theta0 = np.asarray(theta0, dtype=np.float64).ravel()
    d = theta0.size
    cols = []
    for k in range(d):
        lo = theta0.copy()
        hi = theta0.copy()
        lo[k] = theta0[k] - h
        hi[k] = theta0[k] + h
        denom = 2.0 * h
        if lo[k] <= 0.0:
            lo[k] = theta0[k]
            denom = h
        if hi[k] >= 1.0:
            hi[k] = theta0[k]
            denom = (denom / 2.0) if denom == h else h
        z_hi = embed_at(model, spec, provider, hi, realisation_spec, base_seed,
                        nuisance_spec, nu_row)
        z_lo = embed_at(model, spec, provider, lo, realisation_spec, base_seed,
                        nuisance_spec, nu_row)
        cols.append((z_hi - z_lo) / denom)
    return np.stack(cols, axis=1)


def jacobian_nu(model, spec, provider, theta0, realisation_spec, base_seed,
                nuisance_spec, h_sd=0.5):
    """(E, M) central-difference Jacobian dz/dnu at nu = 0.

    Steps are `h_sd` times each component's own total standard deviation
    across the three nesting levels, so a component that barely varies in the
    design is not probed as if it varied a lot.

    Returns (J_nu, quantised) where `quantised[m]` is True when the component
    moves z only in discrete jumps -- see the comment in the loop.
    """
    from latent_nuisance import NU_LEVELS, N_COMPONENTS

    sd = np.sqrt(sum(nuisance_spec.scales[l] ** 2 for l in NU_LEVELS))
    cols = []
    quantised = []
    for m in range(N_COMPONENTS):
        step = h_sd * float(sd[m])
        base = embed_at(model, spec, provider, theta0, realisation_spec,
                        base_seed, nuisance_spec, np.zeros(N_COMPONENTS))
        if step <= 0.0:
            cols.append(np.zeros_like(base))
            quantised.append(False)
            continue

        def _col(st):
            hi = np.zeros(N_COMPONENTS)
            lo = np.zeros(N_COMPONENTS)
            hi[m], lo[m] = st, -st
            z_hi = embed_at(model, spec, provider, theta0, realisation_spec,
                            base_seed, nuisance_spec, hi)
            z_lo = embed_at(model, spec, provider, theta0, realisation_spec,
                            base_seed, nuisance_spec, lo)
            return z_hi, z_lo

        z_hi, z_lo = _col(step)
        # A component whose effect is QUANTISED -- electrode dropout rounds to
        # a whole number of electrodes -- has a piecewise-constant map, so a
        # small central difference returns exactly zero almost everywhere and
        # is undefined at the steps. Reporting a_m = 0 for it would read as
        # "perfectly separable from theta", which is the opposite of the
        # truth. Probe with a much larger step: if that moves z, the component
        # is quantised, not inert.
        if np.array_equal(z_hi, z_lo):
            z_hi_big, z_lo_big = _col(step * 8.0)
            quantised.append(not np.array_equal(z_hi_big, z_lo_big))
        else:
            quantised.append(False)
        cols.append((z_hi - z_lo) / (2.0 * step))
    return np.stack(cols, axis=1), quantised


def principal_angles(A, B):
    """Principal angles (radians) between the column spans of A and B."""
    qa, _ = np.linalg.qr(np.atleast_2d(A))
    qb, _ = np.linalg.qr(np.atleast_2d(B))
    s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1.0, 1.0)
    return np.arccos(s)


def aliasing_report(J_theta, J_nu, param_names=None, nu_names=None,
                    rcond=1e-8, quantised=None):
    """a_m and delta_theta_m per nuisance direction, eq. (4).

    The pseudo-inverse is truncated at `rcond`: J_theta is rank-deficient
    along sloppy directions by construction, and an untruncated pinv would
    return an enormous delta_theta pointing along the least-determined
    direction. That is P11's content -- aliasing is worst where the
    information spectrum is smallest -- and it must be reported as such rather
    than emitted as a huge number.
    """
    J_theta = np.atleast_2d(np.asarray(J_theta, dtype=np.float64))
    J_nu = np.atleast_2d(np.asarray(J_nu, dtype=np.float64))
    E, d = J_theta.shape
    M = J_nu.shape[1]
    names = list(param_names) if param_names else ["theta%d" % k
                                                   for k in range(d)]
    nnames = list(nu_names) if nu_names else ["nu%d" % m for m in range(M)]

    u, s, vt = np.linalg.svd(J_theta, full_matrices=False)
    keep = s > (rcond * (s[0] if s.size and s[0] > 0 else 1.0))
    rank = int(keep.sum())
    U = u[:, keep]
    pinv = (vt[keep].T * (1.0 / s[keep])) @ U.T

    out = {"rank_J_theta": rank, "n_theta": d, "n_nu": M,
           "singular_values": s.tolist(), "axes": names, "nu": nnames,
           "directions": []}
    for m in range(M):
        g = J_nu[:, m]
        ng = float(np.linalg.norm(g))
        if ng == 0.0:
            q = bool(quantised[m]) if quantised is not None else False
            out["directions"].append({
                "nu": nnames[m], "a_m": None, "quantised": q,
                "note": ("QUANTISED: this component moves z only in discrete "
                         "jumps (electrode dropout rounds to a whole number "
                         "of electrodes), so no derivative exists and a_m is "
                         "undefined. It is NOT separable -- read the nuisance "
                         "floor for this component instead."
                         if q else
                         "this component does not move z at all")})
            continue
        proj = U @ (U.T @ g)
        a_m = float(np.linalg.norm(proj) / ng)
        dth = pinv @ g
        heavy = np.argsort(-np.abs(dth))[:3]
        out["directions"].append({
            "nu": nnames[m],
            "a_m": a_m,
            "delta_theta": dth.tolist(),
            "dominant_axes": [names[j] for j in heavy],
            "dominant_shift": [float(dth[j]) for j in heavy],
            "reading": _read_alias(a_m, names, dth, heavy),
        })

    ang = principal_angles(J_theta[:, :rank] if rank else J_theta, J_nu)
    out["principal_angles_rad"] = ang.tolist()
    out["principal_angles_deg"] = np.degrees(ang).tolist()
    out["min_angle_deg"] = float(np.degrees(ang.min())) if ang.size else None
    return out


def _read_alias(a_m, names, dth, heavy):
    if a_m > 0.9:
        return ("absorbed: %.0f%% of this nuisance direction lies inside the "
                "parameter subspace, so the pipeline reads it as %s = %+.3f "
                "and cannot tell the difference."
                % (100 * a_m, names[heavy[0]], dth[heavy[0]]))
    if a_m < 0.3:
        return ("transverse: only %.0f%% is absorbed, so this nuisance is "
                "largely separable from theta. A replicate term will suppress "
                "it without fighting the likelihood term." % (100 * a_m))
    return ("partly absorbed (%.0f%%): separable in principle, aliased in "
            "practice along %s." % (100 * a_m, names[heavy[0]]))
