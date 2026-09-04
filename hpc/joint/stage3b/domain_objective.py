"""Decomposing A0's deficit into a DOMAIN effect and an OBJECTIVE effect.

Plan v0.6, Stage 3b, eq. (9). A0 differs from A1 in two ways at once -- the
objective its encoder is fitted with, and the domain that encoder is fitted on
-- so `L(A0) - L(A1)` is one number answering two questions. A0s (same
objective, no domain shift) separates them:

    L(A0) - L(A1)  =  [L(A0) - L(A0s)]  +  [L(A0s) - L(A1)]              (9)
    -----------       ----------------      -----------------
       total            domain effect        objective effect

and the joint-side analogue is `L(A2) - L(A2s)`.

Why the decomposition is worth its own module
---------------------------------------------
The two terms imply different fixes and the plan says so: if the DOMAIN term
dominates, the answer is to fit the encoder on simulated windows and the
phenotype weight is second-order; if the OBJECTIVE term dominates,
lambda_dsn and the Stage 4 search are the whole answer. Reading a single
A0 - A1 number gives no way to choose, and the recorded
r_eff(sim) = 1.017 under an encoder that never saw a simulated window in
training is consistent with either (and with a third possibility, O2).

Every score here is the PRIMARY endpoint (pseudo-real held-out NLL, D10) when
it is present, falling back to the simulated split with a loud note.

Deviation from the plan, stated rather than buried
--------------------------------------------------
Plan Stage 3b asks for "the same split per axis". A per-AXIS NLL requires the
flow's marginals, which a normalizing flow does not give in closed form (the
same reason `joint_diagnostics` reports contraction rather than per-axis
gain). The per-axis decomposition here is therefore computed on CONTRACTION,
not on NLL. It answers "which axes did the domain shift stop the encoder from
constraining" rather than "how many nats per axis", and it is labelled as such
in the output. The aggregate decomposition is on the endpoint itself and is
exact.

Pure ASCII, LF only.
"""

import json
import os

import numpy as np

__all__ = ["primary_score", "per_row_primary", "decompose", "sigma_seed",
           "decomposition_table", "REQUIRED_ARMS"]

# The four arms eq. (9) needs. A2/A2s are the joint-side analogue and are
# optional; without A0s the decomposition is not defined at all.
REQUIRED_ARMS = ("A0", "A0s", "A1")


def primary_score(rec):
    """(L, endpoint_name) for one run, preferring the pseudo-real endpoint."""
    v = rec.get("L_pseudo_real")
    if v is None:
        return float(rec["L"]), "simulated"
    return float(v), "pseudo-real"


def per_row_primary(rec):
    """(nll, group, endpoint_name) arrays for one run.

    `rec` must already carry the `_nll` / `_group` arrays loaded from the
    run's `_perrow.npz`; `report_joint_arms.load_runs` does that and prefers
    the pseudo-real arrays when present.
    """
    if "_nll" not in rec:
        raise KeyError("run %s seed %s has no per-row NLL; rerun with the "
                       "current runner" % (rec.get("arm"), rec.get("seed")))
    return rec["_nll"], rec["_group"], rec.get("_endpoint", "unknown")


def sigma_seed(runs_for_arm):
    """Across-seed SD of the primary score. NaN with fewer than two seeds."""
    vals = [primary_score(r)[0] for r in runs_for_arm]
    if len(vals) < 2:
        return float("nan")
    return float(np.std(np.asarray(vals, dtype=np.float64), ddof=1))


def decompose(by_arm_seed, seed, bp=None):
    """The three terms of eq. (9) for ONE seed, paired at the row level.

    Parameters
    ----------
    by_arm_seed : dict arm -> dict seed -> run record (with _nll/_group)
    seed : int
    bp : the bootstrap_paired module, or None

    Returns a dict with the three D values, their intervals when available,
    and an `identity_residual` that MUST be zero: the decomposition is an
    algebraic identity, so a non-zero residual means the three arms were not
    scored on the same rows and nothing else in the record can be trusted.
    """
    need = {}
    for arm in REQUIRED_ARMS:
        r = by_arm_seed.get(arm, {}).get(seed)
        if r is None:
            return {"seed": seed, "error": "arm %s missing at seed %d"
                    % (arm, seed)}
        need[arm] = r

    hashes = {arm: r["split_hash"] for arm, r in need.items()}
    if len(set(hashes.values())) != 1:
        return {"seed": seed,
                "error": "split_hash differs across arms (%s); the arms were "
                         "scored on different rows, so eq. (9) does not hold"
                         % hashes}

    rows = {arm: per_row_primary(r) for arm, r in need.items()}
    shapes = {arm: v[0].shape for arm, v in rows.items()}
    if len(set(shapes.values())) != 1:
        return {"seed": seed, "error": "row counts differ: %s" % shapes}
    endpoints = {v[2] for v in rows.values()}
    if len(endpoints) != 1:
        return {"seed": seed, "error": "arms scored on different endpoints: %s"
                % endpoints}

    nll = {arm: v[0] for arm, v in rows.items()}
    group = rows["A1"][1]

    d_total = nll["A0"] - nll["A1"]
    d_domain = nll["A0"] - nll["A0s"]
    d_objective = nll["A0s"] - nll["A1"]

    out = {
        "seed": seed,
        "endpoint": endpoints.pop(),
        "split_hash": hashes["A1"],
        "n_rows": int(d_total.size),
        "D_total": float(np.mean(d_total)),
        "D_domain": float(np.mean(d_domain)),
        "D_objective": float(np.mean(d_objective)),
    }
    out["identity_residual"] = float(
        out["D_total"] - out["D_domain"] - out["D_objective"])

    if bp is not None:
        for name, d in (("total", d_total), ("domain", d_domain),
                        ("objective", d_objective)):
            try:
                res = bp.paired_bootstrap(d, groups=group, within="all")
                ci = (res.get("ci") if isinstance(res, dict)
                      else getattr(res, "ci", None))
                out["CI_" + name] = (list(ci) if ci is not None
                                     and len(ci) == 2 else None)
            except Exception as exc:                   # noqa: BLE001
                out["CI_" + name] = None
                out.setdefault("bootstrap_errors", []).append(
                    "%s: %s: %s" % (name, type(exc).__name__, exc))
    return out


def joint_side(by_arm_seed, seed):
    """The joint-side split L(A2) - L(A2s). Optional; None when absent."""
    a2 = by_arm_seed.get("A2", {}).get(seed)
    a2s = by_arm_seed.get("A2s", {}).get(seed)
    if a2 is None or a2s is None:
        return None
    if a2["split_hash"] != a2s["split_hash"]:
        return {"seed": seed, "error": "A2/A2s split_hash differ"}
    n2, _, e2 = per_row_primary(a2)
    n2s, _, e2s = per_row_primary(a2s)
    if n2.shape != n2s.shape or e2 != e2s:
        return {"seed": seed, "error": "A2/A2s not comparable"}
    return {"seed": seed, "D_joint_domain": float(np.mean(n2 - n2s)),
            "endpoint": e2}


def per_axis_contraction_split(by_arm_seed, seed, param_names=None):
    """The eq. (9) split applied to per-axis CONTRACTION, not NLL.

    See the module docstring: this is a deviation from the plan's wording and
    answers a related but different question. Returned separately from the
    aggregate so the two cannot be confused in a table.
    """
    need = {}
    for arm in REQUIRED_ARMS:
        r = by_arm_seed.get(arm, {}).get(seed)
        if r is None or "contraction" not in r:
            return None
        need[arm] = np.asarray(r["contraction"], dtype=np.float64)
    d = len(need["A1"])
    names = list(param_names) if param_names else ["axis%d" % k for k in range(d)]
    return {
        "quantity": "contraction (NOT nats; see module docstring)",
        "axes": names,
        "total": (need["A0"] - need["A1"]).tolist(),
        "domain": (need["A0"] - need["A0s"]).tolist(),
        "objective": (need["A0s"] - need["A1"]).tolist(),
    }


def decomposition_table(results, sig):
    """Markdown for the aggregate decomposition, one row per seed."""
    lines = ["| seed | endpoint | D_total | D_domain | D_objective | "
             "dominant | residual |", "|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append("| %s | - | - | - | - | ERROR | %s |"
                         % (r.get("seed"), r["error"]))
            continue
        dom = ("domain" if abs(r["D_domain"]) > abs(r["D_objective"])
               else "objective")
        lines.append("| %d | %s | %.4f | %.4f | %.4f | %s | %.2e |"
                     % (r["seed"], r["endpoint"], r["D_total"], r["D_domain"],
                        r["D_objective"], dom, r["identity_residual"]))
    good = [r for r in results if "error" not in r]
    if len(good) > 1:
        means = {k: float(np.mean([r[k] for r in good]))
                 for k in ("D_total", "D_domain", "D_objective")}
        lines.append("| **mean** | %s | %.4f | %.4f | %.4f | %s | |"
                     % (good[0]["endpoint"], means["D_total"],
                        means["D_domain"], means["D_objective"],
                        "domain" if abs(means["D_domain"])
                        > abs(means["D_objective"]) else "objective"))
    return "\n".join(lines)


def verdict(results, sig):
    """The S2.4 two-clause reading, applied to each term of eq. (9)."""
    good = [r for r in results if "error" not in r]
    if not good:
        return ["No seed produced a valid decomposition."]
    means = {k: float(np.mean([r[k] for r in good]))
             for k in ("D_total", "D_domain", "D_objective")}
    s = max([v for v in sig.values() if np.isfinite(v)] or [float("nan")])
    out = []
    for name, key in (("total", "D_total"), ("domain", "D_domain"),
                      ("objective", "D_objective")):
        D = means[key]
        ci = [r.get("CI_" + name) for r in good]
        have_ci = any(c is not None for c in ci)
        clause_seed = np.isfinite(s) and abs(D) > 2 * s
        if not have_ci:
            out.append("- %s effect: D = %.4f nats/row. sigma_seed clause %s "
                       "(2*sigma_seed = %s). No interval -- **no call**."
                       % (name, D, "met" if clause_seed else "NOT met",
                          "%.4f" % (2 * s) if np.isfinite(s) else "undefined"))
        else:
            out.append("- %s effect: D = %.4f nats/row, sigma_seed clause %s, "
                       "intervals present -- read both clauses together."
                       % (name, D, "met" if clause_seed else "NOT met"))
    out.append("- The larger of |D_domain| and |D_objective| decides which "
               "fix applies: domain -> fit the encoder on simulated windows; "
               "objective -> lambda_dsn and the Stage 4 search.")
    return out
