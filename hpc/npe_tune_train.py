#!/usr/bin/env python3
"""
npe_tune_train.py -- independent member training, persistence, extension.

Scope boundary: this module trains members and evaluates their densities. It
does not choose configurations, does not gate, and does not decide splits.

Design decisions locked here, with the reason for each:

  * MEMBERS ARE TRAINED INDEPENDENTLY, one optimisation per member, exactly
    as npe_model.train_ensemble does. The mixture is NEVER trained. Joint
    mixture training would let members co-adapt and specialise, so their
    disagreement would become a fitted partition of the data rather than an
    estimate of epistemic uncertainty, and the extra width an ensemble
    supplies would be a property of the fit. Independence is what makes the
    spread honest.

  * BECAUSE MEMBERS ARE INDEPENDENT, q_m DOES NOT DEPEND ON M. An ensemble
    trained at M_probe extends to M by training only the missing seeds. That
    is what makes the probe-then-promote design cheap rather than wasteful:
    probe members are not thrown away, they are the first members of the
    finalist ensemble. extend_ensemble() implements exactly this and asserts
    that the reused members were trained under the same configuration.

  * THE SEED LIST IS AN ORDERED TUPLE SUPPLIED BY THE USER, not derived from
    a base seed. Member m is defined by seed s_m, so "the first M_probe
    members" is well defined and reproducible across jobs and machines.

  * TRAINING PROVENANCE IS CAPTURED. sbi's own training summary (best
    validation log-probability, epochs trained, whether the epoch ceiling
    was hit) is extracted and recorded. This is the project's open point O3:
    save_ensemble persists the config and contract but not convergence, so
    "converged" has been an assumption rather than a record. The extractor
    probes several attribute spellings because the exact one is a property
    of the installed sbi version; when it finds nothing it records that fact
    rather than a plausible-looking blank.

  * _train_single_with_summary MIRRORS npe_model.train_single deliberately.
    It exists only because train_single returns the posterior and discards
    the inference object that carries the summary. The parts that encode
    scientific choices -- the flow family, the parameter standardisation,
    the prior reaching the builder as x_dist -- are NOT duplicated: they are
    obtained from npe_model.build_estimator_builder and npe_model.NPEConfig.
    If train_single ever grows a return-the-inference flag, delete this
    function and call it instead.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MemberRecord",
    "make_config",
    "train_members",
    "extend_ensemble",
    "save_members",
    "load_members",
    "build_ensemble",
    "evaluate_log_probs",
    "library_versions",
]


@dataclass
class MemberRecord:
    """What one member training produced, beyond the network itself."""

    seed: int
    epochs: int = -1
    best_val_log_prob: float = float("nan")
    hit_max_epochs: bool = False
    train_seconds: float = float("nan")
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": int(self.seed), "epochs": int(self.epochs),
                "best_val_log_prob": float(self.best_val_log_prob),
                "hit_max_epochs": bool(self.hit_max_epochs),
                "train_seconds": float(self.train_seconds),
                "notes": self.notes}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def make_config(config: Dict[str, Any], device: str = "cpu",
                **overrides: Any):
    """Build an npe_model.NPEConfig from a plain dict of searched knobs.

    Unknown keys are refused rather than ignored: a typo in a knob name
    would otherwise silently train the default configuration and report it
    under the proposed one, which is the worst kind of failure -- a wrong
    number that looks right.
    """
    import npe_model

    fields = set(npe_model.NPEConfig.__dataclass_fields__.keys())
    merged: Dict[str, Any] = dict(config)
    merged.update(overrides)
    merged["device"] = device
    unknown = sorted(set(merged) - fields)
    if unknown:
        raise KeyError(
            "unknown NPEConfig field(s) %r. Known fields: %s"
            % (unknown, sorted(fields)))
    return npe_model.NPEConfig(**merged)


def library_versions() -> Dict[str, str]:
    """Record what actually ran. A score is only comparable to another score
    produced by the same stack."""
    out: Dict[str, str] = {}
    for name in ("numpy", "torch", "sbi", "zuko", "skopt", "scipy", "sklearn"):
        try:
            mod = __import__(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[name] = "absent"
    return out


# ---------------------------------------------------------------------------
# Convergence extraction (open point O3)
# ---------------------------------------------------------------------------

def _extract_summary(inference, max_num_epochs: int) -> Tuple[int, float, bool, str]:
    """Pull (epochs, best validation log-prob, hit-ceiling, notes) from sbi.

    The attribute carrying the summary has moved between sbi releases, so
    several spellings are probed and the one that was found is reported. A
    missing summary yields sentinels plus a note, never a fabricated number.
    """
    notes: List[str] = []
    epochs, best, found = -1, float("nan"), []

    summary = None
    for attr in ("summary", "_summary"):
        if hasattr(inference, attr):
            cand = getattr(inference, attr)
            if isinstance(cand, dict):
                summary = cand
                found.append(attr)
                break

    if summary is not None:
        for key in ("epochs_trained", "epochs", "epoch"):
            if key in summary:
                try:
                    v = summary[key]
                    epochs = int(v[-1] if isinstance(v, (list, tuple)) else v)
                    found.append(key)
                    break
                except Exception:
                    pass
        for key in ("best_validation_log_prob", "best_validation_log_probs",
                    "best_validation_loss", "validation_log_probs",
                    "validation_loss"):
            if key in summary:
                try:
                    v = summary[key]
                    val = v[-1] if isinstance(v, (list, tuple)) else v
                    best = float(val)
                    if "loss" in key:
                        best = -best      # store as a log-probability
                        notes.append("converted %s (a loss) to a log-prob"
                                     % key)
                    found.append(key)
                    break
                except Exception:
                    pass

    # Fall back to attributes carried directly on the inference object.
    if epochs < 0 and hasattr(inference, "epoch"):
        try:
            epochs = int(getattr(inference, "epoch"))
            found.append("inference.epoch")
        except Exception:
            pass
    if not np.isfinite(best) and hasattr(inference, "_val_log_prob"):
        try:
            best = float(getattr(inference, "_val_log_prob"))
            found.append("inference._val_log_prob")
        except Exception:
            pass

    if epochs < 0 and not np.isfinite(best):
        notes.append("no training summary found on the inference object; "
                     "convergence is UNRECORDED for this member (the sbi "
                     "attribute name should be confirmed on this cluster)")
    else:
        notes.append("summary via " + ",".join(found))

    hit = bool(epochs >= int(max_num_epochs) > 0)
    if hit:
        notes.append("hit max_num_epochs: early stopping never fired, so "
                     "this member is NOT converged by the configured rule")
    return epochs, best, hit, "; ".join(notes)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_single_with_summary(z: np.ndarray,
                               theta: np.ndarray,
                               prior,
                               config,
                               seed: int):
    """One member. Mirrors npe_model.train_single, keeping the summary.

    See the module docstring for why this mirror exists and what it does NOT
    duplicate.
    """
    import torch
    import npe_model
    from sbi.inference import NPE

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    inference = NPE(
        prior=prior,
        density_estimator=npe_model.build_estimator_builder(config, prior=prior),
        device=config.device,
        show_progress_bars=config.show_progress_bars,
    )
    theta_t = torch.as_tensor(np.asarray(theta), dtype=torch.float32,
                              device=config.device)
    z_t = torch.as_tensor(np.asarray(z), dtype=torch.float32,
                          device=config.device)
    inference.append_simulations(theta_t, z_t)
    inference.train(
        training_batch_size=config.training_batch_size,
        learning_rate=config.learning_rate,
        validation_fraction=config.validation_fraction,
        stop_after_epochs=config.stop_after_epochs,
        max_num_epochs=config.max_num_epochs,
        clip_max_norm=config.clip_max_norm,
        show_train_summary=False,
    )
    posterior = inference.build_posterior()
    return posterior, inference


def train_members(z: np.ndarray,
                  theta: np.ndarray,
                  prior,
                  config,
                  seeds: Sequence[int],
                  verbose: bool = True) -> Tuple[List[Any], List[MemberRecord]]:
    """Train one member per seed, independently. Returns (posteriors, records).

    There is no coupling between iterations: each member sees the same rows
    and differs only through its seed, which controls initialisation, batch
    order and sbi's internal validation split.
    """
    posteriors: List[Any] = []
    records: List[MemberRecord] = []
    for j, s in enumerate(seeds):
        if verbose:
            print("[train] member %d/%d (seed %d)" % (j + 1, len(seeds), s),
                  flush=True)
        t0 = time.time()
        post, inference = _train_single_with_summary(z, theta, prior, config,
                                                     seed=int(s))
        dt = time.time() - t0
        ep, best, hit, notes = _extract_summary(inference,
                                                config.max_num_epochs)
        posteriors.append(post)
        records.append(MemberRecord(seed=int(s), epochs=ep,
                                    best_val_log_prob=best, hit_max_epochs=hit,
                                    train_seconds=float(dt), notes=notes))
        if verbose:
            print("        epochs=%s best_val_log_prob=%s  %.1fs"
                  % (ep if ep >= 0 else "?",
                     ("%.4f" % best) if np.isfinite(best) else "?", dt),
                  flush=True)
    return posteriors, records


# ---------------------------------------------------------------------------
# Persistence, in a layout npe_model.load_ensemble can also read
# ---------------------------------------------------------------------------

def save_members(posteriors: Sequence[Any],
                 out_dir: str,
                 seeds: Sequence[int],
                 config,
                 records: Sequence[MemberRecord],
                 contract=None,
                 extra: Optional[Dict[str, Any]] = None) -> str:
    """Persist members as member_%02d.pt plus an ensemble.json.

    The filename pattern and the n_members key match npe_model.load_ensemble
    exactly, so an ensemble written here is loadable by the existing code
    path. The seeds and the per-member convergence block are additions, not
    replacements.
    """
    import torch
    from dataclasses import asdict

    os.makedirs(out_dir, exist_ok=True)
    for j, post in enumerate(posteriors):
        torch.save(post, os.path.join(out_dir, "member_%02d.pt" % j))
    meta: Dict[str, Any] = {
        "n_members": len(posteriors),
        "seeds": [int(s) for s in seeds],
        "config": asdict(config),
        "members": [r.to_dict() for r in records],
    }
    if contract is not None:
        meta["contract"] = contract.to_dict()
    if extra:
        meta.update(extra)
    with open(os.path.join(out_dir, "ensemble.json"), "w",
              encoding="ascii") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    return out_dir


def load_members(out_dir: str) -> Tuple[List[Any], Dict[str, Any]]:
    """Reload members and their metadata. Returns (posteriors, meta)."""
    import torch

    with open(os.path.join(out_dir, "ensemble.json"), "r",
              encoding="ascii") as fh:
        meta = json.load(fh)
    posteriors = []
    for j in range(int(meta["n_members"])):
        posteriors.append(
            torch.load(os.path.join(out_dir, "member_%02d.pt" % j),
                       weights_only=False))
    return posteriors, meta


def build_ensemble(posteriors: Sequence[Any]):
    """Wrap members in sbi's EnsemblePosterior (the arithmetic mixture)."""
    from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior

    return EnsemblePosterior(list(posteriors))


def extend_ensemble(model_dir: str,
                    z: np.ndarray,
                    theta: np.ndarray,
                    prior,
                    config,
                    seeds: Sequence[int],
                    contract=None,
                    verbose: bool = True) -> Tuple[List[Any], List[MemberRecord]]:
    """Grow a saved ensemble to the full seed list, training only what is missing.

    This is the operational payoff of member independence: q_m does not
    depend on M, so promoting a probe ensemble to the deliverable size costs
    only the members that do not exist yet.

    Refuses to reuse members trained under a DIFFERENT configuration, since
    a mixture of members from two configurations is neither configuration
    and would be reported as one of them.
    """
    from dataclasses import asdict

    want = [int(s) for s in seeds]
    existing: List[Any] = []
    have: List[int] = []
    recs: List[MemberRecord] = []

    if os.path.exists(os.path.join(model_dir, "ensemble.json")):
        existing, meta = load_members(model_dir)
        have = [int(s) for s in meta.get("seeds", [])]
        old_cfg = meta.get("config", {})
        new_cfg = asdict(config)
        keys = set(old_cfg) | set(new_cfg)
        differ = [k for k in sorted(keys)
                  if k not in ("device", "show_progress_bars")
                  and old_cfg.get(k) != new_cfg.get(k)]
        if differ:
            raise ValueError(
                "refusing to extend %r: the saved members were trained under "
                "a different configuration (%s differ). Train a fresh "
                "ensemble instead."
                % (model_dir, ", ".join(differ)))
        for r in meta.get("members", []):
            recs.append(MemberRecord(
                seed=int(r.get("seed", -1)), epochs=int(r.get("epochs", -1)),
                best_val_log_prob=float(r.get("best_val_log_prob", float("nan"))),
                hit_max_epochs=bool(r.get("hit_max_epochs", False)),
                train_seconds=float(r.get("train_seconds", float("nan"))),
                notes=str(r.get("notes", "")) + " [reused]"))

    if have[:len(have)] != want[:len(have)]:
        raise ValueError(
            "refusing to extend %r: saved seeds %r are not the prefix of the "
            "requested seed list %r, so 'the first members' is ambiguous."
            % (model_dir, have, want))

    missing = [s for s in want if s not in have]
    if verbose:
        print("[extend] %d member(s) reused, %d to train"
              % (len(have), len(missing)), flush=True)
    if missing:
        new_posts, new_recs = train_members(z, theta, prior, config, missing,
                                            verbose=verbose)
        existing = list(existing) + list(new_posts)
        recs = list(recs) + list(new_recs)

    save_members(existing, model_dir, want, config, recs, contract=contract)
    return existing, recs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_log_probs(posteriors: Sequence[Any],
                       theta: np.ndarray,
                       z: np.ndarray,
                       ensemble=None) -> Tuple[np.ndarray, np.ndarray]:
    """log-densities of the true theta under each member and the mixture.

    Returns (member_log_probs (M, N), mixture_log_prob (N,)).

    The mixture value comes from the ensemble's own log_prob when an
    ensemble object is supplied, and from the per-member logs otherwise. The
    smoke test asserts the two agree rather than trusting either: that
    equality is what confirms the mixture is arithmetic (as required) and
    not a geometric mean, which would be sharper than any member and make
    overconfidence worse.
    """
    import npe_diagnostics as D
    import npe_tune_score as S

    per_member = []
    for post in posteriors:
        lp = D.posterior_log_probs(post, np.asarray(theta), np.asarray(z))
        per_member.append(np.asarray(lp, dtype=np.float64).reshape(-1))
    member_log_probs = np.vstack(per_member)

    if ensemble is not None:
        mix = np.asarray(
            D.posterior_log_probs(ensemble, np.asarray(theta), np.asarray(z)),
            dtype=np.float64).reshape(-1)
    else:
        mix = S.mixture_log_prob(member_log_probs)
    return member_log_probs, mix
