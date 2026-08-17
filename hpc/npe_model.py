#!/usr/bin/env python3
"""
npe_model.py -- density estimator construction, training, and ensembling.

Scope boundary: this module builds and trains estimators and returns
posterior objects. It does NOT load files (see npe_contract.py), does not
compute diagnostics, and does not plot. Swapping the flow family, the
ensemble size, or the training schedule must not require touching anything
else.

Design decisions locked here, with the reason for each:

  * Flow family: zuko-backed neural spline flow ("zuko_nsf"). Rational
    quadratic splines are the standard choice for NPE with a moderate
    parameter dimension and a low-dimensional conditioner.

  * Parameter standardisation: "transform_to_unconstrained". With a box
    prior this maps the bounded support onto all of R^p, so the flow cannot
    place posterior mass outside the physical bounds. Plain z-scoring leaves
    the tails free to leak out of the box, which then has to be rejected at
    sampling time.

  * Conditioner standardisation: "independent" z-scoring of each embedding
    component. The embedding lies on the unit sphere, so its components are
    dependent by construction; per-component standardisation is still the
    right normalisation and does not attempt to undo the constraint.

  * Ensembling: n independently initialised and independently trained
    estimators, combined as the ARITHMETIC mixture

        q_bar(theta | z) = (1/n) sum_j q_j(theta | z)                  (E1)

    evaluated in log space as logsumexp_j(log q_j) - log n. This is what
    inflates credible regions and raises coverage. The geometric mean,
    (1/n) sum_j log q_j, is a product of experts: it is SHARPER than any
    member and would make overconfidence worse, not better. It also admits
    no closed-form sampler. sbi's EnsemblePosterior implements (E1); the
    smoke test asserts this numerically rather than trusting it.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "NPEConfig",
    "build_estimator_builder",
    "train_single",
    "train_ensemble",
    "save_ensemble",
    "load_ensemble",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NPEConfig:
    """Hyperparameters for one estimator and for the training schedule.

    Defaults are deliberately modest. They are a starting point to be tuned
    against held-out log-probability, not a tuned configuration: no tuning
    has been done against the real data.
    """

    # -- flow architecture --
    model: str = "zuko_nsf"
    hidden_features: int = 128
    num_transforms: int = 8
    num_bins: int = 10

    # -- standardisation --
    z_score_theta: str = "transform_to_unconstrained"
    z_score_x: str = "independent"

    # -- optimisation --
    learning_rate: float = 5e-4
    training_batch_size: int = 512
    validation_fraction: float = 0.1
    stop_after_epochs: int = 20
    max_num_epochs: int = 500
    clip_max_norm: float = 5.0

    # -- runtime --
    device: str = "cpu"
    show_progress_bars: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Estimator construction
# ---------------------------------------------------------------------------

def build_estimator_builder(config: NPEConfig, prior=None):
    """Return an sbi density-estimator builder for the configured flow.

    NAMING TRAP, worth stating explicitly because it is easy to get
    backwards. Inside sbi's low-level flow builders the ESTIMATED variable
    is called `x` and the CONDITIONER is called `y`. For NPE the estimated
    variable is theta, so the argument that carries the prior over theta is
    named `x_dist`, not `theta_dist`. Passing the prior anywhere else
    silently leaves the unconstrained transform unconfigured.

    `x_dist` is required whenever z_score_theta is
    "transform_to_unconstrained": the transform needs the support of the
    prior to know which bijection to build.
    """
    from sbi.neural_nets import posterior_nn

    kwargs = {}
    if config.z_score_theta == "transform_to_unconstrained":
        if prior is None:
            raise ValueError(
                "z_score_theta='transform_to_unconstrained' requires the prior; "
                "pass prior= to build_estimator_builder.")
        kwargs["x_dist"] = prior

    return posterior_nn(
        model=config.model,
        z_score_theta=config.z_score_theta,
        z_score_x=config.z_score_x,
        hidden_features=config.hidden_features,
        num_transforms=config.num_transforms,
        num_bins=config.num_bins,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _as_tensors(z: np.ndarray, theta: np.ndarray, device: str):
    import torch

    theta_t = torch.as_tensor(np.asarray(theta), dtype=torch.float32, device=device)
    z_t = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=device)
    return theta_t, z_t


def train_single(
    z: np.ndarray,
    theta: np.ndarray,
    prior,
    config: Optional[NPEConfig] = None,
    seed: int = 0,
    extra_batches: Optional[Sequence] = None,
):
    """Train one NPE estimator and return its DirectPosterior.

    Parameters
    ----------
    z : ndarray, shape (n, E)
        Conditioner. The DSN embedding; the network's `x`.
    theta : ndarray, shape (n, p)
        Training label, in INFERENCE coordinates.
    prior : torch Distribution
        The box prior. Required so that the unconstrained transform and the
        posterior's support are both defined.
    seed : int
        Controls initialisation and the train/validation split, so ensemble
        members differ.
    extra_batches : sequence of (z, theta) pairs, optional
        Additional shards appended before training. Provided so that
        incremental delivery of simulation batches does not require
        reloading everything into one array upstream.

    Returns
    -------
    DirectPosterior
    """
    import torch
    from sbi.inference import NPE

    config = config or NPEConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)

    inference = NPE(
        prior=prior,
        density_estimator=build_estimator_builder(config, prior=prior),
        device=config.device,
        show_progress_bars=config.show_progress_bars,
    )

    theta_t, z_t = _as_tensors(z, theta, config.device)
    inference.append_simulations(theta_t, z_t)
    if extra_batches:
        for z_b, th_b in extra_batches:
            th_bt, z_bt = _as_tensors(z_b, th_b, config.device)
            inference.append_simulations(th_bt, z_bt)

    inference.train(
        training_batch_size=config.training_batch_size,
        learning_rate=config.learning_rate,
        validation_fraction=config.validation_fraction,
        stop_after_epochs=config.stop_after_epochs,
        max_num_epochs=config.max_num_epochs,
        clip_max_norm=config.clip_max_norm,
        show_train_summary=False,
    )
    return inference.build_posterior()


def train_ensemble(
    z: np.ndarray,
    theta: np.ndarray,
    prior,
    n_members: int = 10,
    config: Optional[NPEConfig] = None,
    base_seed: int = 0,
    extra_batches: Optional[Sequence] = None,
    verbose: bool = True,
):
    """Train `n_members` independent estimators and return their mixture.

    Members differ only in random seed: initialisation, batch order, and
    the train/validation split. They see the same data, which is the deep
    ensemble construction -- the disagreement between members is what
    supplies the extra uncertainty.

    Returns an EnsemblePosterior whose log_prob is the arithmetic mixture
    (E1) and whose sample() draws a member uniformly and then samples it.
    """
    from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior

    config = config or NPEConfig()
    members = []
    for j in range(n_members):
        if verbose:
            print("[ensemble] training member %d/%d" % (j + 1, n_members), flush=True)
        members.append(
            train_single(z, theta, prior, config=config,
                         seed=base_seed + j, extra_batches=extra_batches)
        )
    return EnsemblePosterior(members)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_ensemble(ensemble, out_dir: str, contract=None, config: Optional[NPEConfig] = None) -> str:
    """Persist ensemble members plus the contract and config that produced them.

    The contract is written alongside deliberately: a trained estimator is
    meaningless without the parameter names, coordinates, and prior box it
    was trained under, and separating them invites a silent axis permutation
    at load time.
    """
    import torch

    os.makedirs(out_dir, exist_ok=True)
    for j, post in enumerate(ensemble.posteriors):
        torch.save(post, os.path.join(out_dir, "member_%02d.pt" % j))
    meta = {"n_members": len(ensemble.posteriors)}
    if config is not None:
        meta["config"] = asdict(config)
    if contract is not None:
        meta["contract"] = contract.to_dict()
    with open(os.path.join(out_dir, "ensemble.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return out_dir


def load_ensemble(out_dir: str):
    """Reload a saved ensemble. Returns (EnsemblePosterior, meta dict)."""
    import torch
    from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior

    with open(os.path.join(out_dir, "ensemble.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    members: List = []
    for j in range(int(meta["n_members"])):
        members.append(torch.load(os.path.join(out_dir, "member_%02d.pt" % j),
                                  weights_only=False))
    return EnsemblePosterior(members), meta
