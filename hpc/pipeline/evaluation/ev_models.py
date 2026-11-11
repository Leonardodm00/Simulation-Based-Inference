"""The rebuild of an ensemble's member (block 19; specs/SPEC.md sec. 2.5 and
block 19). The ensembles themselves are found by `ev_plan` (standard
library only); its names are re-exported here.

    model, info = ev_models.rebuild(record_path, index)

A member is rebuilt exactly as the runner built it (hpc/joint/stage3/
run_joint_arms.py): the backbone by `make_backbone` from the record's
`config`, the flow by `build_joint_model` on the box prior with the first 64
training rows of the record's split for sbi's shapes, then the checkpoint
with its digest check.

Pure ASCII, LF only.
"""

import json
import os

import _evalpath  # noqa: F401  (sys.path; the pipeline's config first)

import numpy as np
import torch

from ev_plan import (Ensemble, RefuseError, ensembles, find, load_project,  # noqa: F401
                     member_record, search_view, sha256_file)

import run_joint_arms as R
from joint_model import build_joint_model


def rebuild(record_path, index):
    """(model, info): the member of `record_path`, rebuilt and loaded, in
    eval mode, against the simulated bank `index` (a ShardIndex)."""
    with open(record_path) as fh:
        rec = json.load(fh)
    a = rec["config"]
    ckpt_path = record_path[:-len(".json")] + "_ckpt.pt"
    if not os.path.isfile(ckpt_path):
        raise RefuseError("%s: no checkpoint %s" % (record_path, ckpt_path))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ck.get("meta", {})
    split, h = R.grouped_split(index.donor, fracs=tuple(rec["split_fracs"]),
                               seed=int(rec["split_seed"]))
    if h != rec["split_hash"]:
        raise RefuseError("%s: the bank's split %s is not the record's %s (another bank?)"
                          % (record_path, h[:12], str(rec["split_hash"])[:12]))
    ex = np.flatnonzero(split == 0)[:64]
    theta_ex = torch.as_tensor(np.asarray(index.theta)[ex], dtype=torch.float32)
    x_ex = R._rows_x(index, ex)
    if meta.get("fixed_summary"):
        backbone, kind = R.FixedStatsSummary(), "fixed-stats"
    else:
        backbone, kind = R.make_backbone(
            int(meta["W"]), int(meta["embedding_size"]), a.get("dsn_main_dir"), int(a["seed"]),
            encoder=dict((k, a[k]) for k in ("depth_exponent", "width_multiplier",
                                             "block_family", "head_fusion", "dropout")),
            head_pool_ops=int(a.get("head_pool_ops", 0)))
    d = int(meta["d_theta"])
    from sbi.utils import BoxUniform
    model = build_joint_model(backbone, BoxUniform(low=torch.zeros(d), high=torch.ones(d)),
                              theta_ex, x_ex, hidden_features=int(a["hidden_features"]),
                              num_transforms=int(a["num_transforms"]),
                              num_bins=int(a["num_bins"]),
                              z_score_theta=meta.get("z_score_theta", "transform_to_unconstrained"),
                              z_score_x=meta.get("z_score_x", "none"),
                              flow_hidden_layers=a.get("flow_hidden_layers"), meta=meta)
    model.load_checkpoint(ck)
    model.eval()
    info = {"record": record_path, "ckpt": ckpt_path, "ckpt_sha256": sha256_file(ckpt_path),
            "arm": rec["arm"], "seed": int(rec["seed"]), "split_seed": int(rec["split_seed"]),
            "split_hash": rec["split_hash"], "backbone": kind, "d_theta": d,
            "embedding_size": int(meta["embedding_size"]),
            "L": rec.get("L"), "L_pseudo_real": rec.get("L_pseudo_real")}
    return model, info
