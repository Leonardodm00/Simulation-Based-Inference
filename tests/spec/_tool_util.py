"""Helpers of the tester's spec tests of the tool (blocks 40-46), 2026-11-28.

Inputs only: ptrain folders written with scipy.io.savemat in the formats the
extractor declares (a raster (n, 1) uint8 of 0/1; a scipy.sparse (n, 1) of
the peaks), Recording objects built by hand, a minimal bundle for the
commands that only read a bundle's manifest. No expected value is computed
here.
"""

import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SBIX = os.environ.get("SBIX_CLONE", "/home/claude/sbix")
os.environ.setdefault("SBIX_CLONE", SBIX)


def write_ptrain(folder, spikes, n_samples, fmt="raster", varname="ptrain"):
    """One ptrain_<k>.mat per electrode k of `spikes` (k -> sample indices)."""
    import scipy.io as sio
    import scipy.sparse as sp
    os.makedirs(folder, exist_ok=True)
    for k, s in spikes.items():
        s = np.asarray(s, dtype=np.int64)
        if fmt == "raster":
            r = np.zeros((n_samples, 1), dtype=np.uint8)
            r[s, 0] = 1
            sio.savemat(os.path.join(folder, "ptrain_%d.mat" % k), {varname: r})
        else:
            vals = np.full(s.size, -50.0)            # peak amplitudes: ignored by the reader
            m = sp.csc_matrix((vals, (s, np.zeros(s.size, dtype=np.int64))),
                              shape=(n_samples, 1))
            sio.savemat(os.path.join(folder, "ptrain_%d.mat" % k), {varname: m})
    return folder


def extraction(**kw):
    e = {"electrodes_per_subset": 9, "n_subsets": 2, "mfr_threshold_hz": 0.0,
         "w_size_s": 0.01, "gaussian_window_s": 0.02, "grid_width": 6, "index_base": 0,
         "pitch_um": 60.0, "fs_raw_hz": 10000.0, "detection": None}
    e.update(kw)
    return e


def grid_recording(spikes, n_samples, fs=10000.0, width=6, base=0, pitch=60.0, name="g"):
    """A Recording with a grid: electrode id = its linear index (row-major)."""
    from sbitool.interfaces import Recording
    pos = None
    if pitch is not None:
        pos = {}
        for e in spikes:
            i0 = int(e) - base
            pos[int(e)] = (float((i0 % width) * pitch), float((i0 // width) * pitch))
    return Recording(name=name, spikes={int(k): np.asarray(v, dtype=np.int64)
                                        for k, v in spikes.items()},
                     fs_raw_hz=float(fs), n_samples=int(n_samples), positions_um=pos,
                     grid=(int(width), int(base), None if pitch is None else float(pitch)))


def positions_recording(spikes_by_pos, n_samples, fs=10000.0, name="p", rename=None):
    """A Recording with positions and no grid: spikes_by_pos {(x, y): train};
    electrode ids 0.. in sorted (x, y) order, or rename[(x, y)] when given."""
    from sbitool.interfaces import Recording
    spikes, pos = {}, {}
    for i, xy in enumerate(sorted(spikes_by_pos)):
        e = i if rename is None else int(rename[xy])
        spikes[e] = np.asarray(spikes_by_pos[xy], dtype=np.int64)
        pos[e] = (float(xy[0]), float(xy[1]))
    return Recording(name=name, spikes=spikes, fs_raw_hz=float(fs), n_samples=int(n_samples),
                     positions_um=pos, grid=None)


def manifest_for(extraction_block, W=10, M=1, p=2, d_z=3):
    """A valid manifest (sec. 2.7) for a minimal bundle; digests filled by write()."""
    H = "a" * 64
    return {
        "schema": "sbitool_bundle/1", "name": "mini", "created": "2026-11-28T00:00:00+00:00",
        "card_complete": False, "arm": "A1", "M": M, "p": p, "d_z": d_z, "W": W,
        "members": [{"file": "members/member_%d.safetensors" % k, "sha256": H, "seed": 1 + k,
                     "state_sha256": H, "split_seed": 1, "split_hash": H} for k in range(M)],
        "architecture": {"backbone": "fixed-stats", "embedding_size": 8, "depth_exponent": 0,
                         "width_multiplier": 1.0, "block_family": 0, "head_fusion": 0,
                         "dropout": 0.0, "head_pool_ops": 0, "hidden_features": 16,
                         "num_transforms": 2, "num_bins": 4, "hidden_layers": None,
                         "z_score_theta": "transform_to_unconstrained", "z_score_x": "none",
                         "backbone_repr": "FixedStatsSummary()"},
        "parameters": {"names": ["a%d" % k for k in range(p)], "units": ["1"] * p,
                       "bounds": None},
        "extraction": copy.deepcopy(extraction_block),
        "gate": {"factors": [1.0], "alpha": 0.5, "n_ref": 2, "n_rank": 2, "n_null": 2,
                 "k_gate": 1, "size_mean": 0.33, "size_lo": 0.01, "size_hi": 0.8,
                 "level": 0.95},
        "sample": {"n_post": 4, "seed": 1, "batch_rows": 4},
        "provenance": {"project": "mini", "ensemble": "S-A1:A1", "search": "S-A1",
                       "campaign": "S-A1", "best_fit_complete_sha256": H,
                       "sample_json_sha256": H, "commit": "abc", "sources": {}},
        "software": {"python": "3", "torch": "2", "sbi": "0", "zuko": "1", "numpy": "2",
                     "safetensors": "0"},
        "files": {},
    }


def mini_bundle(path, extraction_block, W=10):
    """A bundle that passes bundle.check, for the commands that read only its
    manifest (windows, device-figure). Its states are placeholders: it cannot
    be rebuilt and is never used for an inference."""
    import torch
    from sbitool import bundle as B
    man = manifest_for(extraction_block, W=W)
    g = dict((k, np.zeros((1, 2, 3), dtype=np.float32)) for k in ("z_ref", "z_rank", "z_null"))
    # [2026-11-28_0151] block 41's fix round (D-180): bundle.check compares the manifest's
    # gate with evaluation/sim.json's, so a hand-made bundle's sim.json holds that gate
    import json
    sim = json.dumps({"gate": man["gate"]}, sort_keys=True).encode("ascii")
    B.write(path, man, "---\ntitle: TODO\n---\n", [{"w": torch.zeros(1)}], g, sim)
    return path
