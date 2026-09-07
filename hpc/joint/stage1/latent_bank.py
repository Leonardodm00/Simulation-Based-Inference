"""Bank shards and the sidecar contract (plan v0.6, S4.3).

One shard is an .npz of per-row arrays plus a JSON sidecar describing the
contract everything downstream depends on. npz rather than parquet because a
window is a (W,) array and a parquet column per sample is the wrong shape.

The contract digest is a sha256 over a CANONICAL subset of the sidecar --
the fields whose change would invalidate a frozen split or a trained model.
Fields that are provenance-only (timestamps, host, shard index) are excluded
deliberately, so that regenerating one shard on another node does not
invalidate the campaign. Which fields are in and which are out is the whole
content of this module, so it is spelled out in CONTRACT_FIELDS rather than
derived by "everything except".

Pure ASCII, LF only. numpy + stdlib only.
"""

import hashlib
import json
import os

import numpy as np

SCHEMA_VERSION = "latent_bank/1"

# Per-row arrays. Name -> (ndim, dtype kind). ndim 2 means (n_rows, k).
ROW_ARRAYS = {
    "x": (2, "f"),                 # (n, W) the window
    "theta": (2, "f"),             # (n, n_latent) == phi
    "cls": (1, "i"),               # (n,) class label
    "nu": (2, "f"),                # (n, N_COMPONENTS)
    "realisation_id": (1, "u"),    # (n,) uint64
    "donor": (1, "i"),
    "well": (1, "i"),
    "batch": (1, "i"),
    "subregion": (1, "i"),
    "window_idx": (1, "i"),
    "contaminated": (1, "b"),      # (n,) bool, from latent_gap (d)
}

# Sidecar fields that enter the digest. Order is fixed here, not by dict order.
CONTRACT_FIELDS = (
    "schema_version",
    "param_names",
    "bounds_theta",
    "coord",
    "fs",
    "w_size",
    "T_win",
    "W",
    "scale_convention",
    "latent_spec",
    "nuisance_spec",
    "realisation_spec",
    "gap_spec",
    "generator_sha256",
    "theta_withheld",
    "arm",
)


class ContractError(ValueError):
    """Raised when a shard or sidecar violates the contract."""


def canonical_json(obj):
    """Deterministic JSON: sorted keys, no whitespace jitter, ASCII only."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def contract_digest(sidecar):
    """sha256 over the contract subset of the sidecar."""
    missing = [k for k in CONTRACT_FIELDS if k not in sidecar]
    if missing:
        raise ContractError("sidecar missing contract fields: %r" % (missing,))
    subset = {k: sidecar[k] for k in CONTRACT_FIELDS}
    return hashlib.sha256(canonical_json(subset).encode("ascii")).hexdigest()


def build_sidecar(param_names, bounds_theta, coord, fs, w_size, T_win, W,
                  scale_convention, latent_spec, nuisance_spec,
                  realisation_spec, gap_spec, generator_sha256,
                  theta_withheld, arm, extra=None):
    """Assemble a sidecar and stamp its digest.

    `extra` carries provenance that is deliberately OUTSIDE the digest.
    """
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "param_names": list(param_names),
        "bounds_theta": [[float(a), float(b)] for a, b in bounds_theta],
        "coord": str(coord),
        "fs": float(fs),
        "w_size": float(w_size),
        "T_win": float(T_win),
        "W": int(W),
        "scale_convention": str(scale_convention),
        "latent_spec": latent_spec,
        "nuisance_spec": nuisance_spec,
        "realisation_spec": realisation_spec,
        "gap_spec": gap_spec,
        "generator_sha256": str(generator_sha256),
        "theta_withheld": bool(theta_withheld),
        "arm": str(arm),
    }
    sidecar["contract_digest"] = contract_digest(sidecar)
    sidecar["provenance"] = dict(extra or {})
    return sidecar


def check_arrays(arrays):
    """Validate the per-row arrays against ROW_ARRAYS. Raises ContractError."""
    missing = [k for k in ROW_ARRAYS if k not in arrays]
    if missing:
        raise ContractError("shard missing arrays: %r" % (missing,))
    extra = [k for k in arrays if k not in ROW_ARRAYS]
    if extra:
        raise ContractError("shard has unknown arrays: %r" % (extra,))

    n_rows = None
    for name, (ndim, kind) in sorted(ROW_ARRAYS.items()):
        a = np.asarray(arrays[name])
        if a.ndim != ndim:
            raise ContractError("%s: expected ndim %d, got %d"
                                % (name, ndim, a.ndim))
        if a.dtype.kind != kind:
            raise ContractError("%s: expected dtype kind %r, got %r"
                                % (name, kind, a.dtype.kind))
        if n_rows is None:
            n_rows = a.shape[0]
        elif a.shape[0] != n_rows:
            raise ContractError("%s: %d rows, expected %d"
                                % (name, a.shape[0], n_rows))
    if n_rows == 0:
        raise ContractError("shard has zero rows")

    if not np.all(np.isfinite(np.asarray(arrays["x"]))):
        raise ContractError("x contains non-finite values")
    if not np.all(np.isfinite(np.asarray(arrays["theta"]))):
        raise ContractError("theta contains non-finite values")
    return n_rows


def write_shard(path, arrays, sidecar):
    """Write one shard plus its sidecar. Atomic: write, then rename."""
    n_rows = check_arrays(arrays)
    if sidecar.get("contract_digest") != contract_digest(sidecar):
        raise ContractError("sidecar digest does not match its contract fields")
    if np.asarray(arrays["x"]).shape[1] != sidecar["W"]:
        raise ContractError("x width %d disagrees with sidecar W %d"
                            % (np.asarray(arrays["x"]).shape[1], sidecar["W"]))

    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    tmp_npz = path + ".tmp"
    np.savez_compressed(tmp_npz, **{k: np.asarray(v) for k, v in arrays.items()})
    # np.savez appends .npz if the name lacks it; handle both.
    written = tmp_npz if os.path.exists(tmp_npz) else tmp_npz + ".npz"
    os.replace(written, path)

    side_path = os.path.splitext(path)[0] + ".json"
    tmp_json = side_path + ".tmp"
    with open(tmp_json, "w") as fh:
        fh.write(canonical_json(sidecar))
    os.replace(tmp_json, side_path)
    return n_rows


def read_shard(path):
    """Read one shard. Returns (arrays dict, sidecar dict), both validated."""
    side_path = os.path.splitext(path)[0] + ".json"
    with open(side_path, "r") as fh:
        sidecar = json.load(fh)
    if sidecar.get("contract_digest") != contract_digest(sidecar):
        raise ContractError("sidecar digest mismatch on read: %s" % side_path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    check_arrays(arrays)
    return arrays, sidecar


def concat_shards(paths):
    """Read several shards and concatenate, asserting one common contract."""
    if not paths:
        raise ContractError("no shards given")
    all_arrays = None
    digest = None
    sidecar0 = None
    for p in paths:
        arrays, sidecar = read_shard(p)
        if digest is None:
            digest = sidecar["contract_digest"]
            sidecar0 = sidecar
            all_arrays = {k: [v] for k, v in arrays.items()}
        else:
            if sidecar["contract_digest"] != digest:
                raise ContractError("shard %s has a different contract" % p)
            for k in all_arrays:
                all_arrays[k].append(arrays[k])
    merged = {k: np.concatenate(v, axis=0) for k, v in all_arrays.items()}
    check_arrays(merged)
    return merged, sidecar0


def shape_report(arrays, sidecar):
    """A short human-readable summary, for the probe run's expected output."""
    n = np.asarray(arrays["x"]).shape[0]
    lines = [
        "arm              : %s" % sidecar["arm"],
        "rows             : %d" % n,
        "W                : %d" % sidecar["W"],
        "p (latent dim)   : %d" % np.asarray(arrays["theta"]).shape[1],
        "theta_withheld   : %s" % sidecar["theta_withheld"],
        "distinct donors  : %d" % len(np.unique(arrays["donor"])),
        "distinct wells   : %d" % len(np.unique(arrays["well"])),
        "distinct realis. : %d" % len(np.unique(arrays["realisation_id"])),
        "contaminated     : %d" % int(np.sum(arrays["contaminated"])),
        "contract_digest  : %s" % sidecar["contract_digest"][:16],
    ]
    return "\n".join(lines)
