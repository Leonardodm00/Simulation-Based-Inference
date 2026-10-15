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


SMALL_ARRAYS = tuple(k for k in ROW_ARRAYS if k != "x")


def npz_member_shape(z, name):
    """The shape of member `name` of an open NpzFile from its .npy header,
    without decompressing the array (the header is the first bytes of the
    compressed stream, which the zip reader inflates incrementally)."""
    with z.zip.open(name + ".npy") as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(fh)
        else:
            shape, _fortran, _dtype = np.lib.format.read_array_header_2_0(fh)
    return tuple(int(s) for s in shape)


class ShardIndex(object):
    """The bank without its windows: every shard's small arrays read once,
    `x` on demand per shard (the streaming loader, D-084; specs/SPEC.md
    block 5).

    Rows are numbered in the concatenated order of `concat_shards` (the
    given path order, which callers sort): shard k holds rows
    offsets[k]:offsets[k+1]. Every shard must carry the first shard's
    contract digest, as `concat_shards` requires.

    An `.npz` member is decompressed only when it is accessed, so reading
    the small arrays costs no window.
    """

    def __init__(self, paths):
        paths = list(paths)
        if not paths:
            raise ContractError("no shards given")
        self.paths = paths
        self.sidecar = None
        self.n_shards = len(paths)
        per_shard = []
        n_rows = []
        for p in paths:
            side_path = os.path.splitext(p)[0] + ".json"
            with open(side_path, "r") as fh:
                sidecar = json.load(fh)
            if sidecar.get("contract_digest") != contract_digest(sidecar):
                raise ContractError("sidecar digest mismatch on read: %s" % side_path)
            if self.sidecar is None:
                self.sidecar = sidecar
            elif sidecar["contract_digest"] != self.sidecar["contract_digest"]:
                raise ContractError("shard %s has a different contract" % p)
            with np.load(p) as z:
                missing = [k for k in ROW_ARRAYS if k not in z.files]
                if missing:
                    raise ContractError("shard %s missing arrays: %r" % (p, missing))
                small = dict((k, z[k]) for k in SMALL_ARRAYS)
                # the windows' row count from the .npy header: x itself is
                # NOT decompressed here (the whole point of the index)
                x_shape = npz_member_shape(z, "x")
                n = int(x_shape[0]) if x_shape else 0
                if len(x_shape) != 2 or x_shape[1] != int(self.sidecar["W"]):
                    raise ContractError("shard %s: x has shape %r, expected (n, %d)"
                                        % (p, x_shape, int(self.sidecar["W"])))
            for name in SMALL_ARRAYS:
                ndim, kind = ROW_ARRAYS[name]
                a = small[name]
                if a.ndim != ndim or a.dtype.kind != kind:
                    raise ContractError("%s: %s expected ndim %d kind %r, got ndim %d kind %r"
                                        % (p, name, ndim, kind, a.ndim, a.dtype.kind))
                if a.shape[0] != small["theta"].shape[0]:
                    raise ContractError("%s: %s has %d rows, theta %d"
                                        % (p, name, a.shape[0], small["theta"].shape[0]))
            n_small = int(np.asarray(small["theta"]).shape[0])
            if n != n_small:
                raise ContractError("shard %s: x has %d rows, theta %d" % (p, n, n_small))
            if n == 0:
                raise ContractError("shard %s has zero rows" % p)
            per_shard.append(small)
            n_rows.append(n)
        self.n_rows_per_shard = tuple(n_rows)
        self.offsets = np.concatenate([[0], np.cumsum(n_rows)]).astype(np.int64)
        self.n_rows = int(self.offsets[-1])
        self.W = int(self.sidecar["W"])
        for k in SMALL_ARRAYS:
            setattr(self, k, np.concatenate([s[k] for s in per_shard], axis=0))
        if not np.all(np.isfinite(self.theta)):
            raise ContractError("theta contains non-finite values")

    def rows_of(self, k):
        """The global row ids of shard k, in order."""
        return np.arange(self.offsets[k], self.offsets[k + 1], dtype=np.int64)

    def shard_of_row(self, row):
        """The shard index holding global row `row`."""
        k = int(np.searchsorted(self.offsets, row, side="right") - 1)
        if k < 0 or k >= self.n_shards:
            raise IndexError(row)
        return k

    def load_x(self, k):
        """The windows of shard k, (n_k, W) float32, read now."""
        with np.load(self.paths[k]) as z:
            x = np.asarray(z["x"], dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.W or x.shape[0] != self.n_rows_per_shard[k]:
            raise ContractError("shard %s: x has shape %r, expected (%d, %d)"
                                % (self.paths[k], x.shape, self.n_rows_per_shard[k], self.W))
        if not np.all(np.isfinite(x)):
            raise ContractError("shard %s: x contains non-finite values" % self.paths[k])
        return x

    def small_arrays(self):
        """The small arrays as one dict (the shape of `concat_shards`' output
        without `x`)."""
        return dict((k, getattr(self, k)) for k in SMALL_ARRAYS)

    def iter_batches(self, row_mask, batch_size):
        """Yield (rows, x) over the masked rows in global order, in batches
        of `batch_size` consecutive masked rows that may span shards -- the
        SAME batches an in-memory caller forms from the concatenated split,
        so sums over them agree bit for bit. At most two shards' windows
        are resident at a time."""
        row_mask = np.asarray(row_mask, dtype=bool).ravel()
        if row_mask.size != self.n_rows:
            raise ValueError("row_mask has %d entries, the bank %d rows"
                             % (row_mask.size, self.n_rows))
        batch_size = int(batch_size)
        pending_rows, pending_x = [], []
        n_pending = 0
        for k in range(self.n_shards):
            rows = self.rows_of(k)
            keep = row_mask[rows]
            if not keep.any():
                continue
            x = self.load_x(k)[keep]
            rows = rows[keep]
            start = 0
            while start < rows.size:
                take = min(batch_size - n_pending, rows.size - start)
                pending_rows.append(rows[start:start + take])
                pending_x.append(x[start:start + take])
                n_pending += take
                start += take
                if n_pending == batch_size:
                    yield (np.concatenate(pending_rows), np.concatenate(pending_x, axis=0))
                    pending_rows, pending_x, n_pending = [], [], 0
        if n_pending:
            yield (np.concatenate(pending_rows), np.concatenate(pending_x, axis=0))


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
