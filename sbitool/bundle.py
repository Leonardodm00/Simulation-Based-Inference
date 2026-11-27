"""Block 41: a model bundle as a file -- write, read, check (specs/SPEC.md
sec. 2.7, "The model bundle", and block 41; D-174, D-175).

A bundle is one zip: manifest.json, CARD.md, members/member_<k>.safetensors,
evaluation/sim.json, evaluation/gate.safetensors. Reading one never calls
torch.load, pickle or numpy.load.

CONTRACT (wave B): the names and signatures below are fixed.

Every digest is checked by one function, _digest_problem, both by check()
and when a member is read (Bundle.state, gate_sets, sim_metrics), so that a
file changed after it was opened is refused too.

Pure ASCII, LF only.
"""

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile

from sbitool import schema as S

EXIT_ARGS, EXIT_REFUSE = 2, 3
EXTENSION = ".sbibundle"

FORBIDDEN_SUFFIXES = (".pt", ".pkl", ".npy", ".npz")
# the zip entries' time: fixed, so that the same content gives the same bytes
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_PART = ".part"


class BundleError(ValueError):
    """A file is not a valid bundle; the message lists every problem."""


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _digest_problem(name, data, want):
    """None when sha256(data) is `want`; otherwise the problem, naming the
    member."""
    got = _sha256(data)
    if got != want:
        return ("%s: its sha256 %s is not the manifest's %s (the member was changed)"
                % (name, got, want))
    return None


def _member_names(M):
    return [S.MEMBER_FILE % k for k in range(M)]


def _expected_names(M):
    """sec. 2.7's members, in the order write() puts them."""
    return [S.FIXED_MEMBERS[0], S.FIXED_MEMBERS[1]] + _member_names(M) + list(S.FIXED_MEMBERS[2:])


def _manifest_bytes(manifest):
    return (json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n").encode("ascii")


def _parse_json(data):
    """json.loads that refuses NaN and Infinity (they are not JSON)."""
    def bad(c):
        raise ValueError("%s is not a JSON number" % c)
    return json.loads(data.decode("utf-8"), parse_constant=bad)


# ---------------------------------------------------------------- check

def _gate_problems(data, manifest):
    """The gate's file: exactly the three float32 arrays (M, N, d_z), N the
    manifest's gate sizes."""
    from safetensors.numpy import load as st_load
    name = S.FIXED_MEMBERS[3]
    try:
        arrays = st_load(data)
    except Exception as exc:  # noqa: BLE001  (a malformed file: any error of the parser)
        return ["%s: not a safetensors file (%s)" % (name, exc)]
    probs = []
    if sorted(arrays) != sorted(S.GATE_ARRAYS):
        probs.append("%s holds %s, not exactly %s" % (name, sorted(arrays),
                                                     ", ".join(S.GATE_ARRAYS)))
        return probs
    sizes = dict(z_ref=manifest["gate"]["n_ref"], z_rank=manifest["gate"]["n_rank"],
                 z_null=manifest["gate"]["n_null"])
    for k in S.GATE_ARRAYS:
        a = arrays[k]
        want = (manifest["M"], sizes[k], manifest["d_z"])
        if str(a.dtype) != "float32" or tuple(a.shape) != want:
            probs.append("%s: %s is %s %s, not float32 %s (M, the manifest's gate.%s, d_z)"
                         % (name, k, a.dtype, tuple(a.shape), want, "n_" + k[2:]))
    return probs


def _state_problems(name, data):
    from safetensors.torch import load as st_load
    try:
        st_load(data)
    except Exception as exc:  # noqa: BLE001
        return ["%s: not a safetensors file (%s)" % (name, exc)]
    return []


def check(path):
    """Every problem of the file as a bundle, as a list of strings ([] when
    valid): not a zip, members other than exactly sec. 2.7's, a manifest
    that does not validate, a digest that does not hold, a member whose
    name ends in .pt, .pkl, .npy or .npz."""
    if not os.path.isfile(path):
        return ["%s: no such file" % path]
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        return ["%s: not a zip file (%s)" % (path, exc)]
    probs = []
    with zf:
        names = zf.namelist()
        for n in sorted(set(x for x in names if names.count(x) > 1)):
            probs.append("member %s appears %d times" % (n, names.count(n)))
        for n in names:
            if n.lower().endswith(FORBIDDEN_SUFFIXES):
                probs.append("member %s: a %s file is not allowed in a bundle (no pickle, "
                             "no numpy file)" % (n, os.path.splitext(n)[1]))
        data = {}
        for n in sorted(set(names)):
            try:
                data[n] = zf.read(n)
            except (zipfile.BadZipFile, OSError, RuntimeError, EOFError) as exc:
                probs.append("member %s: cannot be read (%s)" % (n, exc))
    man_name = S.FIXED_MEMBERS[0]
    if man_name not in names:
        probs.append("missing member %s" % man_name)
        return probs
    if man_name not in data:
        return probs
    try:
        manifest = _parse_json(data[man_name])
    except (ValueError, UnicodeDecodeError) as exc:
        probs.append("%s: not valid JSON (%s)" % (man_name, exc))
        return probs
    mprobs = S.validate_manifest(manifest)
    probs.extend("%s: %s" % (man_name, p) for p in mprobs)
    M = manifest.get("M") if isinstance(manifest, dict) else None
    if not (isinstance(M, int) and not isinstance(M, bool) and M >= 1):
        return probs
    expected = _expected_names(M)
    for n in expected:
        if n not in names:
            probs.append("missing member %s" % n)
    for n in sorted(set(names) - set(expected)):
        probs.append("unexpected member %s (a bundle holds exactly %s)"
                     % (n, ", ".join(expected)))
    if mprobs:
        return probs            # the digests and shapes below need a valid manifest
    for k, mem in enumerate(manifest["members"]):
        n = mem["file"]
        if n in data:
            p = _digest_problem(n, data[n], mem["sha256"])
            if p:
                probs.append(p)
            else:
                probs.extend(_state_problems(n, data[n]))
    fixed = set(S.FIXED_MEMBERS[1:])
    files = manifest["files"]
    for n in sorted(fixed - set(files)):
        probs.append("%s: files has no digest of %s" % (man_name, n))
    for n in sorted(set(files) - fixed):
        probs.append("%s: files names %s, which is not a member to digest there (%s)"
                     % (man_name, n, ", ".join(sorted(fixed))))
    for n in sorted(fixed & set(files)):
        if n in data:
            p = _digest_problem(n, data[n], files[n])
            if p:
                probs.append(p)
            elif n == S.FIXED_MEMBERS[3]:
                probs.extend(_gate_problems(data[n], manifest))
            elif n == S.FIXED_MEMBERS[2]:
                try:
                    _parse_json(data[n])
                except (ValueError, UnicodeDecodeError) as exc:
                    probs.append("%s: not valid JSON (%s)" % (n, exc))
            elif n == S.FIXED_MEMBERS[1]:
                try:
                    data[n].decode("utf-8")
                except UnicodeDecodeError as exc:
                    probs.append("%s: not UTF-8 text (%s)" % (n, exc))
    return probs


# ---------------------------------------------------------------- reading

class Bundle:
    """An open bundle. Attributes: `path`; `manifest` (dict, validated);
    `card` (str, CARD.md's text); `sha256` (of the zip file). Methods:
    `state(k) -> {name: torch.Tensor}` member k's state, its file's sha256
    verified when read; `gate_sets() -> {"z_ref", "z_rank", "z_null":
    numpy float32 arrays (M, N, d_z)}`; `sim_metrics() -> dict`
    (evaluation/sim.json)."""

    def __init__(self, path):
        path = os.path.abspath(path)
        probs = check(path)
        if probs:
            raise BundleError("%s is not a valid bundle:\n%s" % (path, "\n".join(probs)))
        with open(path, "rb") as fh:
            raw = fh.read()
        self.path = path
        self.sha256 = _sha256(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            self.manifest = _parse_json(zf.read(S.FIXED_MEMBERS[0]))
            card = zf.read(S.FIXED_MEMBERS[1])
        p = _digest_problem(S.FIXED_MEMBERS[1], card, self.manifest["files"][S.FIXED_MEMBERS[1]])
        if p:
            raise BundleError(p)
        self.card = card.decode("utf-8")

    def __repr__(self):
        return "Bundle(%r: %s, %d members)" % (self.path, self.manifest["name"],
                                               self.manifest["M"])

    def _read(self, name, want):
        """The member's bytes, read now from the file, their sha256 verified
        against the manifest read at opening."""
        try:
            with zipfile.ZipFile(self.path) as zf:
                data = zf.read(name)
        except (zipfile.BadZipFile, OSError, KeyError, RuntimeError, EOFError) as exc:
            raise BundleError("%s: member %s cannot be read (%s)" % (self.path, name, exc))
        p = _digest_problem(name, data, want)
        if p:
            raise BundleError("%s: %s" % (self.path, p))
        return data

    def state(self, k):
        """Member k's state_dict {name: torch.Tensor}, from its safetensors
        file, whose sha256 is verified as it is read."""
        from safetensors.torch import load as st_load
        M = self.manifest["M"]
        if not (isinstance(k, int) and 0 <= k < M):
            raise IndexError("member %r of a bundle of %d members" % (k, M))
        mem = self.manifest["members"][k]
        return st_load(self._read(mem["file"], mem["sha256"]))

    def gate_sets(self):
        """{"z_ref", "z_rank", "z_null"}: float32 arrays (M, N, d_z)."""
        from safetensors.numpy import load as st_load
        name = S.FIXED_MEMBERS[3]
        arrays = st_load(self._read(name, self.manifest["files"][name]))
        return dict((k, arrays[k]) for k in S.GATE_ARRAYS)

    def sim_metrics(self):
        """evaluation/sim.json: block 36's <stem>_sim.json of the ensemble."""
        name = S.FIXED_MEMBERS[2]
        return _parse_json(self._read(name, self.manifest["files"][name]))


def read(path):
    """The Bundle at `path`; BundleError listing the problems when
    check(path) finds any."""
    return Bundle(path)


# ---------------------------------------------------------------- writing

def _state_bytes(state):
    """A state_dict as safetensors bytes: every tensor, each its own
    contiguous CPU copy (two names of one shared tensor are both kept, so
    that the state loads strictly)."""
    import torch
    from safetensors.torch import save as st_save
    out = {}
    for name, t in state.items():
        if not isinstance(t, torch.Tensor):
            raise BundleError("state entry %r is a %s, not a tensor" % (name, type(t).__name__))
        out[name] = t.detach().cpu().clone().contiguous()
    return st_save(out)


def _gate_bytes(gate_sets):
    import numpy as np
    from safetensors.numpy import save as st_save
    if sorted(gate_sets) != sorted(S.GATE_ARRAYS):
        raise BundleError("the gate's sets are %s, not exactly %s"
                          % (sorted(gate_sets), ", ".join(S.GATE_ARRAYS)))
    return st_save(dict((k, np.ascontiguousarray(gate_sets[k], dtype=np.float32))
                        for k in S.GATE_ARRAYS))


def _zip_bytes(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return buf.getvalue()


def write(path, manifest, card_text, states, gate_sets, sim_json_bytes):
    """Writes a bundle to `path` (a temporary name, re-read and checked,
    then renamed). `manifest` complete except `members[k]["sha256"]` and
    `files`, which this function fills; `states` a list of M state dicts;
    `gate_sets` the three arrays; `sim_json_bytes` the metrics' file as
    bytes. Returns the final manifest. BundleError when the result does not
    pass check()."""
    man = json.loads(json.dumps(manifest))          # a copy; the caller's is not changed
    M = man.get("M")
    if len(states) != M or len(man.get("members") or []) != M:
        raise BundleError("%d state(s) and %d member entries for M = %r"
                          % (len(states), len(man.get("members") or []), M))
    entries = []
    for k, st in enumerate(states):
        data = _state_bytes(st)
        man["members"][k]["sha256"] = _sha256(data)
        entries.append((man["members"][k]["file"], data))
    card = card_text.encode("utf-8")
    gate = _gate_bytes(gate_sets)
    sim = bytes(sim_json_bytes)
    man["files"] = {S.FIXED_MEMBERS[1]: _sha256(card), S.FIXED_MEMBERS[2]: _sha256(sim),
                    S.FIXED_MEMBERS[3]: _sha256(gate)}
    blob = _zip_bytes([(S.FIXED_MEMBERS[0], _manifest_bytes(man)), (S.FIXED_MEMBERS[1], card)]
                      + entries + [(S.FIXED_MEMBERS[2], sim), (S.FIXED_MEMBERS[3], gate)])
    path = os.path.abspath(path)
    tmp = path + _PART
    with open(tmp, "wb") as fh:
        fh.write(blob)
    probs = check(tmp)
    if probs:
        os.remove(tmp)
        raise BundleError("the bundle written does not pass its check:\n%s" % "\n".join(probs))
    os.replace(tmp, path)
    return man


# ---------------------------------------------------------------- command line

def main(argv=None):
    """`python3 -m sbitool bundle-check FILE [--rebuild]`: 0 with BUNDLE
    CHECK OK: ..., EXIT_ARGS, EXIT_REFUSE with one line per problem;
    --rebuild also builds and loads every member (model.load)."""
    ap = argparse.ArgumentParser(prog="python3 -m sbitool bundle-check",
                                 description="Checks a model bundle (specs/SPEC.md block 41).")
    ap.add_argument("bundle", help="the bundle (<name>%s)" % EXTENSION)
    ap.add_argument("--rebuild", action="store_true",
                    help="also build every member and load its state (verifies the digests "
                         "of the loaded states)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return EXIT_ARGS if exc.code else 0
    probs = check(args.bundle)
    if probs:
        for p in probs:
            print("REFUSE: %s: %s" % (args.bundle, p), flush=True)
        return EXIT_REFUSE
    b = read(args.bundle)
    rebuilt = ""
    if args.rebuild:
        from sbitool import model
        try:
            model.load(b)
        except BundleError as exc:
            for line in str(exc).splitlines():
                print("REFUSE: %s: %s" % (args.bundle, line), flush=True)
            return EXIT_REFUSE
        rebuilt = ", every member rebuilt and its state's digest verified"
    man = b.manifest
    print("BUNDLE CHECK OK: %s (%s, %d members, %d bytes, %s%s)"
          % (args.bundle, man["name"], man["M"], os.path.getsize(b.path),
             "card complete" if man["card_complete"] else "draft", rebuilt), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
