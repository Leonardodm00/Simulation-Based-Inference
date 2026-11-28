"""Block 41: a model bundle as a file -- write, read, check (specs/SPEC.md
sec. 2.7, "The model bundle", and block 41; D-174, D-175).

A bundle is one zip: manifest.json, CARD.md, members/member_<k>.safetensors,
evaluation/sim.json, evaluation/gate.safetensors, and, when the manifest's
`training` is not null (format "sbitool_bundle/2", block 48),
training/member_<k>.json and, when it says so, training/search.json.
Reading one never calls torch.load, pickle or numpy.load.

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
from sbitool import training_info as TI

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


def _expected_names(M, training=None):
    """sec. 2.7's members, in the order write() puts them, then the training's
    files the manifest's `training` announces (block 48)."""
    return ([S.FIXED_MEMBERS[0], S.FIXED_MEMBERS[1]] + _member_names(M) + list(S.FIXED_MEMBERS[2:])
            + S.training_files(training))


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


_ABSENT = object()


def _card_header(data):
    """CARD.md's header by block 42's card.parse, or None when the card does
    not parse (a card that does not parse is not complete)."""
    from sbitool import card as CARD     # block 42; imported only to read a card
    try:
        header, _body = CARD.parse(data.decode("utf-8"))
    except (CARD.CardError, UnicodeDecodeError):
        return None
    return header


def _agreement_problems(data, manifest):
    """The manifest compared with the members it describes (block 41, fix
    round of 2026-11-28, D-180): card_complete with the bundle's own card,
    gate with evaluation/sim.json's gate on schema.GATE_KEYS,
    parameters.names and .bounds with sim.json's axes where it holds axes.
    One problem per disagreement; each member compared only when its digest
    holds (a changed member is reported by its digest). M and d_z against
    the gate's sets are _gate_problems'."""
    man_name, card_name, sim_name = S.FIXED_MEMBERS[0], S.FIXED_MEMBERS[1], S.FIXED_MEMBERS[2]
    files = manifest["files"]
    probs = []
    if card_name in data and _sha256(data[card_name]) == files.get(card_name):
        header = _card_header(data[card_name])
        complete = (header is not None
                    and not S.validate_card_header(header, manifest, complete=True))
        if bool(manifest["card_complete"]) != complete:
            probs.append("%s: card_complete is %s but %s is %s" % (
                man_name, "true" if manifest["card_complete"] else "false", card_name,
                "a complete card" if complete else
                ("a card that does not parse" if header is None else "not a complete card")))
    if sim_name in data and _sha256(data[sim_name]) == files.get(sim_name):
        try:
            sim = _parse_json(data[sim_name])
        except (ValueError, UnicodeDecodeError):
            return probs                 # reported by check() as not valid JSON
        gate = sim.get("gate", _ABSENT) if isinstance(sim, dict) else _ABSENT
        if not isinstance(gate, dict):
            probs.append("%s has no gate (an object) to compare the manifest's gate with"
                         % sim_name)
        else:
            for k in S.GATE_KEYS:
                a, b = manifest["gate"][k], gate.get(k, _ABSENT)
                if b is _ABSENT or a != b:
                    probs.append("%s: gate.%s is %s but %s's gate.%s is %s" % (
                        man_name, k, json.dumps(a), sim_name, k,
                        "absent" if b is _ABSENT else json.dumps(b)))
        if isinstance(sim, dict) and "axes" in sim:
            axes = sim["axes"]
            for k in ("names", "bounds"):
                a = manifest["parameters"][k]
                b = axes.get(k, _ABSENT) if isinstance(axes, dict) else _ABSENT
                if b is _ABSENT or a != b:
                    probs.append("%s: parameters.%s is %s but %s's axes.%s is %s" % (
                        man_name, k, json.dumps(a), sim_name, k,
                        "absent" if b is _ABSENT else json.dumps(b)))
    return probs


def _training_problems(n, data, manifest):
    """A training file whose digest holds: valid JSON of block 48's form;
    a member's seed the manifest's member's."""
    try:
        d = _parse_json(data)
    except (ValueError, UnicodeDecodeError) as exc:
        return ["%s: not valid JSON (%s)" % (n, exc)]
    if n == S.TRAINING_SEARCH_FILE:
        return TI.search_problems(d)
    k = int(n[len(S.TRAINING_PREFIX) + len("member_"):-len(".json")])
    return TI.member_problems(d, k, seed=manifest["members"][k]["seed"])


def check(path):
    """Every problem of the file as a bundle, as a list of strings ([] when
    valid): not a zip, members other than exactly sec. 2.7's (and the
    training's files the manifest announces, block 48: a training/ member
    it does not announce is refused), a manifest
    that does not validate, a digest that does not hold, a member whose
    name ends in .pt, .pkl, .npy or .npz; a manifest that disagrees with the
    members it describes (card_complete with CARD.md, gate and parameters
    with evaluation/sim.json, M and d_z with the gate's sets). A manifest
    edited into another valid one that agrees with every member passes:
    the file cannot vouch for itself, its sha256 is its identity."""
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
    training = manifest.get("training")
    tr_names = S.training_files(training)
    expected = _expected_names(M, training)
    for n in expected:
        if n not in names:
            probs.append("missing member %s" % n)
    for n in sorted(set(names) - set(expected)):
        if n.startswith(S.TRAINING_PREFIX):
            probs.append("unexpected member %s: a %s member the manifest's training (%s) does "
                         "not announce" % (n, S.TRAINING_PREFIX, json.dumps(training)))
        else:
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
    fixed = set(S.FIXED_MEMBERS[1:]) | set(tr_names)
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
            elif n in tr_names:
                probs.extend(_training_problems(n, data[n], manifest))
    probs.extend(_agreement_problems(data, manifest))
    return probs


# ---------------------------------------------------------------- reading

class Bundle:
    """An open bundle. Attributes: `path`; `manifest` (dict, validated);
    `card` (str, CARD.md's text); `sha256` (of the zip file). Methods:
    `training() -> None | {"members": [...], "search": dict | None}` (block
    48; each file's sha256 verified when read);
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

    def training(self):
        """None when the manifest's training is null; else {"members": the M
        dicts of training/member_<k>.json, "search": training/search.json's
        dict or None}, each file's sha256 verified as it is read."""
        tr = self.manifest.get("training")
        if tr is None:
            return None
        files = self.manifest["files"]
        members = [_parse_json(self._read(S.TRAINING_MEMBER_FILE % k,
                                          files[S.TRAINING_MEMBER_FILE % k]))
                   for k in range(tr["members"])]
        search = None
        if tr["search"]:
            search = _parse_json(self._read(S.TRAINING_SEARCH_FILE,
                                            files[S.TRAINING_SEARCH_FILE]))
        return {"members": members, "search": search}


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


def _training_entries(training, man):
    """[(name, bytes)] of the training's files, and the manifest's
    `training` value; BundleError when a member's curve is not of block 48's
    form (the five lists of one length >= 1, its seed the manifest's
    member's) or the search's summary is not."""
    if training is None:
        return [], None
    try:
        members, search = training
    except (TypeError, ValueError):
        raise BundleError("training must be None or (members, search), got %r"
                          % type(training).__name__)
    M = man.get("M")
    if not isinstance(members, (list, tuple)) or len(members) != M:
        raise BundleError("training: %s member curve(s) for M = %r"
                          % (len(members) if isinstance(members, (list, tuple)) else "no", M))
    entries, probs = [], []
    for k, m in enumerate(members):
        m = dict(m) if isinstance(m, dict) else m
        if isinstance(m, dict):
            for key in TI.MEMBER_LISTS:
                if isinstance(m.get(key), (list, tuple)):
                    m[key] = [TI.finite_or_none(x) for x in m[key]]
        p = TI.member_problems(m, k, seed=man["members"][k].get("seed"))
        probs.extend(p)
        if not p:
            entries.append((S.TRAINING_MEMBER_FILE % k, TI.json_bytes(m)))
    if search is not None:
        p = TI.search_problems(search)
        probs.extend(p)
        if not p:
            entries.append((S.TRAINING_SEARCH_FILE, TI.json_bytes(search)))
    if probs:
        raise BundleError("the training cannot be written:\n%s" % "\n".join(probs))
    return entries, {"members": M, "search": search is not None}


def write(path, manifest, card_text, states, gate_sets, sim_json_bytes, training=None):
    """Writes a bundle to `path` (a temporary name, re-read and checked,
    then renamed). `manifest` complete except `members[k]["sha256"]`,
    `training` and `files`, which this function fills; `states` a list of M
    state dicts; `gate_sets` the three arrays; `sim_json_bytes` the metrics'
    file as bytes; `training` None (the manifest's training null) or
    (members, search) as training_info.collect returns them (block 48; a NaN
    in a curve is written null). Returns the final manifest. BundleError
    when the training is not of block 48's form or the result does not
    pass check()."""
    man = json.loads(json.dumps(manifest))          # a copy; the caller's is not changed
    M = man.get("M")
    if len(states) != M or len(man.get("members") or []) != M:
        raise BundleError("%d state(s) and %d member entries for M = %r"
                          % (len(states), len(man.get("members") or []), M))
    tr_entries, man["training"] = _training_entries(training, man)
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
    for n, data in tr_entries:
        man["files"][n] = _sha256(data)
    blob = _zip_bytes([(S.FIXED_MEMBERS[0], _manifest_bytes(man)), (S.FIXED_MEMBERS[1], card)]
                      + entries + [(S.FIXED_MEMBERS[2], sim), (S.FIXED_MEMBERS[3], gate)]
                      + tr_entries)
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
    CHECK OK: ..., sha256 <the file's, in full>, EXIT_ARGS, EXIT_REFUSE with one line per problem;
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
    # the line ends with the file's sha256 in full: the bundle's identity, which a
    # publication or a hub holds outside the file (block 41, fix round of 2026-11-28)
    print("BUNDLE CHECK OK: %s (%s, %d members, %d bytes, %s%s), sha256 %s"
          % (args.bundle, man["name"], man["M"], os.path.getsize(b.path),
             "card complete" if man["card_complete"] else "draft", rebuilt, b.sha256),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
