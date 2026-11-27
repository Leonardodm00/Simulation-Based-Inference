"""Block 41: `python3 -m sbitool export` -- a trained ensemble of a recorded
project, with its evaluation (blocks 35-36), written as a model bundle
(specs/SPEC.md block 41).

    python3 -m sbitool export --config <tag>.json --eval-config <tag>_rec_eval.json \\
        --ensemble <name> --out <file>.sbibundle [--extraction FILE.json] [--card CARD.md] \\
        [--name NAME]

What is read: the project's view and the evaluation's configuration (block
33); the members, rebuilt by the pipeline's own ev_models.rebuild on the
simulated bank's index (the one place a checkpoint, a .pt, is read); block
35's <stem>_sample.json and <stem>_sim.npz (the gate's sets: the rows of `z`
at the roles 1, 2, 3 of `gate_role`, in row order); block 36's
<stem>_sim.json and METRICS.json. Every refusal (exit 3) comes before
anything is written; the bundle is written to a temporary name, re-read,
checked and every member rebuilt from it (model.load), then renamed.

CONTRACT (wave B): the signatures below are fixed.

Pure ASCII, LF only.
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import types

from sbitool import _sources
from sbitool import bundle as B
from sbitool import schema as S

EXIT_ARGS, EXIT_REFUSE = 2, 3
_TMP = ".tmp"


class ExportRefused(RuntimeError):
    """The export cannot be made; nothing is written (exit 3)."""


def _pipeline():
    """The evaluation's modules, through _sources (model and evaluation
    groups); returns a namespace of them."""
    sources = _sources.use("model", "evaluation")
    import _evalpath  # noqa: F401  (the pipeline's config and stages on sys.path)
    import config as C
    import ev_models
    import ev_plan
    import ev_rec_config
    import ev_rec_data
    import ev_rec_metrics
    import ev_rec_plan
    import ev_gate
    import joint_model
    return types.SimpleNamespace(
        sources=sources, C=C, ev_models=ev_models, ev_plan=ev_plan, ev_rec_config=ev_rec_config,
        ev_rec_data=ev_rec_data, ev_rec_metrics=ev_rec_metrics, ev_rec_plan=ev_rec_plan,
        ev_gate=ev_gate, joint_model=joint_model)


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- the card's header

def _card_header(text):
    """The card's header by block 42's card.parse; a malformed card is a
    refusal."""
    from sbitool import card as CARD
    try:
        header, _body = CARD.parse(text)
    except CARD.CardError as exc:
        raise ExportRefused("the card is malformed: %s" % exc)
    return header


# ---------------------------------------------------------------- the manifest's parts

def _architecture(rec, meta, kind):
    """schema.ARCHITECTURE_KEYS of one member: the runner's record `config`
    and the checkpoint's meta (as ev_models.rebuild reads them)."""
    a = rec["config"]
    hl = a.get("flow_hidden_layers")
    return {"backbone": kind, "embedding_size": int(meta["embedding_size"]),
            "depth_exponent": int(a["depth_exponent"]),
            "width_multiplier": float(a["width_multiplier"]),
            "block_family": int(a["block_family"]), "head_fusion": int(a["head_fusion"]),
            "dropout": float(a["dropout"]), "head_pool_ops": int(a.get("head_pool_ops", 0)),
            "hidden_features": int(a["hidden_features"]),
            "num_transforms": int(a["num_transforms"]), "num_bins": int(a["num_bins"]),
            "hidden_layers": None if hl is None else int(hl),
            "z_score_theta": meta.get("z_score_theta", "transform_to_unconstrained"),
            "z_score_x": meta.get("z_score_x", "none"),
            "backbone_repr": meta["backbone_repr"]}


def _units(sidecar, p, bounds, given=None):
    """parameters.units, p strings (block 41, "Inputs"; D-178): the list of
    --units when given; else the sidecar's latent_spec.units when it holds
    p non-empty strings; else "1" on every axis when the declared
    coordinate is the unit box itself (no bounds, or every bound [0, 1]:
    the value reported is u, a pure number); else a refusal naming --units.
    A unit is never guessed."""
    def ok(u):
        return isinstance(u, list) and len(u) == p and all(isinstance(v, str) and v for v in u)
    if given is not None:
        if not ok(given):
            raise ExportRefused("--units must be a JSON list of %d non-empty strings, got %r"
                                % (p, given))
        return list(given)
    side = sidecar or {}
    lat = side.get("latent_spec") or {}
    u = lat.get("units") if isinstance(lat, dict) else None
    if ok(u):
        return list(u)
    if bounds is None or (isinstance(bounds, list) and len(bounds) == p
                          and all(isinstance(x, (list, tuple)) and len(x) == 2
                                  and float(x[0]) == 0.0 and float(x[1]) == 1.0 for x in bounds)):
        return ["1"] * p
    raise ExportRefused("the parameters' units are not known: the simulated bank's sidecar has "
                        "no latent_spec.units of %d entries and the parameters have declared "
                        "bounds; give them with --units FILE.json (a list of %d strings, \"1\" "
                        "for a dimensionless one)" % (p, p))


def _n_train_networks(P, index, rec):
    """The networks of the training split of a member's record: the bank's
    donors on the rows grouped_split gives the record's fractions and seed
    (split 0), as the runner and ev_models.rebuild split it."""
    import numpy as np
    split, h = P.ev_models.R.grouped_split(index.donor, fracs=tuple(rec["split_fracs"]),
                                 seed=int(rec["split_seed"]))
    if h != rec["split_hash"]:
        raise ExportRefused("the bank's split is not the record's (another bank?)")
    return int(np.unique(np.asarray(index.donor)[np.asarray(split) == 0]).size)


def _software():
    import numpy
    import safetensors
    import sbi
    import torch
    import zuko
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "sbi": str(sbi.__version__), "zuko": str(zuko.__version__),
            "numpy": str(numpy.__version__), "safetensors": str(safetensors.__version__)}


def _commit():
    """The repository's commit (git rev-parse HEAD), "unknown" without git."""
    try:
        r = subprocess.run(["git", "-C", _sources.REPO, "rev-parse", "HEAD"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           universal_newlines=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    out = r.stdout.strip()
    return out if r.returncode == 0 and out else "unknown"


def _metrics_left_out(P, ecfg, e, sample_json):
    """Why ensemble e's metrics (block 36) are absent or not current: None
    when METRICS.json reports it, its <stem>_sim.json exists, and the index
    was made from this very sample (its report_inputs' digest of the
    sample's provenance) under this configuration's report_params."""
    mdir = ecfg.sub("metrics")
    idx_path = os.path.join(mdir, "METRICS.json")
    if not os.path.isfile(idx_path):
        return "no metrics: %s absent" % idx_path
    idx = _read_json(idx_path)
    if e.name not in (idx.get("ensembles") or {}):
        why = [x.get("why") for x in idx.get("left_out") or [] if x.get("ensemble") == e.name]
        return "the metrics do not report it%s" % (": %s" % why[0] if why else "")
    sim = os.path.join(mdir, "%s_sim.json" % e.stem)
    if not os.path.isfile(sim):
        return "no metrics: %s absent" % sim
    ri = idx.get("report_inputs") or {}
    if (ri.get("samples") or {}).get(e.stem) != _sha256_file(sample_json):
        return ("the metrics were not made from the current sample (%s changed since; run the "
                "metrics again)" % sample_json)
    if ri.get("params") != ecfg.report_params():
        return "the metrics were made with other values: %s" % ri.get("params")
    return None


def _gate_sets(P, npz_path, sim_gate):
    """The rows of z at the roles 1, 2, 3 of gate_role, in row order;
    ExportRefused when the sample holds no such sets."""
    import numpy as np
    G = P.ev_gate
    with np.load(npz_path) as z:
        if not all(k in z.files for k in ("gate_role", "z_idx", "z")):
            raise ExportRefused("the gate's sets are not in the sample: %s has no gate_role, "
                                "z_idx or z" % npz_path)
        role, z_idx, Z = z["gate_role"], z["z_idx"], z["z"]
    pos = np.full(role.shape[0], -1, dtype=np.int64)
    pos[z_idx] = np.arange(z_idx.size)
    out = {}
    for name, r, n_key in (("z_ref", G.ROLE_REF, "n_ref"), ("z_rank", G.ROLE_RANK, "n_rank"),
                           ("z_null", G.ROLE_NULL, "n_null")):
        rows = np.flatnonzero(role == r)
        if rows.size == 0 or np.any(pos[rows] < 0):
            raise ExportRefused("the gate's sets are not in the sample: role %d (%s) has %d "
                                "row(s), %d of them without an embedding (%s)"
                                % (r, name, rows.size, int(np.sum(pos[rows] < 0)), npz_path))
        if rows.size != int(sim_gate[n_key]):
            raise ExportRefused("the sample's %s has %d row(s), the metrics' gate.%s %s"
                                % (name, rows.size, n_key, sim_gate[n_key]))
        out[name] = np.ascontiguousarray(Z[:, pos[rows], :], dtype=np.float32)
    return out


def _placeholder(manifest):
    """A copy with digests of the right form where write() fills them, so
    that the manifest is validated before anything is written."""
    m = json.loads(json.dumps(manifest))
    for mem in m["members"]:
        mem["sha256"] = "0" * 64
    m["files"] = dict((n, "0" * 64) for n in S.FIXED_MEMBERS[1:])
    return m


# ---------------------------------------------------------------- the export

def run(cfg_path, ecfg_path, name, out, extraction=None, card=None, bundle_name=None,
        log=print, units=None):
    """Exports ensemble `name` of the project at `cfg_path`, evaluated under
    `ecfg_path`, to the bundle `out`. `extraction`: None or the path of a
    JSON file with exactly schema.EXTRACTION_KEYS; `card`: None (block 42's
    template, a draft) or the path of a CARD.md. Returns the manifest.
    ExportRefused (or ev_plan.RefuseError) before anything is written."""
    P = _pipeline()
    out = os.path.abspath(out)
    if not out.endswith(B.EXTENSION):
        raise ExportRefused("the bundle's file must end in %s: %s" % (B.EXTENSION, out))
    if not os.path.isdir(os.path.dirname(out)):
        raise ExportRefused("the bundle's folder %s does not exist" % os.path.dirname(out))
    bname = bundle_name or os.path.basename(out)[:-len(B.EXTENSION)]

    # ---- the optional inputs first: a refusal costs nothing
    ext = None
    if extraction is not None:
        try:
            ext = _read_json(extraction)
        except (OSError, ValueError) as exc:
            raise ExportRefused("--extraction %s cannot be read as JSON: %s" % (extraction, exc))
        probs = S.validate_extraction(ext)
        if probs:
            raise ExportRefused("--extraction %s is not valid: %s" % (extraction, "; ".join(probs)))
    units_given = None
    if units is not None:
        try:
            units_given = _read_json(units)
        except (OSError, ValueError) as exc:
            raise ExportRefused("--units %s cannot be read as JSON: %s" % (units, exc))
    card_text = card_header = None
    if card is not None:
        try:
            with open(card, encoding="utf-8") as fh:
                card_text = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            raise ExportRefused("--card %s cannot be read: %s" % (card, exc))
        card_header = _card_header(card_text)

    # ---- the ensemble, its sample and its metrics (current, block 36's rule)
    ecfg = P.ev_rec_config.load(ecfg_path)
    view = P.ev_rec_plan.load_view(cfg_path, ecfg)
    e = P.ev_rec_plan.find(view, name)
    why = P.ev_rec_metrics.sample_left_out(ecfg, e)
    if why is not None:
        raise ExportRefused("ensemble %s: its sample is absent or not current: %s" % (name, why))
    sample_json = P.ev_rec_plan.sample_json(ecfg, e.stem)
    why = _metrics_left_out(P, ecfg, e, sample_json)
    if why is not None:
        raise ExportRefused("ensemble %s: its metrics are absent or not current: %s"
                            % (name, why))
    prov = _read_json(sample_json)
    sim_path = os.path.join(ecfg.sub("metrics"), "%s_sim.json" % e.stem)
    with open(sim_path, "rb") as fh:
        sim_bytes = fh.read()
    sim = json.loads(sim_bytes.decode("utf-8"))
    gate = dict((k, sim["gate"][k]) for k in S.GATE_KEYS)
    gate_sets = _gate_sets(P, os.path.join(ecfg.sub("samples"), "%s_sim.npz" % e.stem),
                           sim["gate"])
    log("ensemble %s: %d member(s); sample %s and metrics %s current"
        % (e.name, len(e.records), sample_json, sim_path))

    # ---- the members, by the pipeline's rebuild (the one place a .pt is read)
    index, _paths = P.ev_rec_data.sim_index(view)
    states, members, records, arch = [], [], [], None
    p = W = None
    for k, rpath in enumerate(e.records):
        model, info = P.ev_models.rebuild(rpath, index)
        rec = _read_json(rpath)
        meta = model.meta                       # the checkpoint's meta (load_checkpoint)
        pm = prov["members"][k] if k < len(prov.get("members") or []) else {}
        if pm.get("ckpt_sha256") != info["ckpt_sha256"] or pm.get("seed") != info["seed"]:
            raise ExportRefused("member %d (seed %d): the sample was made from another "
                                "checkpoint (%s), not %s; sample it again"
                                % (k, info["seed"], pm.get("ckpt_sha256"), info["ckpt_sha256"]))
        a = _architecture(rec, meta, info["backbone"])
        if arch is None:
            arch, p, W = a, int(meta["d_theta"]), int(meta["W"])
        elif a != arch or int(meta["d_theta"]) != p or int(meta["W"]) != W:
            raise ExportRefused("member %d (seed %d) has another architecture than member 0: "
                                "one manifest describes every member" % (k, info["seed"]))
        state = model.state_dict()
        states.append(state)
        members.append({"file": S.MEMBER_FILE % k, "sha256": None, "seed": int(info["seed"]),
                        "state_sha256": P.joint_model.state_sha256(state),
                        "split_seed": int(rec["split_seed"]), "split_hash": rec["split_hash"]})
        records.append(rec)
        log("  member %d: seed %d, %s backbone, state %s" % (k, info["seed"], info["backbone"],
                                                             members[-1]["state_sha256"][:12]))
    M = len(members)
    if int(prov["p"]) != p:
        raise ExportRefused("the sample has p = %s, the members' networks %d" % (prov["p"], p))
    axes = prov.get("axes") or {}
    names = list(axes.get("names") or [])
    bounds = axes.get("bounds")
    units = _units(index.sidecar, p, bounds, units_given)
    d_z = int(gate_sets["z_ref"].shape[2])

    # ---- the manifest
    tag = view.cfg.data["dataset"]
    sources = dict((rel, v["sha256"]) for rel, v in sorted(P.sources.items()))
    manifest = {
        "schema": S.BUNDLE_SCHEMA, "name": bname,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "card_complete": False, "arm": e.arm, "M": M, "p": p, "d_z": d_z, "W": W,
        "members": members, "architecture": arch,
        "parameters": {"names": names, "units": units, "bounds": bounds},
        "extraction": ext, "gate": gate,
        "sample": {"n_post": int(ecfg.n_post), "seed": int(ecfg.seed),
                   "batch_rows": int(ecfg.batch_rows)},
        "provenance": {"project": tag, "ensemble": e.name,
                       "search": e.search if e.search is not None else "best",
                       "campaign": e.campaign,
                       "best_fit_complete_sha256": prov["best_fit_complete_sha256"],
                       "sample_json_sha256": _sha256_file(sample_json),
                       "commit": _commit(), "sources": sources},
        "software": _software(), "files": {}}
    probs = S.validate_manifest(_placeholder(manifest))
    if probs:
        raise ExportRefused("the manifest would not be valid: %s" % "; ".join(probs))

    # ---- the card: the given one, validated against the manifest, or block 42's template
    if card_header is not None:
        probs = S.validate_card_header(card_header, manifest)
        if probs:
            raise ExportRefused("--card %s is not valid for this bundle: %s"
                                % (card, "; ".join(probs)))
        complete = not S.validate_card_header(card_header, manifest, complete=True)
    else:
        from sbitool import card as CARD
        card_text = CARD.template(manifest, records, project=tag,
                                  n_networks=_n_train_networks(P, index, records[0]))
        complete = False
    manifest["card_complete"] = bool(complete)

    # ---- written to a temporary name, re-read, checked, rebuilt; then renamed
    tmp = out + _TMP
    try:
        final = B.write(tmp, manifest, card_text, states, gate_sets, sim_bytes)
        from sbitool import model as SM
        SM.load(B.read(tmp))
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, out)
    size = os.path.getsize(out)
    log("BUNDLE EXPORT OK: %s (%d members, %d bytes, %s)"
        % (out, M, size, "card complete" if complete else "draft"))
    return final


def main(argv=None):
    """The command line: 0 with BUNDLE EXPORT OK: ..., EXIT_ARGS,
    EXIT_REFUSE."""
    ap = argparse.ArgumentParser(prog="python3 -m sbitool export",
                                 description="Exports a trained ensemble as a model bundle "
                                             "(specs/SPEC.md block 41).")
    ap.add_argument("--config", required=True, help="the project's configuration (<tag>.json)")
    ap.add_argument("--eval-config", required=True,
                    help="the evaluation's configuration (<tag>_rec_eval.json)")
    ap.add_argument("--ensemble", required=True, help="the ensemble's name (<q>:<arm>)")
    ap.add_argument("--out", required=True, help="the bundle (<file>%s)" % B.EXTENSION)
    ap.add_argument("--extraction", default=None,
                    help="a JSON file with exactly the extraction's keys")
    ap.add_argument("--units", default=None,
                    help="a JSON list of the parameters' units (\"1\" for a dimensionless one)")
    ap.add_argument("--card", default=None, help="a CARD.md (default: block 42's template)")
    ap.add_argument("--name", default=None, help="the bundle's name (default: the file's)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return EXIT_ARGS if exc.code else 0
    P = _pipeline()
    refusals = (ExportRefused, B.BundleError, P.ev_plan.RefuseError,
                P.ev_rec_config.RecEvalConfigError, P.C.ConfigError)
    try:
        run(args.config, args.eval_config, args.ensemble, args.out,
            extraction=args.extraction, card=args.card, bundle_name=args.name,
            units=args.units,
            log=lambda t: print(t, flush=True))
    except refusals as exc:
        print("REFUSE: %s" % exc, flush=True)
        return EXIT_REFUSE
    except NotImplementedError as exc:
        print("REFUSE: no --card given and the card's template is not available: %s" % exc,
              flush=True)
        return EXIT_REFUSE
    return 0


if __name__ == "__main__":
    sys.exit(main())
