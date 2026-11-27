"""Block 42: the card of a model bundle (specs/SPEC.md sec. 2.7, "The
card", and block 42; D-158 (4), D-173, D-175). PyYAML's safe_load and
safe_dump only.

A card (CARD.md) is a YAML header between two lines `---`, with exactly the
keys schema.CARD_KEYS, then free Markdown. The template pre-fills the fields
the manifest and the members' records hold (training.arm, members,
n_windows, split_fracs, hyperparameters; simulator.searched_parameters;
extraction; software) and writes schema.TODO in every other field; its body
has one headed section per field left to fill, saying what to write there.

CONTRACT (wave B): the signatures below are fixed.

Pure ASCII, LF only.
"""

import argparse
import datetime
import os
import sys
import tempfile

import yaml

from sbitool import schema

EXIT_ARGS, EXIT_REFUSE = 2, 3


class CardError(ValueError):
    """The text is not a card: no header between two --- lines, YAML that
    does not parse or would build Python objects, a header that is not a
    mapping."""


# ---------------------------------------------------------------- the runner's config
# Which keys of a member record's `config` (vars(args) of
# hpc/joint/stage3/run_joint_arms.py's build_parser) go into the card's
# training.hyperparameters: those that describe the model and its
# optimisation. Every other key of the parser is named in RUN_KEYS with the
# reason it stays out, so that the two lists together are the parser's
# destinations exactly (T42a checks it against the parser's source): a new
# flag of the runner fails the smoke test until it is placed in one of them.
MODEL_KEYS = (
    "embedding_size", "depth_exponent", "width_multiplier", "block_family", "head_fusion",
    "dropout", "head_pool_ops", "hidden_features", "num_transforms", "num_bins",
    "flow_hidden_layers")
OPTIMISATION_KEYS = (
    "epochs", "steps_per_epoch", "passes", "patience", "b_sim", "b_met", "b_rep", "lr",
    "weight_decay", "one_minus_beta1", "lambda_dsn", "lambda_rep", "warmup_frac_rep",
    "n_posterior_draws", "encoder_steps", "loss_type", "mining_strategy", "margin",
    "angular_alpha_deg", "lambda_sep", "sep_warmup_frac", "strict_semihard", "rep_pairs",
    "rep_group_key")
HYPERPARAMETER_KEYS = MODEL_KEYS + OPTIMISATION_KEYS
RUN_KEYS = {
    # fields of their own in the card, or a member's own seed
    "arm": "training.arm", "split_fracs": "training.split_fracs",
    "seed": "a member's seed (the manifest's members)",
    "split_seed": "the split's seed (the manifest's members)",
    # paths
    "sim_shards": "path", "real_shards": "path", "out_dir": "path",
    "warm_start_ckpt": "path", "dsn_main_dir": "path", "sbi_hpc_dir": "path",
    # switches of a run
    "dry_run": "switch", "resume": "switch", "stop_after_epoch": "switch",
    "probe_steps": "switch", "probe_val_rows": "switch", "threads": "switch",
    "buffer_shards": "switch (the loader streams the same rows, D-084)",
}


# ---------------------------------------------------------------- parse and dump

def _one_line(msg):
    return " ".join(str(msg).split())


def parse(text):
    """(header, body): the YAML header as a dict and the Markdown after it.
    CardError on a malformed card."""
    if not isinstance(text, str):
        raise CardError("a card is text, got %s" % type(text).__name__)
    first = text.find("\n")
    if first < 0 or text[:first].rstrip("\r") != "---":
        raise CardError("no header: a card begins with a line --- (then the YAML header, "
                        "then a line ---)")
    start = pos = first + 1
    while True:
        nl = text.find("\n", pos)
        end = len(text) if nl < 0 else nl
        if text[pos:end].rstrip("\r") == "---":
            header_text = text[start:pos]
            body = "" if nl < 0 else text[nl + 1:]
            break
        if nl < 0:
            raise CardError("the header has no closing line ---")
        pos = nl + 1
    try:
        header = yaml.safe_load(header_text)
    except yaml.YAMLError as exc:
        # a python/* tag (an object safe_load will not build) is a ConstructorError
        raise CardError("the header is not safe YAML: %s" % _one_line(exc))
    except RecursionError:
        raise CardError("the header is too deeply nested to read")
    if not isinstance(header, dict):
        raise CardError("the header must be a YAML mapping with the keys %s, got %s"
                        % (", ".join(schema.CARD_KEYS), type(header).__name__))
    return header, body


# the order of each nested object of the header (the card's key sets)
_ORDER = {
    (): schema.CARD_KEYS,
    ("simulator",): schema.CARD_SIMULATOR_KEYS,
    ("training",): schema.CARD_TRAINING_KEYS,
    ("extraction",): schema.EXTRACTION_KEYS,
    ("extraction", "detection"): schema.DETECTION_KEYS,
    ("authors", "*"): schema.CARD_AUTHOR_KEYS,
    ("data",): schema.CARD_DATA_KEYS,
    ("data", "simulated_bank"): schema.CARD_BANK_KEYS,
    ("data", "recordings"): schema.CARD_RECORDINGS_KEYS,
    ("software",): schema.SOFTWARE_KEYS,
}


# the types safe_load builds: a header holds nothing else, so that
# parse(dump(h, b)) == (h, b) (safe_dump would write a tuple as a list)
_YAML_SCALARS = (str, int, float, bool, type(None), bytes, datetime.date)


def _ordered(v, path=()):
    """A copy of v whose objects have the card's keys first, in the schema's
    order, then any other key in its own order. A copy, so that no object is
    shared and safe_dump writes no anchor. CardError on a value safe_load
    would not build."""
    if isinstance(v, dict):
        order = _ORDER.get(path, ())
        keys = [k for k in order if k in v] + [k for k in v if k not in order]
        return dict((k, _ordered(v[k], path + (k,))) for k in keys)
    if isinstance(v, list):
        return [_ordered(x, path + ("*",)) for x in v]
    if isinstance(v, set):          # YAML's !!set
        return set(v)
    if not isinstance(v, _YAML_SCALARS):
        raise CardError("the header holds a %s at %s, which YAML's safe loader would not "
                        "read back" % (type(v).__name__, ".".join(str(k) for k in path) or
                                       "the top"))
    return v


def dump(header, body):
    """The card's text: the header by safe_dump in schema.CARD_KEYS' order
    between two --- lines, then the body. parse(dump(h, b)) == (h, b)."""
    if not isinstance(header, dict):
        raise CardError("the header must be a mapping, got %s" % type(header).__name__)
    if not isinstance(body, str):
        raise CardError("the body must be text, got %s" % type(body).__name__)
    try:
        y = yaml.safe_dump(_ordered(header), sort_keys=False, default_flow_style=False,
                           allow_unicode=False)
    except yaml.YAMLError as exc:
        # a value safe_dump does not write (a tuple, an object): not a card's
        raise CardError("the header cannot be written as safe YAML: %s" % _one_line(exc))
    except RecursionError:
        raise CardError("the header is too deeply nested (or refers to itself) to write")
    return "---\n" + y + "---\n" + body


# ---------------------------------------------------------------- the template

def _need(d, *keys):
    """d[k0][k1]...; ValueError naming the key when absent."""
    v = d
    for i, k in enumerate(keys):
        if not isinstance(v, dict) or k not in v:
            raise ValueError("the manifest has no %s" % ".".join(keys[:i + 1]))
        v = v[k]
    return v


def _common(records, getter, what):
    """The one value all records share; TODO when no record holds it; a
    ValueError when they disagree (one shared split, D-089: never a guess)."""
    vals = [getter(r) for r in records]
    vals = [v for v in vals if v is not _MISSING]
    if not vals or len(vals) != len(records):
        if vals:
            raise ValueError("%s is in some members' records and not in others" % what)
        return schema.TODO
    for k, v in enumerate(vals[1:], 1):
        if v != vals[0]:
            raise ValueError("the members' records disagree on %s: %r (member 0) and %r "
                             "(member %d)" % (what, vals[0], v, k))
    return vals[0]


_MISSING = object()


def _hyperparameters(rec):
    cfg = rec.get("config")
    if not isinstance(cfg, dict):
        return _MISSING
    return dict((k, cfg[k]) for k in HYPERPARAMETER_KEYS if k in cfg)


def _template_header(manifest, records):
    """The template's header (a dict in CARD_KEYS' order)."""
    if not isinstance(manifest, dict):
        raise ValueError("the manifest must be a dict, got %s" % type(manifest).__name__)
    arm, M = _need(manifest, "arm"), _need(manifest, "M")
    names = _need(manifest, "parameters", "names")
    extraction = _need(manifest, "extraction")
    software = _need(manifest, "software")
    records = list(records or [])
    if records:
        if len(records) != M:
            raise ValueError("%d records for M = %r members" % (len(records), M))
        seeds = [m.get("seed") for m in manifest.get("members") or []
                 if isinstance(m, dict)]
        for k, r in enumerate(records):
            if not isinstance(r, dict):
                raise ValueError("record %d is not a dict" % k)
            if r.get("arm") != arm:
                raise ValueError("record %d is of arm %r, the manifest's arm is %r"
                                 % (k, r.get("arm"), arm))
            if len(seeds) == M and r.get("seed") != seeds[k]:
                raise ValueError("record %d has seed %r, the manifest's member %d has seed %r "
                                 "(the records go in member order)"
                                 % (k, r.get("seed"), k, seeds[k]))
        # n_train counts the rows of the training split (run_joint_arms.py:
        # n_train = tr.sum() over the bank's rows), and a row is one window
        n_windows = _common(records, lambda r: r.get("n_train", _MISSING), "n_train")
        split_fracs = _common(records, lambda r: r.get("split_fracs", _MISSING),
                              "split_fracs")
        hyper = _common(records, _hyperparameters, "the hyperparameters")
        if isinstance(hyper, dict) and not hyper:
            hyper = schema.TODO
    else:
        n_windows = split_fracs = hyper = schema.TODO
    T = schema.TODO
    header = {
        "title": T,
        "scientific_question": T,
        "simulator": {"model_type": T, "reference": T, "fixed_parameters": T,
                      "searched_parameters": list(names) if isinstance(names, list) else names},
        "training": {"arm": arm, "members": M,
                     # the records hold rows (windows), never networks (donors)
                     "n_networks": T,
                     "n_windows": n_windows, "split_fracs": split_fracs,
                     "hyperparameters": hyper, "hyperparameter_search": T},
        "extraction": schema.NOT_APPLICABLE if extraction is None else extraction,
        "references": T, "authors": T, "licence": T, "cite": T, "data": T,
        "software": software,
        "limitations": T,
    }
    return _ordered(header)


# what an author writes in each field the template leaves to fill (a dotted
# name of schema.card_todo_fields -> plain words); the body's sections
_HELP = {
    "title": (
        "One line naming the model: what it infers, of which system, from which "
        "recordings."),
    "scientific_question": (
        "The question the model answers, in one to three sentences: which parameters of "
        "which system are inferred from which kind of recording, and why they matter."),
    "simulator.model_type": (
        "The kind of simulator that generated the training data, in words: the model "
        "family (for example a network of spiking neurons and its neuron and synapse "
        "models) and the software that ran it."),
    "simulator.reference": (
        "Where the simulator is described or kept: a paper's DOI, or a repository's URL "
        "and the commit that was run."),
    "simulator.fixed_parameters": (
        "Every parameter of the simulator held fixed while the training data were made, "
        "as a mapping name: value with its unit (for example `n_neurons: 1000 neurons`). "
        "The parameters that were varied are listed for you under "
        "simulator.searched_parameters."),
    "training.n_networks": (
        "The number of distinct simulated networks (the donors of the grouped split) "
        "whose windows are in the training split. The runner's records count windows, not "
        "networks, so this is not filled for you: count the distinct donors of the "
        "training rows of the simulated bank under the manifest's split_seed."),
    "training.n_windows": (
        "The number of windows (rows) in the training split; the runner's record holds "
        "it as n_train."),
    "training.split_fracs": (
        "The fractions (train, selection, report) of the split by donor, as three "
        "numbers."),
    "training.hyperparameters": (
        "The runner's settings that describe the model and its optimisation, as a mapping "
        "name: value (the record's config, without paths and switches)."),
    "training.hyperparameter_search": (
        "How the hyperparameters were chosen: the search (its method, the number of "
        "trials and the ranges searched), the quantity it minimised, and how the "
        "finalist was picked."),
    "references": (
        "A list of the works a user of this model should read, one string each (a "
        "citation or a DOI)."),
    "authors": (
        "A list of the people responsible for the model, each entry with `name:` and "
        "`contact:` (an e-mail address or a web page)."),
    "licence": (
        "The licence under which others may use the bundle, as an SPDX identifier (for "
        "example CC-BY-4.0 or MIT)."),
    "cite": (
        "The text a user should cite when they use the model (a paper, or the bundle's "
        "own DOI)."),
    "data": (
        "Where the data are, with two entries. `simulated_bank:` its `location`, "
        "`public` (true or false), and `size_bytes` and `sha256` of the published archive "
        "(null for both when it is not published). `recordings:` the recordings the model "
        "was evaluated on, their `location` and `public`."),
    "limitations": (
        "What the model should not be used for: it was trained only inside the parameter "
        "box of the bounds in the manifest, only on windows made as the extraction "
        "describes, and only on data from its simulator; say which kinds of recording it was not "
        "tested on and any known bias."),
    "extraction": (
        "The extraction of the manifest, or `not applicable` when the bundle takes "
        "windows only."),
    "software": (
        "The software versions of the manifest (python, torch, sbi, zuko, numpy, "
        "safetensors)."),
}


def _help_for(name):
    """The plain-words help of a dotted field name (an author's entry
    authors[0].name falls under `authors`, a nested one under its field)."""
    key = name
    while key and key not in _HELP:
        cut = max(key.rfind("."), key.rfind("["))
        key = key[:cut] if cut > 0 else ""
    return _HELP.get(key, "Write the value this field asks for (specs/SPEC.md sec. 2.7, "
                          "\"The card\").")


def _field_rank(name):
    """A dotted field's place in the header's order (the body's sections
    follow the header)."""
    top, _, sub = name.partition(".")
    top = top.split("[")[0]
    order = _ORDER.get((top,), ())
    sub = sub.split(".")[0].split("[")[0]
    return (schema.CARD_KEYS.index(top) if top in schema.CARD_KEYS else len(schema.CARD_KEYS),
            order.index(sub) if sub in order else -1 if not sub else len(order), name)


def _body(header, manifest, project):
    name = manifest.get("name") if isinstance(manifest.get("name"), str) else "this bundle"
    prov = manifest.get("provenance") if isinstance(manifest.get("provenance"), dict) else {}
    if project is None:
        project = prov.get("project")
    where = []
    if project:
        where.append("project `%s`" % project)
    if prov.get("ensemble"):
        where.append("ensemble `%s`" % prov["ensemble"])
    todo = sorted(schema.card_todo_fields(header), key=_field_rank)
    out = ["", "# The card of `%s`%s" % (name, (" (%s)" % ", ".join(where)) if where else ""),
           "",
           "The header above is the card. The fields that hold `TODO` are yours to fill; "
           "the others were filled from the pipeline's records and the bundle's manifest, "
           "and must keep agreeing with the manifest. Replace each `TODO` with the value "
           "its section below asks for, then run",
           "",
           "    python3 -m sbitool card-check CARD.md --bundle <the bundle>",
           "",
           "which says `CARD OK: complete` when nothing is left to fill. You may replace "
           "this text with any Markdown you like (a longer description, figures); only the "
           "header is checked.",
           ""]
    if todo:
        out += ["## Fields to fill (%d)" % len(todo), ""]
        for f in todo:
            out += ["### `%s`" % f, "", _help_for(f), ""]
    out += ["## Filled for you", "",
            "`training.arm`, `training.members` (the number of members M) and "
            "`simulator.searched_parameters` are the manifest's `arm`, `M` and "
            "`parameters.names`; `extraction` and `software` are the manifest's own. "
            "`training.n_windows`, `training.split_fracs` and `training.hyperparameters` "
            "come from the members' training records, where they hold them; a field the "
            "records do not hold is left `TODO` above rather than guessed.", ""]
    return "\n".join(out)


def template(manifest, records, project=None, n_networks=None):
    """A CARD.md for the bundle of `manifest`: every field the manifest and
    the members' records hold pre-filled, schema.TODO elsewhere, and a body
    of headed sections that say what to write. `records`: the members'
    record dicts (the runner's <arm>_seed<k>.json), in member order.
    `n_networks`: the networks of the training split (no record holds
    them; block 41's export counts them on the bank), or None for TODO."""
    header = _template_header(manifest, records)
    if n_networks is not None:
        if isinstance(n_networks, bool) or not isinstance(n_networks, int) or n_networks < 1:
            raise ValueError("n_networks must be an integer >= 1, got %r" % (n_networks,))
        header["training"]["n_networks"] = int(n_networks)
    return dump(header, _body(header, manifest, project))


def check(text, manifest=None, complete=False):
    """Every problem of the card's text (schema.validate_card_header on its
    header); CardError on a malformed card."""
    header, _ = parse(text)
    return schema.validate_card_header(header, manifest, complete)


# ---------------------------------------------------------------- command lines

class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        sys.stderr.write("%s: error: %s\n" % (self.prog, message))
        raise _ArgsExit()


class _ArgsExit(Exception):
    pass


def _parse_args(parser, argv):
    try:
        return parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    except _ArgsExit:
        return None


def _read_bundle(path):
    """The bundle's manifest and card text (block 41's reader), or a refusal
    line."""
    from sbitool import bundle   # block 41; imported only when a bundle is named
    try:
        b = bundle.read(path)
    except bundle.BundleError as exc:
        return None, None, _one_line(exc)
    return b.manifest, b.card, None


# the training fields the template takes from the records; rebuilt from a
# bundle they come from the bundle's own card
_TRAINING_FROM_RECORDS = ("n_networks", "n_windows", "split_fracs", "hyperparameters")


def main_template(argv=None):
    """`python3 -m sbitool card-template --bundle FILE --out CARD.md`: the
    template rebuilt from a bundle's manifest (the pre-filled fields the
    manifest holds; the training fields from the bundle's own card when it
    has them)."""
    p = _Parser(prog="sbitool card-template",
                description="write a card template (CARD.md) for a model bundle")
    p.add_argument("--bundle", required=True, help="the bundle (.sbibundle)")
    p.add_argument("--out", required=True, help="the CARD.md to write (must not exist)")
    a = _parse_args(p, argv)
    if a is None:
        return EXIT_ARGS
    if not os.path.isfile(a.bundle):
        sys.stderr.write("card-template: no bundle file %s\n" % a.bundle)
        return EXIT_ARGS
    if os.path.exists(a.out):
        sys.stderr.write("CARD TEMPLATE REFUSED: %s exists; the template never overwrites "
                         "a card\n" % a.out)
        return EXIT_REFUSE
    manifest, card_text, err = _read_bundle(a.bundle)
    if err:
        sys.stderr.write("CARD TEMPLATE REFUSED: %s\n" % err)
        return EXIT_REFUSE
    try:
        header = _template_header(manifest, None)
    except ValueError as exc:
        sys.stderr.write("CARD TEMPLATE REFUSED: %s\n" % _one_line(exc))
        return EXIT_REFUSE
    try:
        old, _ = parse(card_text)
    except CardError:
        old = {}
    old_t = old.get("training") if isinstance(old.get("training"), dict) else {}
    for k in _TRAINING_FROM_RECORDS:
        if k in old_t and not (isinstance(old_t[k], str) and old_t[k].strip() == schema.TODO):
            header["training"][k] = old_t[k]
    prov = manifest.get("provenance") if isinstance(manifest.get("provenance"), dict) else {}
    text = dump(header, _body(header, manifest, prov.get("project")))
    probs = check(text, manifest)
    if probs:
        for pr in probs:
            sys.stderr.write("CARD TEMPLATE REFUSED: %s\n" % pr)
        return EXIT_REFUSE
    d = os.path.dirname(os.path.abspath(a.out))
    fd, tmp = tempfile.mkstemp(prefix=".card_", dir=d)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, a.out)
    todo = schema.card_todo_fields(header)
    print("CARD TEMPLATE OK: %s, %d field(s) to fill: %s" % (a.out, len(todo), ", ".join(todo)))
    return 0


def main_check(argv=None):
    """`python3 -m sbitool card-check CARD.md [--bundle FILE]`: 0 with CARD
    OK: complete, or CARD OK: draft, <k> field(s) to fill: <names>;
    EXIT_REFUSE on a malformed card or a key problem."""
    p = _Parser(prog="sbitool card-check", description="check a model bundle's card")
    p.add_argument("card", help="the CARD.md")
    p.add_argument("--bundle", default=None,
                   help="a bundle whose manifest the card must agree with")
    a = _parse_args(p, argv)
    if a is None:
        return EXIT_ARGS
    if not os.path.isfile(a.card):
        sys.stderr.write("card-check: no card file %s\n" % a.card)
        return EXIT_ARGS
    if a.bundle is not None and not os.path.isfile(a.bundle):
        sys.stderr.write("card-check: no bundle file %s\n" % a.bundle)
        return EXIT_ARGS
    with open(a.card, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        sys.stderr.write("CARD REFUSED: %s is not UTF-8 text (%s)\n" % (a.card, _one_line(exc)))
        return EXIT_REFUSE
    manifest = None
    if a.bundle is not None:
        manifest, _, err = _read_bundle(a.bundle)
        if err:
            sys.stderr.write("CARD REFUSED: the bundle: %s\n" % err)
            return EXIT_REFUSE
    try:
        header, _ = parse(text)
    except CardError as exc:
        sys.stderr.write("CARD REFUSED: %s\n" % _one_line(exc))
        return EXIT_REFUSE
    probs = schema.validate_card_header(header, manifest)
    if probs:
        for pr in probs:
            sys.stderr.write("CARD REFUSED: %s\n" % _one_line(pr))
        return EXIT_REFUSE
    todo = schema.card_todo_fields(header)
    if todo:
        print("CARD OK: draft, %d field(s) to fill: %s" % (len(todo), ", ".join(todo)))
    else:
        rest = schema.validate_card_header(header, manifest, complete=True)
        if rest:  # card_todo_fields and the complete check see the same fields
            for pr in rest:
                sys.stderr.write("CARD REFUSED: %s\n" % _one_line(pr))
            return EXIT_REFUSE
        print("CARD OK: complete")
    if manifest is None:
        print("(not compared with a manifest: no --bundle given)")
    return 0


if __name__ == "__main__":
    sys.exit(main_check())
