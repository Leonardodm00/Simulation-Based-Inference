#!/usr/bin/env python3
"""
inventory_joint_knobs.py -- the machine-readable inventory of every tunable
parameter of the joint DSN+NPE stack (hpc/joint), read from the source by AST
and from the job scripts by regex. Pure Python 3.9+, no third-party imports,
nothing is executed: this file never imports the stack it inventories.

What it is for
--------------
The parameter documents under hpc/joint/docs/ (P0..P7) are written against
this inventory and checked against it: `--check-index` compares the table
block of 00_INDEX.md with a fresh extraction, so a parameter added to the
code without a ledger row, or a default that moved, fails the check instead
of going stale silently. The extraction is deterministic (sorted rows, repr-
grade values), so two runs on the same tree are byte-identical.

Surfaces
--------
  cli        every `add_argument(...)` of the CLI modules, with its
             subcommand context where the parser has subparsers
  signature  keyword defaults of named functions and `__init__` methods
  dataclass  field defaults of named dataclasses (with a name filter)
  constant   module-level and class-level ALL_CAPS assignments
  job_var    `: "${VAR:=default}"` lines, `NAME="${NAME:-default}"` lines,
             bash arrays and `#PBS` directives of the job scripts

Every row carries an OWNER, the document of the set that explains it
(P1..P7, or `plumbing` for paths and switches that P0 lists without a deep
dive). The owner table is explicit; an unowned row is reported, and with
`--strict` it is an error, so the mapping is complete by construction.

Usage
-----
  python3 inventory_joint_knobs.py --hpc-dir <repo>/hpc --out-json inventory.json --out-md inventory.md
  python3 inventory_joint_knobs.py --hpc-dir <repo>/hpc --check-index 00_INDEX.md
  python3 inventory_joint_knobs.py --hpc-dir <repo>/hpc --list-unowned

Pure ASCII, LF only.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "TARGETS", "OWNER_RULES", "extract_all", "extract_cli",
    "extract_signatures", "extract_dataclasses", "extract_constants",
    "extract_job_script", "assign_owners", "render_markdown", "render_json",
    "check_index", "INDEX_BEGIN", "INDEX_END",
]

INDEX_BEGIN = "<!-- inventory:begin -->"
INDEX_END = "<!-- inventory:end -->"
VALUE_SHORT = 72          # characters shown in the markdown; JSON keeps all
ALLCAPS = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
_STAMP = re.compile(r" at commit `[^`]*`")

# ---------------------------------------------------------------------------
# What is read. Paths are relative to --hpc-dir. The joint stack (D-037) plus
# the DSN and NPE-tuner objects the joint stack reads its fixed values and
# ranges from (BackboneConfig, SearchConfig, the DSN TrainConfig loss fields,
# condition_space, npe_tune_search.default_space / SpaceSpec).
# ---------------------------------------------------------------------------

TARGETS: Dict[str, Any] = {
    "cli": [
        "joint/stage1/build_latent_bank.py",
        "joint/stage1/demo_classes_generate.py",
        "joint/stage1/demo_classes_plot.py",
        "joint/stage3/run_joint_arms.py",
        "joint/stage3/probe_dsn_runtime.py",
        "joint/stage3/report_joint_arms.py",
        "joint/stage3b/run_stage3b.py",
        "joint/stage3c/run_stage3c.py",
        "joint/stage4/npe_tune_joint.py",
        "joint/stage4/joint_space.py",
    ],
    # file -> qualified names: "func" or "Class.__init__" or "Class.method"
    "signature": {
        "joint/stage2/joint_train.py": ["TrainConfig.__init__", "train_joint",
                                        "evaluate_npe"],
        "joint/stage2/joint_batches.py": ["BatchSpec.__init__",
                                          "ThreeStreamBatcher.__init__"],
        "joint/stage2/joint_losses.py": ["ReplicateConsistencyLoss.__init__",
                                         "replicate_statistic", "replicate_loss",
                                         "box_prior_covariance"],
        "joint/stage2/joint_model.py": ["build_joint_model"],
        "joint/stage2/dsn_loss_adapter.py": ["DSNLossConfig.__init__",
                                             "build_dsn_loss"],
        "joint/stage3/run_joint_arms.py": ["grouped_split", "train_encoder_only",
                                           "make_backbone"],
        "joint/stage4/joint_space.py": ["default_joint_space", "boundary_axes",
                                        "apply_control_shuffle"],
        "joint/stage4/npe_tune_joint.py": ["build_argv"],
        "joint/stage1/latent_nuisance.py": ["NuisanceSpec.__init__"],
        "joint/stage1/latent_gap.py": ["GapSpec.__init__"],
        "joint/stage1/latent_realisation.py": ["RealisationSpec.__init__"],
        "joint/stage1/latent_sbi_simulator.py": ["LatentSBISpec.__init__"],
        "npe_tune_search.py": ["default_space"],
    },
    # file -> {class: [field names] or None for all}
    "dataclass": {
        "joint/stage4/joint_space.py": {"JointSpaceSpec": None,
                                        "Campaign": None},
        "dsn/backbone.py": {"BackboneConfig": None},
        "dsn/config.py": {
            "SearchConfig": "_range|_choices",     # regex on the field name
            "RegularizationConfig": "_range",      # dropout_range lives here
            "TrainConfig": "^(loss_type|margin|swap|mining_strategy|"
                           "angular_alpha_deg|strict_semihard|lambda_sep|"
                           "sep_warmup_frac|lr|weight_decay|one_minus_beta1|"
                           "one_minus_beta2|batch_size|max_epochs|patience|"
                           "grad_clip|positives_mode)$",
        },
        "npe_tune_search.py": {"SpaceSpec": None},
    },
    # files whose module-level and class-level ALL_CAPS assignments are read
    "constant": [
        "joint/stage2/joint_losses.py",
        "joint/stage2/joint_train.py",
        "joint/stage2/joint_batches.py",
        "joint/stage3/run_joint_arms.py",
        "joint/stage4/joint_space.py",
        "joint/stage4/npe_tune_joint.py",
        "joint/stage1/bench_burst_provider.py",
        "joint/stage1/latent_gap.py",
        "joint/stage1/latent_nuisance.py",
        "joint/stage1/latent_realisation.py",
        "joint/stage1/build_latent_bank.py",
        "joint/stage3c/run_stage3c.py",
        "dsn/condition_space.py",
        "npe_tune_search.py",
    ],
    "job": [
        "joint/stage1/jobs/build_latent_bank.pbs",
        "joint/stage3/jobs/joint_arms.pbs",
        "joint/stage3b/jobs/stage3b.pbs",
        "joint/stage3c/jobs/stage3c.pbs",
        "joint/stage4/jobs/joint_tune.pbs",
        "joint/stage4/jobs/launch_joint_tune.sh",
    ],
}

# ---------------------------------------------------------------------------
# Owners. Rules are (regex over "surface|relpath|name", owner), first match
# wins. `name` is the flag (--lr), the dotted parameter (TrainConfig.__init__.lr),
# the dataclass field (JointSpaceSpec.lr), the constant (DEFAULT_JITTER) or
# the job variable (LAMBDA_DSN). No catch-all: what nothing matches is
# reported as unowned.
# ---------------------------------------------------------------------------

OWNER_RULES: List[Tuple[str, str]] = [
    # --- plumbing: paths, selectors, dry runs, outputs, identifiers ---------
    (r"\|--(out|out-dir|out-json|out-dir|results-dir|runs-dir|sim-shards|"
     r"real-shards|ckpt|probe-ckpt|bootstrap-dir|dsn-main-dir|sbi-hpc-dir|"
     r"runner|demo|config-json|pending-id|tag|split-hash|contract-digest|"
     r"validation|campaign|dry-run|plan|score|warm-start-ckpt|arm|shard-index|"
     r"max-records|allow-real|nuisance|embed)$", "plumbing"),
    (r"\|(OUT_DIR|SIM_SHARDS|REAL_SHARDS|RUNS_DIR|OUT|CKPT|RESULTS_DIR|"
     r"CAMPAIGN|TAG|SPLIT_HASH|CONTRACT_DIGEST|DRYRUN|ENV_NAME|SKIP_CONDA|"
     r"PYBIN|WARM_START_CKPT|ARM|PROVIDER|PBS_-N|PBS_-j|PBS_-l|PENDING_DIR)$",
     "plumbing"),
    (r"\|(PENDING_FILE|LEDGER_FILE|CONTROLS_FILE|FINALISTS_FILE|"
     r"DEFAULT_RUNNER|_JOINT_DIR|_HERE|INDEX_BEGIN)$", "plumbing"),
    (r"\|build_argv\.(config|arm|sim_shards|out_dir|seed|real_shards|runner|"
     r"python|dsn_main_dir|sbi_hpc_dir|dry_run|spec_fixed|extra)$", "plumbing"),
    (r"\|(make_backbone|train_encoder_only|grouped_split|build_joint_model|"
     r"build_dsn_loss|train_joint|evaluate_npe|default_joint_space|"
     r"boundary_axes|apply_control_shuffle|default_space|replicate_statistic|"
     r"replicate_loss|box_prior_covariance)\.(dsn_main_dir|seed|log_fn|"
     r"backbone|x|cls|dsn_loss_fn|prior|theta_example|x_example|meta|model|"
     r"batcher|cfg|val_theta|val_x|rep_criterion|theta|batch_size|groups|"
     r"config|spec|campaign_name|z|lower|upper|device|m_g|m_gp|C_g|C_gp|"
     r"T|p_eff|W|E|encoder|n_classes|total_steps|p|embedding_dim|d_theta|"
     r"n_train|steps|lr)$", "plumbing"),
    (r"\|ThreeStreamBatcher\.__init__\.(sim_theta|sim_x|real_x|real_cls|"
     r"real_donor|surrogate|spec|seed|met_x|met_cls)$", "plumbing"),
    (r"\|ReplicateConsistencyLoss\.__init__\.(Sigma0|n_draws)$", "P3"),
    # --- encoder block (P1) --------------------------------------------------
    (r"\|--(depth-exponent|width-multiplier|block-family|embedding-size|"
     r"head-fusion|dropout)$", "P1"),
    (r"\|(JointSpaceSpec|SearchConfig|RegularizationConfig)\.(depth_exponent|"
     r"width_multiplier|block_family|embedding_size|head_fusion|dropout|"
     r"head_pool_ops)(_range|_choices)?$", "P1"),
    (r"\|BackboneConfig\.", "P1"),
    (r"\|(HEAD_POOL_OPS_LEVELS)$", "P1"),
    # --- DSN-loss block (P2) -------------------------------------------------
    (r"\|--(loss-type|mining-strategy|margin|angular-alpha-deg|lambda-sep|"
     r"sep-warmup-frac|strict-semihard|lambda-dsn|b-met|encoder-steps|"
     r"n-classes|total-steps)$", "P2"),
    (r"\|(LAMBDA_DSN|ENCODER_STEPS)$", "P2"),
    (r"\|DSNLossConfig\.__init__\.", "P2"),
    (r"\|(JointSpaceSpec|SearchConfig|TrainConfig)\.(loss_type|mining_strategy|"
     r"margin|angular_alpha_deg|lambda_sep|sep_warmup_frac|strict_semihard|"
     r"swap|log10_lambda_dsn|positives_mode|sep_centre_means)"
     r"(_range|_choices)?$", "P2"),
    (r"\|(MINING_STRATEGIES|LOSS_TYPES|LOSS_HP_SUPERSET|_ACTIVE|_MINING_TAG|"
     r"_LOSS_TAG)$", "P2"),
    (r"\|build_dsn_loss\.", "P2"),
    (r"\|train_encoder_only\.", "P2"),
    (r"\|BatchSpec\.__init__\.b_met$", "P2"),
    # --- replicate block (P3) ------------------------------------------------
    (r"\|--(lambda-rep|warmup-frac-rep|n-posterior-draws|b-rep|n-post-draws)$",
     "P3"),
    (r"\|(LAMBDA_REP|N_DRAWS)$", "P3"),
    (r"\|JointSpaceSpec\.(log10_lambda_rep|warmup_frac_rep|n_posterior_draws)$",
     "P3"),
    (r"\|(DEFAULT_JITTER|DEFAULT_T_FLOOR|DEFAULT_P_EFF_MIN)$", "P3"),
    (r"\|ReplicateConsistencyLoss\.__init__\.", "P3"),
    (r"\|replicate_statistic\.(n_draws|correct_mc|jitter|detach_metric)$", "P3"),
    (r"\|replicate_loss\.t_floor$", "P3"),
    (r"\|box_prior_covariance\.dtype$", "P3"),
    (r"\|BatchSpec\.__init__\.b_rep$", "P3"),
    (r"\|train_joint\.n_posterior_draws$", "P3"),
    # --- flow block (P4) -----------------------------------------------------
    (r"\|--(hidden-features|num-transforms|num-bins)$", "P4"),
    (r"\|(JointSpaceSpec|SpaceSpec)\.(hidden_features|num_transforms|num_bins)$",
     "P4"),
    (r"\|build_joint_model\.(hidden_features|num_transforms|num_bins|"
     r"z_score_theta|z_score_x)$", "P4"),
    (r"\|KNOB_ORDER$", "P4"),
    # --- optimiser and schedule (P5) ----------------------------------------
    (r"\|--(lr|weight-decay|one-minus-beta1|epochs|steps-per-epoch|b-sim|"
     r"seed)$", "P5"),
    (r"\|(EPOCHS|STEPS_PER_EPOCH|B_SIM|SEED)$", "P5"),
    (r"\|TrainConfig\.__init__\.", "P5"),
    (r"\|(JointSpaceSpec|SearchConfig|SpaceSpec|TrainConfig|RegularizationConfig)"
     r"\.(lr|one_minus_beta1|one_minus_beta2|weight_decay|batch_size_npe|"
     r"batch_sizes|learning_rate|batch_size|max_epochs|patience|grad_clip)"
     r"(_range)?$", "P5"),
    (r"\|BatchSpec\.__init__\.b_sim$", "P5"),
    (r"\|grouped_split\.fracs$", "P5"),
    (r"\|evaluate_npe\.batch_size$", "P5"),
    (r"\|FixedStatsSummary\.N_STATS$", "P5"),
    # --- search driver (P6) --------------------------------------------------
    (r"\|--(n-points|n-initial-points|sigma-seed|top|top-k|rank-split|"
     r"gate-split|n-control|n-seeds|alpha|floor|delta-min-provisional|"
     r"p|d-theta|embedding-dim|n-train)$", "P6"),
    (r"\|(P|EMBEDDING_DIM|D_THETA|NPE_ADAPTER)$", "P6"),
    (r"\|(JOINT_KNOB_ORDER|BLOCKS|RANGE_PROVENANCE|INACTIVE_CANONICAL|"
     r"_INT_AXES|_STR_AXES|_DSN_AXES_OFF|_REP_AXES_OFF|CAMPAIGNS|"
     r"SHUFFLE_FIELDS|_NO_BOUNDARY|AXIS_TO_FLAG|DERIVED|UNREACHABLE|"
     r"ARMS)$", "P6"),
    (r"\|JointSpaceSpec\.(fixed|anchored_to)$", "P6"),
    (r"\|Campaign\.", "P6"),
    (r"\|SpaceSpec\.anchored_to$", "P6"),
    (r"\|default_joint_space\.(strict_semihard|n_posterior_draws_max)$", "P6"),
    (r"\|boundary_axes\.rel_tol$", "P6"),
    (r"\|build_argv\.(epochs|steps_per_epoch)$", "P6"),
    (r"\|apply_control_shuffle\.", "P6"),
    (r"\|(VALUE_SHORT)$", "plumbing"),
    # --- upstream bank, bench, stage 3b/3c, jobs (P7) ------------------------
    (r"\|joint/stage1/.*\|--", "P7"),
    (r"\|joint/stage1/.*\|(NuisanceSpec|GapSpec|RealisationSpec|LatentSBISpec)"
     r"\.__init__\.", "P7"),
    (r"\|joint/stage1/.*\|([A-Za-z_][A-Za-z0-9_]*\.)?_?[A-Z][A-Z0-9_]*$", "P7"),
    (r"\|joint/stage1/jobs/.*\|", "P7"),
    (r"\|joint/stage3b/.*\|", "P7"),
    (r"\|joint/stage3c/.*\|", "P7"),
    (r"\|joint/stage3/probe_dsn_runtime\.py\|", "P7"),
    (r"\|joint/stage3/report_joint_arms\.py\|", "P7"),
    (r"\|joint/stage3/jobs/.*\|", "P7"),
    (r"\|joint/stage4/jobs/.*\|", "P7"),
]


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _unparse(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return " ".join(ast.unparse(node).split())
    except Exception:  # pragma: no cover - ast.unparse is 3.9+
        return "<unrenderable>"


def _const_str(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _read(path: str) -> str:
    with open(path, "r", encoding="ascii", errors="strict") as fh:
        return fh.read()


def _parse(path: str) -> ast.Module:
    return ast.parse(_read(path), filename=path)


def _call_name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def _cli_row(call: ast.Call, relpath: str, context: str,
             function: str) -> Optional[Dict[str, Any]]:
    """One add_argument(...) call -> a row, or None for positionals."""
    flags = [a.value for a in call.args
             if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    if not flags or not flags[0].startswith("-"):
        return None
    flag = sorted(flags, key=len)[-1]             # the long form
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    dest = _const_str(kw.get("dest")) or flag.lstrip("-").replace("-", "_")
    action = _const_str(kw.get("action"))
    if action in ("store_true", "store_false"):
        typ = "flag"
        default = "False" if action == "store_true" else "True"
    else:
        typ = _unparse(kw.get("type")) or "str"
        default = _unparse(kw["default"]) if "default" in kw else ""
    choices = _unparse(kw.get("choices"))
    required = _unparse(kw.get("required")) == "True"
    help_text = _const_str(kw.get("help")) or ""
    if isinstance(kw.get("help"), ast.BinOp):
        help_text = _unparse(kw["help"])
    help_text = help_text.split("\n")[0].strip()
    return {
        "surface": "cli", "file": relpath, "name": flag, "dest": dest,
        "context": context, "function": function, "type": typ,
        "default": default, "choices": choices, "required": required,
        "help": help_text[:160], "line": call.lineno,
    }


def _walk_stmts(stmts: Sequence[ast.stmt], relpath: str, function: str,
                context: str, out: List[Dict[str, Any]]) -> str:
    """Statement-ordered walk that tracks the active subparser context.

    An assignment or expression whose value calls add_parser("<name>") sets
    the context for the statements that follow it in the same body; an `if`
    whose test is a bare name (the `need_shards` pattern) suffixes the
    context for its body only. Returns the context in force at the end.
    """
    for st in stmts:
        new_ctx = None
        for node in ast.walk(st):
            if isinstance(node, ast.Call) and _call_name(node) == "add_parser":
                name = _const_str(node.args[0]) if node.args else None
                if name:
                    new_ctx = name
        if isinstance(st, ast.If) and isinstance(st.test, ast.Name):
            inner = context + "[" + st.test.id + "]"
            _walk_stmts(st.body, relpath, function, inner, out)
            _walk_stmts(st.orelse, relpath, function, context, out)
            continue
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _walk_stmts(st.body, relpath, st.name, st.name, out)
            continue
        if isinstance(st, (ast.For, ast.While, ast.With, ast.Try)):
            bodies = [getattr(st, "body", [])] + [getattr(st, "orelse", [])]
            if isinstance(st, ast.Try):
                bodies += [h.body for h in st.handlers] + [st.finalbody]
            for b in bodies:
                _walk_stmts(b, relpath, function, context, out)
            continue
        if new_ctx is not None:
            context = new_ctx
        for node in ast.walk(st):
            if isinstance(node, ast.Call) and _call_name(node) == "add_argument":
                row = _cli_row(node, relpath, context, function)
                if row is not None:
                    out.append(row)
    return context


def extract_cli(path: str, relpath: str) -> List[Dict[str, Any]]:
    tree = _parse(path)
    out: List[Dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _walk_stmts(node.body, relpath, node.name, "", out)
        elif isinstance(node, ast.If):            # the __main__ block
            _walk_stmts(node.body, relpath, "__main__", "", out)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _walk_stmts(sub.body, relpath, node.name + "." + sub.name,
                                "", out)
    return out


# ---------------------------------------------------------------------------
# signatures
# ---------------------------------------------------------------------------

def _find_function(tree: ast.Module, qualname: str) -> Optional[ast.FunctionDef]:
    parts = qualname.split(".")
    scope: Sequence[ast.stmt] = tree.body
    node: Optional[ast.AST] = None
    for i, part in enumerate(parts):
        node = None
        for st in scope:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and st.name == part:
                node = st
                break
        if node is None:
            return None
        if i < len(parts) - 1:
            scope = getattr(node, "body", [])
    return node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


def extract_signatures(path: str, relpath: str,
                       qualnames: Sequence[str]) -> List[Dict[str, Any]]:
    tree = _parse(path)
    out: List[Dict[str, Any]] = []
    for q in qualnames:
        fn = _find_function(tree, q)
        if fn is None:
            out.append({"surface": "signature", "file": relpath, "name": q,
                        "param": "", "default": "<MISSING FUNCTION>",
                        "required": False, "line": 0})
            continue
        a = fn.args
        positional = list(a.posonlyargs) + list(a.args)
        n_def = len(a.defaults)
        for i, arg in enumerate(positional):
            if arg.arg in ("self", "cls"):
                continue
            j = i - (len(positional) - n_def)
            default = _unparse(a.defaults[j]) if j >= 0 else ""
            out.append({"surface": "signature", "file": relpath,
                        "name": q + "." + arg.arg, "param": arg.arg,
                        "default": default, "required": j < 0,
                        "line": fn.lineno})
        for arg, d in zip(a.kwonlyargs, a.kw_defaults):
            out.append({"surface": "signature", "file": relpath,
                        "name": q + "." + arg.arg, "param": arg.arg,
                        "default": _unparse(d), "required": d is None,
                        "line": fn.lineno})
    return out


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------

def _is_dataclass(cls: ast.ClassDef) -> bool:
    for d in cls.decorator_list:
        name = d.id if isinstance(d, ast.Name) else (
            d.func.id if isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
            else "")
        if name == "dataclass":
            return True
    return False


def extract_dataclasses(path: str, relpath: str,
                        classes: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
    tree = _parse(path)
    out: List[Dict[str, Any]] = []
    for want, pattern in classes.items():
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == want), None)
        if cls is None:
            out.append({"surface": "dataclass", "file": relpath, "name": want,
                        "field": "", "default": "<MISSING CLASS>",
                        "annotation": "", "line": 0})
            continue
        rx = re.compile(pattern) if pattern else None
        for st in cls.body:
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                field = st.target.id
                if rx is not None and not rx.search(field):
                    continue
                out.append({"surface": "dataclass", "file": relpath,
                            "name": want + "." + field, "field": field,
                            "default": _unparse(st.value),
                            "annotation": _unparse(st.annotation),
                            "dataclass": _is_dataclass(cls), "line": st.lineno})
    return out


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

def extract_constants(path: str, relpath: str) -> List[Dict[str, Any]]:
    tree = _parse(path)
    out: List[Dict[str, Any]] = []

    def _take(st: ast.stmt, owner: str) -> None:
        targets: List[ast.expr] = []
        value: Optional[ast.expr] = None
        if isinstance(st, ast.Assign):
            targets, value = list(st.targets), st.value
        elif isinstance(st, ast.AnnAssign) and st.value is not None:
            targets, value = [st.target], st.value
        for t in targets:
            if isinstance(t, ast.Name) and ALLCAPS.match(t.id) \
                    and t.id != "__all__":
                out.append({"surface": "constant", "file": relpath,
                            "name": (owner + "." if owner else "") + t.id,
                            "default": _unparse(value), "line": st.lineno})

    for st in tree.body:
        _take(st, "")
        if isinstance(st, ast.ClassDef):
            for sub in st.body:
                _take(sub, st.name)
    return out


# ---------------------------------------------------------------------------
# job scripts (bash / PBS)
# ---------------------------------------------------------------------------

_RX_DEFAULT = re.compile(r'^\s*:\s*"\$\{([A-Za-z_][A-Za-z0-9_]*):=(.*)\}"\s*$')
_RX_ASSIGN = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)="?\$\{([A-Za-z0-9_]+):-(.*?)\}"?\s*$')
_RX_ARRAY = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=\((.*)\)\s*$')
_RX_PBS = re.compile(r'^#PBS\s+(-[A-Za-z])\s+(.*?)\s*$')


def extract_job_script(path: str, relpath: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="ascii", errors="strict") as fh:
        for i, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            m = _RX_PBS.match(line)
            if m:
                out.append({"surface": "job_var", "file": relpath,
                            "name": "PBS_" + m.group(1), "kind": "directive",
                            "default": m.group(2), "line": i})
                continue
            m = _RX_DEFAULT.match(line)
            if m:
                out.append({"surface": "job_var", "file": relpath,
                            "name": m.group(1), "kind": "-v default",
                            "default": m.group(2), "line": i})
                continue
            m = _RX_ASSIGN.match(line)
            if m:
                src = m.group(2)
                kind = ("positional $" + src if src.isdigit()
                        else "env default")
                out.append({"surface": "job_var", "file": relpath,
                            "name": m.group(1), "kind": kind,
                            "default": m.group(3), "line": i})
                continue
            m = _RX_ARRAY.match(line)
            if m and ALLCAPS.match(m.group(1)):
                out.append({"surface": "job_var", "file": relpath,
                            "name": m.group(1), "kind": "array",
                            "default": " ".join(m.group(2).split()),
                            "line": i})
    return out


# ---------------------------------------------------------------------------
# owners, extraction driver, rendering, checking
# ---------------------------------------------------------------------------

def assign_owners(rows: List[Dict[str, Any]],
                  rules: Sequence[Tuple[str, str]] = OWNER_RULES) -> List[str]:
    """Fill `owner` in place; return the keys that matched no rule."""
    compiled = [(re.compile(rx), owner) for rx, owner in rules]
    unowned: List[str] = []
    for r in rows:
        key = "%s|%s|%s" % (r["surface"], r["file"], r["name"])
        owner = next((o for rx, o in compiled if rx.search(key)), None)
        r["owner"] = owner or "UNOWNED"
        if owner is None:
            unowned.append(key)
    return unowned


def _sort_key(r: Dict[str, Any]) -> Tuple[Any, ...]:
    order = {"cli": 0, "signature": 1, "dataclass": 2, "constant": 3,
             "job_var": 4}
    return (order.get(r["surface"], 9), r["file"], int(r.get("line", 0)),
            r["name"])


def extract_all(hpc_dir: str, targets: Dict[str, Any] = TARGETS) -> Tuple[
        List[Dict[str, Any]], List[str]]:
    """Every row of every surface, sorted, plus the list of missing files."""
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    def _p(rel: str) -> Optional[str]:
        full = os.path.join(hpc_dir, rel)
        if not os.path.isfile(full):
            missing.append(rel)
            return None
        return full

    for rel in targets.get("cli", []):
        full = _p(rel)
        if full:
            rows += extract_cli(full, rel)
    for rel, names in targets.get("signature", {}).items():
        full = _p(rel)
        if full:
            rows += extract_signatures(full, rel, names)
    for rel, classes in targets.get("dataclass", {}).items():
        full = _p(rel)
        if full:
            rows += extract_dataclasses(full, rel, classes)
    for rel in targets.get("constant", []):
        full = _p(rel)
        if full:
            rows += extract_constants(full, rel)
    for rel in targets.get("job", []):
        full = _p(rel)
        if full:
            rows += extract_job_script(full, rel)
    rows.sort(key=_sort_key)
    return rows, missing


def _short(value: str, n: int = VALUE_SHORT) -> str:
    value = value.replace("|", "\\|")
    return value if len(value) <= n else value[:n - 3] + "..."


def knob_name(r: Dict[str, Any]) -> str:
    """The knob a row configures, normalised across surfaces: a CLI flag by
    its dest, a job variable in lower case, a parameter / field / constant by
    its last dotted component without a `_range` / `_choices` suffix."""
    n = r["name"]
    if r["surface"] == "cli":
        return str(r.get("dest") or n.lstrip("-").replace("-", "_"))
    if r["surface"] == "job_var":
        return n.lower()
    last = n.split(".")[-1]
    for suffix in ("_range", "_choices"):
        if last.endswith(suffix):
            last = last[: -len(suffix)]
    return last


def knob_variants(rows: List[Dict[str, Any]]) -> List[Tuple[str, List[Tuple[str, str, str]]]]:
    """Knobs that carry more than one distinct default across surfaces.

    Returns [(knob, [(file, name, default), ...])], sorted, listing only the
    knobs whose defaults disagree. Rows without a default (required params,
    empty job defaults) and PBS directives are left out: a difference of
    absence is not a difference of value.
    """
    groups: Dict[str, Dict[Tuple[str, str, str], None]] = {}
    for r in rows:
        if r["surface"] == "signature" and r.get("required"):
            continue
        if r["surface"] == "job_var" and r.get("kind") == "directive":
            continue
        d = str(r.get("default", ""))
        if d == "":
            continue
        groups.setdefault(knob_name(r), {})[(r["file"], r["name"], d)] = None
    out: List[Tuple[str, List[Tuple[str, str, str]]]] = []
    for k in sorted(groups):
        items = sorted(groups[k])
        if len({d for _, _, d in items}) > 1:
            out.append((k, items))
    return out


def render_markdown(rows: List[Dict[str, Any]], missing: Sequence[str] = (),
                    commit: str = "") -> str:
    """The table block of 00_INDEX.md: one table per surface, sorted."""
    lines: List[str] = []
    lines.append("Generated by `tools/inventory_joint_knobs.py`"
                 + (" at commit `%s`" % commit if commit else "")
                 + "; %d rows. Do not edit by hand: regenerate, then run "
                 "`--check-index`. Required (default-less) function "
                 "parameters are kept in the JSON only." % len(rows))
    if missing:
        lines.append("")
        lines.append("MISSING FILES: " + ", ".join(missing))
    by_surface: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r["surface"] == "signature" and r.get("required"):
            continue
        by_surface.setdefault(r["surface"], []).append(r)
    specs = [
        ("cli", "CLI flags", ["file", "name", "context", "type", "default",
                              "choices", "required", "owner"]),
        ("signature", "Function and `__init__` keyword defaults",
         ["file", "name", "default", "owner"]),
        ("dataclass", "Dataclass fields",
         ["file", "name", "annotation", "default", "owner"]),
        ("constant", "Module and class constants",
         ["file", "name", "default", "owner"]),
        ("job_var", "Job-script variables and directives",
         ["file", "name", "kind", "default", "owner"]),
    ]
    for key, title, cols in specs:
        sub = by_surface.get(key, [])
        lines.append("")
        lines.append("### %s (%d)" % (title, len(sub)))
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r in sub:
            cells = []
            for c in cols:
                v = r.get(c, "")
                v = "yes" if v is True else ("" if v is False else str(v))
                if c in ("default", "choices", "annotation"):
                    v = "`" + _short(v) + "`" if v else ""
                elif c in ("name",):
                    v = "`" + v + "`"
                cells.append(v)
            lines.append("| " + " | ".join(cells) + " |")
    variants = knob_variants(rows)
    lines.append("")
    lines.append("### Same knob, different defaults (%d knobs)" % len(variants))
    lines.append("")
    lines.append("One normalised knob name, every place it carries a default, "
                 "where the defaults disagree. The runner's flag wins at run "
                 "time wherever the runner passes the value explicitly; the "
                 "library default is what a direct caller gets. Grouping is "
                 "by name alone, so a row can be a name collision rather than "
                 "one knob: read it as a prompt, not a verdict.")
    lines.append("")
    lines.append("| knob | file | name | default |")
    lines.append("|---|---|---|---|")
    for k, items in variants:
        for i, (f, n, d) in enumerate(items):
            lines.append("| %s | %s | `%s` | `%s` |"
                         % ("`" + k + "`" if i == 0 else "", f, n, _short(d)))
    return "\n".join(lines) + "\n"


def render_json(rows: List[Dict[str, Any]], missing: Sequence[str] = (),
                commit: str = "") -> str:
    payload = {"commit": commit, "n_rows": len(rows),
               "missing_files": list(missing), "rows": rows}
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def check_index(index_text: str, rendered: str) -> Tuple[bool, str]:
    """Compare the block between the markers of 00_INDEX.md with `rendered`."""
    a, b = index_text.find(INDEX_BEGIN), index_text.find(INDEX_END)
    if a < 0 or b < 0 or b < a:
        return False, "markers %s / %s not found in order" % (INDEX_BEGIN,
                                                               INDEX_END)
    block = index_text[a + len(INDEX_BEGIN):b].strip("\n")
    want = rendered.strip("\n")
    # The commit stamp on the first line is provenance, not content: a check
    # run without --commit, or after a push gave the block a new id, must not
    # fail on it. Everything else is compared exactly.
    got_l, want_l = block.split("\n"), want.split("\n")
    if got_l and want_l:
        got_l[0] = _STAMP.sub("", got_l[0])
        want_l[0] = _STAMP.sub("", want_l[0])
    if got_l == want_l:
        return True, "index block matches the extraction"
    for i, (x, y) in enumerate(zip(got_l, want_l), 1):
        if x != y:
            return False, ("first difference at block line %d:\n  index : %s\n"
                           "  fresh : %s" % (i, x, y))
    return False, ("block lengths differ: index %d lines, fresh %d lines"
                   % (len(got_l), len(want_l)))


# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hpc-dir", required=True,
                    help="the repository's hpc/ directory")
    ap.add_argument("--commit", default="",
                    help="commit id stamped into the outputs")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--check-index", default=None,
                    help="00_INDEX.md to compare against a fresh extraction")
    ap.add_argument("--list-unowned", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="exit 2 on any unowned row or missing file")
    args = ap.parse_args(argv)

    rows, missing = extract_all(args.hpc_dir)
    unowned = assign_owners(rows)
    rc = 0
    if missing:
        print("missing files (%d): %s" % (len(missing), ", ".join(missing)))
        rc = 2 if args.strict else rc
    if unowned or args.list_unowned:
        print("unowned rows (%d):" % len(unowned))
        for k in unowned:
            print("  " + k)
        if unowned and args.strict:
            rc = 2
    md = render_markdown(rows, missing, args.commit)
    if args.out_json:
        with open(args.out_json, "w", encoding="ascii") as fh:
            fh.write(render_json(rows, missing, args.commit))
    if args.out_md:
        with open(args.out_md, "w", encoding="ascii") as fh:
            fh.write(md)
    if args.check_index:
        with open(args.check_index, "r", encoding="ascii") as fh:
            ok, msg = check_index(fh.read(), md)
        print(("OK: " if ok else "FAIL: ") + msg)
        if not ok:
            rc = 1
    print("rows: %d  (cli %d, signature %d, dataclass %d, constant %d, job_var %d)"
          % (len(rows), *[sum(1 for r in rows if r["surface"] == s)
                          for s in ("cli", "signature", "dataclass",
                                    "constant", "job_var")]))
    return rc


if __name__ == "__main__":
    sys.exit(main())
