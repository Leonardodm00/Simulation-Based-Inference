#!/usr/bin/env python3
"""Block 35: one ensemble of a recorded project -- its densities at the
held-out simulations' own parameters, its posterior draws on the draw rows
and on every recorded window, their densities, each member's embeddings
and the gate's roles (specs/SPEC.md sec. 2.6 and block 35; decisions D-159,
D-160, D-162, D-163).

    python3 hpc/pipeline/evaluation/ev_rec_sample.py --config <tag>.json \\
        --eval-config <tag>_rec_eval.json --ensemble <name> [--out DIR] [--ncpus N]

Writes under <out_root>/samples/ (or --out), each file to a temporary name
first:
    <stem>_sim.npz      D_sim, the report rows of the members' split
    <stem>_<r>.npz      each recording bank r of the ensemble
    <stem>_sample.json  the provenance, written last
Pass line: REC SAMPLE OK: <stem> ...; exit 2 on the arguments, 3 on a
refusal, which writes nothing.

CONTRACT (wave B): the constants and the signatures below are fixed by
specs/SPEC.md block 35. `run` and `main` are stubs.

Pure ASCII, LF only.
"""

import _evalpath  # noqa: F401

EXIT_ARGS, EXIT_REFUSE = 2, 3
SCHEMA = "rec_eval_sample/1"

# <stem>_sim.npz: key -> (shape, dtype), with n the rows of D_sim, n_d the
# draw rows, n_z the rows with an embedding (the draw rows and the gate
# rows), M the members, p the axes, d_z the embedding's dimension
SIM_ARRAYS = {
    "rows": ("(n,)", "int64"),                    # bank rows, ascending
    "donor": ("(n,)", "int64"),
    "well": ("(n,)", "int64"),
    "cls": ("(n,)", "int64"),
    "theta_ref": ("(n, p)", "float64"),
    "in_box": ("(n,)", "bool"),
    "logq_ref_members": ("(n, M)", "float64"),    # -inf where not in the box
    "logq_ref": ("(n,)", "float64"),
    "calib": ("(n,)", "bool"),
    "draw": ("(n,)", "bool"),
    "gate_role": ("(n,)", "int8"),                # ev_gate.ROLE_*
    "draw_idx": ("(n_d,)", "int64"),              # indices into the n rows, ascending
    "draws": ("(n_d, n_post, p)", "float32"),
    "draw_member": ("(n_post,)", "int64"),
    "logq_draws": ("(n_d, n_post)", "float32"),
    "logq_draws_own": ("(n_d, n_post)", "float32"),
    "stats": ("(n_d, 8)", "float32"),
    "z_idx": ("(n_z,)", "int64"),                 # indices into the n rows, ascending
    "z": ("(M, n_z, d_z)", "float32"),
}

# <stem>_<r>.npz: n the rows of the recording bank, in bank order. No
# theta_ref, in_box or logq_ref* (D-162).
REC_ARRAYS = {
    "rows": ("(n,)", "int64"),
    "cls": ("(n,)", "int64"),
    "donor": ("(n,)", "int64"),
    "well": ("(n,)", "int64"),
    "batch": ("(n,)", "int64"),
    "subregion": ("(n,)", "int64"),
    "window_idx": ("(n,)", "int64"),
    "rep_group": ("(n,)", "int64"),
    "contaminated": ("(n,)", "bool"),
    "draws": ("(n, n_post, p)", "float32"),
    "draw_member": ("(n_post,)", "int64"),
    "logq_draws": ("(n, n_post)", "float32"),
    "logq_draws_own": ("(n, n_post)", "float32"),
    "z": ("(M, n, d_z)", "float32"),
    "stats": ("(n, 8)", "float32"),
}

# the keys of <stem>_sample.json (block 35, "the keys")
PROVENANCE_KEYS = (
    "schema", "ensemble", "stem", "search", "campaign", "arm", "folder", "view_kind",
    "members", "banks", "bank_globs", "sim_bank", "best_fit_complete_sha256", "sample_params",
    "seeds", "oracle_max_abs_nats", "oracle_mean_abs_nats", "oracle_tol", "gate", "n",
    "n_draw", "n_z", "M", "n_post", "p", "axes", "files", "eval_config", "config",
    "config_sha256", "seconds", "ncpus", "torch_threads")


def run(cfg_path, ecfg_path, name, out=None, ncpus=None, log=print):
    """Samples ensemble `name` of the project at `cfg_path` under the
    evaluation's configuration at `ecfg_path`; writes the files under
    <out_root>/samples/ (or `out`); returns the provenance dict
    (PROVENANCE_KEYS). ev_plan.RefuseError on a refusal, nothing written."""
    raise NotImplementedError("block 35")


def main(argv=None):
    """The command line: 0 with the pass line, EXIT_ARGS, EXIT_REFUSE."""
    raise NotImplementedError("block 35")


if __name__ == "__main__":
    import sys
    sys.exit(main())
