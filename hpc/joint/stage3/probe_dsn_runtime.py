#!/usr/bin/env python3
"""
probe_dsn_runtime.py -- does the DSN half of the joint stack RESOLVE at run
time, on a compute node, in the environment a real job would use?

Why this exists
---------------
The question "does condition_space import and does the real DSN backbone get
built" was previously only answerable by launching a real Stage 4 trial. That
does not work, for reasons that have nothing to do with the DSN:

  run_joint_arms.py main() loads the latent bank at line 334, and its own
  --dry-run returns at line 347. The DSN block is at line 396 and
  make_backbone at line 442. So EVERYTHING about the DSN sits downstream of a
  Stage 1 latent bank that has never been built, and downstream of the
  dry-run return as well.

A shard-free probe separates the two questions. This file answers "does the
DSN code path work in this environment", and it needs no shards, no pending
directory, no latent bank, and no training.

Deliberate deviation, stated rather than hidden
-----------------------------------------------
This probe does NOT import run_joint_arms. Doing so would pull in sbi, zuko,
joint_train and the whole Stage 2 stack, so an unrelated failure anywhere in
that chain would look like a DSN failure. Instead it MIRRORS the resolution
logic, line for line, with the source line numbers named at each check. That
mirroring is a real risk: if run_joint_arms changes how it resolves the DSN,
this probe can pass while the trainer fails. Re-read the referenced lines when
either file changes. The alternative (importing the trainer) trades this risk
for a worse one, so the mirror is the lesser evil, not a free choice.

What a PASS means
-----------------
It means: in THIS environment, on THIS node, condition_space imports, the
legality projection runs, the real DSN backbone builds and produces an
embedding of the right shape, and the DSN loss constructs. It does NOT mean
any training is correct, and it does NOT mean a latent bank exists.

Pass condition is a line that must APPEAR:

    [probe] VERDICT: PASS (7/7 checks)

Never "no [warn] lines", which is also what a crash before the DSN block
produces.

Pure ASCII, LF only.
"""

from __future__ import annotations

import argparse
import os
import sys

# Same sys.path block run_joint_arms.py uses (its lines 48-53), minus the
# stage1 entry: this probe never touches the latent bank. Living next to
# run_joint_arms.py is deliberate -- the mirroring risk in the docstring is
# only manageable if the two files are read together.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "stage2"), _HERE):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)


class Probe(object):
    """Accumulates PASS/FAIL so every check runs even after one fails.

    Stopping at the first failure would hide the second, and the whole point
    of a probe is to spend one queue slot and learn everything.
    """

    def __init__(self):
        self.results = []

    def record(self, cid, ok, detail):
        self.results.append((cid, bool(ok), detail))
        print("[probe] %s %s  %s" % (cid, "PASS" if ok else "FAIL", detail),
              flush=True)
        return bool(ok)

    def n_pass(self):
        return sum(1 for _, ok, _ in self.results if ok)

    def total(self):
        return len(self.results)

    def verdict(self):
        n, t = self.n_pass(), self.total()
        ok = (n == t and t > 0)
        print("[probe] VERDICT: %s (%d/%d checks)"
              % ("PASS" if ok else "FAIL", n, t), flush=True)
        if not ok:
            for cid, good, detail in self.results:
                if not good:
                    print("[probe]   failed: %s  %s" % (cid, detail),
                          flush=True)
        return 0 if ok else 1


def check_torch(p):
    """P1: torch imports. make_backbone calls torch.manual_seed first."""
    try:
        import torch
        return p.record("P1", True, "torch %s" % torch.__version__)
    except Exception as exc:
        return p.record("P1", False, "torch does not import: %r" % (exc,))


def check_pml(p):
    """P2: pytorch_metric_learning imports.

    build_loss_and_miner in the DSN's train.py needs it. It is not a
    dependency of anything else in the joint stack, so a missing install here
    fails only at DSN-loss construction time, deep into a real run.
    """
    try:
        import pytorch_metric_learning as pml
        return p.record("P2", True,
                        "pytorch_metric_learning %s" % pml.__version__)
    except Exception as exc:
        return p.record("P2", False,
                        "pytorch_metric_learning does not import: %r" % (exc,))


def resolve_dsn_dir(p, explicit):
    """P3: DSN_MAIN_DIR resolves to a real directory.

    Mirrors run_joint_arms.py:292 and :416, which both do
    `dsn_main_dir or os.environ.get("DSN_MAIN_DIR")`.

    The three failure states are reported separately on purpose. An unset
    variable and a variable pointing at a dead symlink previously produced the
    same message, which cost a session: $HOME/dsn_main had been deleted and
    the suites reported it as "unset".
    """
    d = explicit or os.environ.get("DSN_MAIN_DIR")
    if not d:
        p.record("P3", False,
                 "DSN_MAIN_DIR is UNSET and --dsn-main-dir was not given")
        return None
    if not os.path.exists(d):
        p.record("P3", False,
                 "DSN_MAIN_DIR=%r points at a path that DOES NOT EXIST "
                 "(dead symlink?)" % d)
        return None
    if not os.path.isdir(d):
        p.record("P3", False, "DSN_MAIN_DIR=%r exists but is not a directory"
                 % d)
        return None
    real = os.path.realpath(d)
    p.record("P3", True, "DSN_MAIN_DIR=%r -> %r" % (d, real))
    return d


REQUIRED_DSN_FILES = ("condition_space.py", "backbone.py", "train.py")
INFORMATIONAL_DSN_FILES = ("latent_burst_generator.py",)


def check_dsn_files(p, dsn_dir):
    """P4: the three DSN modules the joint stack actually imports are present.

    condition_space.py  -> run_joint_arms.py:420 legality projection
    backbone.py         -> run_joint_arms.py:296 real encoder
    train.py            -> dsn_loss_adapter.load_dsn_train_module
    """
    if dsn_dir is None:
        return p.record("P4", False, "skipped: no DSN directory to check")
    missing = [f for f in REQUIRED_DSN_FILES
               if not os.path.isfile(os.path.join(dsn_dir, f))]
    for f in INFORMATIONAL_DSN_FILES:
        present = os.path.isfile(os.path.join(dsn_dir, f))
        print("[probe]    (informational) %s: %s"
              % (f, "present" if present else "ABSENT"), flush=True)
    if missing:
        return p.record("P4", False, "missing under %r: %s"
                        % (dsn_dir, ", ".join(missing)))
    return p.record("P4", True, "all of %s present under %r"
                    % (", ".join(REQUIRED_DSN_FILES), dsn_dir))


def check_condition_space(p, dsn_dir, mining, loss_type, strict):
    """P5: condition_space imports and project_condition runs.

    Mirrors run_joint_arms.py:416-425 exactly, including the sys.path insert.
    The trainer catches ImportError here and merely WARNS, then trains an
    unprojected (mining, loss, strict) triple. That is the silent-wrong-cell
    failure this probe exists to make loud.
    """
    if dsn_dir is None:
        return p.record("P5", False, "skipped: no DSN directory")
    if dsn_dir not in sys.path:
        sys.path.insert(0, dsn_dir)
    try:
        import condition_space as CS
    except Exception as exc:
        return p.record("P5", False,
                        "import condition_space FAILED: %r "
                        "(the trainer would only [warn] and continue "
                        "UNPROJECTED)" % (exc,))
    try:
        out = CS.project_condition(mining, loss_type, strict)
    except Exception as exc:
        return p.record("P5", False,
                        "condition_space imported but project_condition(%r, "
                        "%r, %r) raised %r" % (mining, loss_type, strict, exc))
    try:
        m2, l2, s2 = out
    except Exception:
        return p.record("P5", False,
                        "project_condition returned %r, expected a "
                        "(mining, loss_type, strict) triple" % (out,))
    changed = ((m2, l2, bool(s2)) != (mining, loss_type, bool(strict)))
    return p.record("P5", True,
                    "project_condition(%r, %r, %r) -> (%r, %r, %r)%s"
                    % (mining, loss_type, strict, m2, l2, bool(s2),
                       "  [PROJECTED, input was illegal]" if changed
                       else "  [input already legal]"))


def check_backbone(p, dsn_dir, embedding_size, window, seed):
    """P6: the REAL backbone builds and returns the right embedding shape.

    Mirrors run_joint_arms.py:292-303. The check that matters is not that
    something was built, but that the DSN branch was taken: make_backbone
    falls back to a small GroupNorm CNN whenever backbone.py is not found, and
    the fallback IGNORES the searched encoder axes. A campaign run that way
    would record a cell it never trained.
    """
    if dsn_dir is None:
        return p.record("P6", False, "skipped: no DSN directory")
    if not os.path.isfile(os.path.join(dsn_dir, "backbone.py")):
        return p.record("P6", False,
                        "no backbone.py under %r: make_backbone WOULD SILENTLY "
                        "FALL BACK to the GroupNorm CNN" % dsn_dir)
    if dsn_dir not in sys.path:
        sys.path.insert(0, dsn_dir)
    try:
        import torch
        from backbone import BackboneConfig, build_backbone
    except Exception as exc:
        return p.record("P6", False, "importing backbone failed: %r" % (exc,))
    torch.manual_seed(int(seed))
    try:
        # Exactly the defaults make_backbone uses when an encoder axis is
        # absent (run_joint_arms.py:297-303).
        net = build_backbone(BackboneConfig(
            depth_exponent=3,
            width_multiplier=2.0,
            block_family=0,
            head_fusion=False,
            dropout=0.0,
            stem_width=16,
            embedding_size=int(embedding_size)))
    except Exception as exc:
        return p.record("P6", False, "build_backbone raised %r" % (exc,))
    try:
        with torch.no_grad():
            z = net(torch.zeros(2, int(window)))
    except Exception as exc:
        return p.record("P6", False,
                        "backbone built but the forward pass on a (2, %d) "
                        "input raised %r" % (int(window), exc))
    shape = tuple(z.shape)
    want = (2, int(embedding_size))
    if shape != want:
        return p.record("P6", False,
                        "backbone returned shape %r, expected %r"
                        % (shape, want))
    n_par = sum(int(t.numel()) for t in net.parameters())
    return p.record("P6", True,
                    "real DSN backbone built (%d parameters), forward "
                    "(2, %d) -> %r" % (n_par, int(window), shape))


def check_dsn_loss(p, dsn_dir, n_classes, total_steps, mining, loss_type,
                   strict, margin, angular_alpha_deg, lambda_sep,
                   sep_warmup_frac):
    """P7: the DSN loss constructs through the Stage 2 adapter.

    Mirrors run_joint_arms.py:428-437. Prints the resolved DSNLossConfig, so
    the log carries the same [dsn-loss] evidence a real run would.
    """
    if dsn_dir is None:
        return p.record("P7", False, "skipped: no DSN directory")
    try:
        from dsn_loss_adapter import DSNLossConfig, build_dsn_loss
    except Exception as exc:
        return p.record("P7", False,
                        "import dsn_loss_adapter (Stage 2) failed: %r"
                        % (exc,))
    cfg = DSNLossConfig(loss_type=loss_type,
                        mining_strategy=mining,
                        strict_semihard=bool(strict),
                        margin=float(margin),
                        angular_alpha_deg=float(angular_alpha_deg),
                        lambda_sep=float(lambda_sep),
                        sep_warmup_frac=float(sep_warmup_frac))
    print("[dsn-loss] %s" % cfg.to_dict(), flush=True)
    try:
        adapter = build_dsn_loss(int(n_classes), total_steps=int(total_steps),
                                 cfg=cfg, dsn_main_dir=dsn_dir)
    except Exception as exc:
        return p.record("P7", False, "build_dsn_loss raised %r" % (exc,))
    if not callable(adapter):
        return p.record("P7", False,
                        "build_dsn_loss returned %r, which is not callable"
                        % (adapter,))
    return p.record("P7", True,
                    "DSN loss built: %s" % type(adapter).__name__)


def build_parser():
    p = argparse.ArgumentParser(
        description="Probe whether the DSN code path resolves at run time. "
                    "No shards, no latent bank, no training.")
    p.add_argument("--dsn-main-dir", default=None,
                   help="overrides DSN_MAIN_DIR, same precedence as the "
                        "trainer's own flag")
    p.add_argument("--embedding-size", type=int, default=12,
                   help="E. Stage 4's --embedding-dim default")
    p.add_argument("--window", type=int, default=3000,
                   help="W, the window length in samples (T_win * fs)")
    p.add_argument("--n-classes", type=int, default=3,
                   help="C. joint_sep's separation target is -1/(C-1)")
    p.add_argument("--total-steps", type=int, default=2,
                   help="warm-up horizon; only needs to be positive here")
    p.add_argument("--mining-strategy", default="easy_pos_semihard_neg")
    p.add_argument("--loss-type", default="joint_sep")
    p.add_argument("--strict-semihard", type=int, default=1, choices=(0, 1))
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--angular-alpha-deg", type=float, default=18.0)
    p.add_argument("--lambda-sep", type=float, default=0.1)
    p.add_argument("--sep-warmup-frac", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    print("[probe] interpreter : %s" % sys.executable, flush=True)
    print("[probe] python      : %s" % sys.version.split()[0], flush=True)
    print("[probe] conda env   : %s"
          % os.environ.get("CONDA_DEFAULT_ENV", "<none>"), flush=True)
    print("[probe] host        : %s" % os.uname()[1], flush=True)
    print("[probe] cwd         : %s" % os.getcwd(), flush=True)

    p = Probe()
    check_torch(p)
    check_pml(p)
    dsn_dir = resolve_dsn_dir(p, args.dsn_main_dir)
    check_dsn_files(p, dsn_dir)
    check_condition_space(p, dsn_dir, args.mining_strategy, args.loss_type,
                          bool(args.strict_semihard))
    check_backbone(p, dsn_dir, args.embedding_size, args.window, args.seed)
    check_dsn_loss(p, dsn_dir, args.n_classes, args.total_steps,
                   args.mining_strategy, args.loss_type,
                   bool(args.strict_semihard), args.margin,
                   args.angular_alpha_deg, args.lambda_sep,
                   args.sep_warmup_frac)
    return p.verdict()


if __name__ == "__main__":
    sys.exit(main())
