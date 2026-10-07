"""Block 5 (p5_numbers.py): spec-derived tests.

Oracles (SPEC Block 5): `adamw_step` equals torch.optim.AdamW (foreach=False)
step by step on small float64 tensors, including t = 1 and weight decay > 0;
`clip_coef` equals torch's coefficient min(1, max_norm / (total_norm + 1e-6));
the [RAN] numbers of P5 equal what the blocks print. Also: the public helpers
`coverage` and `grouped_split_counts` against their stated definitions (the
latter against the FREEZE `grouped_split`), and the [REPO] constants the script
hard-codes against the FREEZE files they cite (SPEC sec. 3).

Torch: the sandbox has torch 2.14.1 (2.10.0 needs CUDA-12 libraries absent
here). The tester diffed torch 2.10.0's `_single_tensor_adam` and
`clip_grad.py` against 2.14.1's: they differ only in comments, so 2.14.1 is
used as the reference.
"""
import ast
import math
import os
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from conftest import TOOLS, doc_text, git_show, line_with, load_functions, script_output

sys.path.insert(0, TOOLS)
import p5_numbers as p5          # noqa: E402

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def out():
    return script_output("p5_numbers.py")


@pytest.mark.parametrize("wd", [0.0, 1e-2, 0.3])
@pytest.mark.parametrize("lr,b1,b2,eps", [(1e-3, 0.9, 0.999, 1e-8), (5e-2, 0.8, 0.99, 1e-6)])
def test_adamw_step_equals_torch(wd, lr, b1, b2, eps):
    rng = np.random.default_rng(3)
    p0 = rng.standard_normal(7)
    grads = rng.standard_normal((12, 7)) * np.logspace(-3, 2, 7)
    tp = torch.nn.Parameter(torch.tensor(p0, dtype=torch.float64))
    opt = torch.optim.AdamW([tp], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd, foreach=False)
    p, m, v = p0.copy(), np.zeros(7), np.zeros(7)
    for t in range(1, 13):
        tp.grad = torch.tensor(grads[t - 1], dtype=torch.float64)
        opt.step()
        p, m, v = p5.adamw_step(p, grads[t - 1], m, v, t, lr, b1, b2, eps, wd)
        st = opt.state[tp]
        # float64; torch uses lerp for m (one rounding differs) -> 1e-12 relative
        assert_allclose(p, tp.detach().numpy(), rtol=1e-12, atol=1e-15, err_msg="t=%d" % t)
        assert_allclose(m, st["exp_avg"].numpy(), rtol=1e-12, atol=1e-18)
        assert_allclose(v, st["exp_avg_sq"].numpy(), rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("total", [0.0, 1.0, 4.999999, 5.0, 5.000002, 50.0, 500.0, 1e9])
def test_clip_coef_equals_torch(total):
    g = torch.zeros(4, dtype=torch.float64)
    g[0] = total
    w = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    w.grad = g.clone()
    torch.nn.utils.clip_grad_norm_([w], 5.0, foreach=False)
    coef_torch = float(w.grad[0] / total) if total > 0 else 1.0
    expected = min(1.0, 5.0 / (total + 1e-6))
    assert p5.clip_coef(total, 5.0) == pytest.approx(expected, rel=1e-15)
    assert p5.clip_coef(total) == pytest.approx(coef_torch, rel=1e-12)


def test_coverage_definition():
    # expected fraction of N rows seen by n i.i.d. draws: 1 - (1 - 1/N)^n; brute-force MC check
    rng = np.random.default_rng(5)
    N, n = 50, 80
    seen = np.mean([len(np.unique(rng.integers(0, N, n))) / N for _ in range(4000)])
    assert p5.coverage(N, n) == pytest.approx(1 - (1 - 1 / N) ** n, rel=1e-15)
    assert seen == pytest.approx(p5.coverage(N, n), abs=0.005)    # 4000 trials: SE ~ 6e-4


@pytest.mark.parametrize("n_groups", [32, 383, 7])
@pytest.mark.parametrize("seed", [0, 1])
def test_grouped_split_counts_equals_freeze(n_groups, seed):
    import hashlib
    import json
    ns = load_functions(git_show("hpc/joint/stage3/run_joint_arms.py"), ("grouped_split",),
                        {"hashlib": hashlib, "json": json})
    idx, digest = ns["grouped_split"](np.arange(n_groups), seed=seed)
    tr, se, rp, dg = p5.grouped_split_counts(n_groups, seed=seed)
    assert (tr, se, rp) == (int((idx == 0).sum()), int((idx == 1).sum()), int((idx == 2).sum()))
    assert dg == digest


def _parser_default(src, flag):
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == flag):
            for kw in node.keywords:
                if kw.arg == "default":
                    return ast.literal_eval(kw.value)
    raise KeyError(flag)


def test_hardcoded_repo_constants_equal_freeze():
    run = git_show("hpc/joint/stage3/run_joint_arms.py")
    R = p5.RUNNER
    for key, flag in (("epochs", "--epochs"), ("steps_per_epoch", "--steps-per-epoch"),
                      ("b_sim", "--b-sim"), ("b_met", "--b-met"), ("b_rep", "--b-rep"),
                      ("lr", "--lr"), ("weight_decay", "--weight-decay"),
                      ("one_minus_beta1", "--one-minus-beta1"), ("encoder_steps", "--encoder-steps")):
        assert R[key] == _parser_default(run, flag), key
    assert "patience=%d" % p5.RUNNER_PATIENCE in run
    tc_src = git_show("hpc/joint/stage2/joint_train.py")
    for node in ast.walk(ast.parse(tc_src)):
        if isinstance(node, ast.ClassDef) and node.name == "TrainConfig":
            init = [f for f in node.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"][0]
            names = [a.arg for a in init.args.args][-len(init.args.defaults):]
            dflt = {n: ast.literal_eval(d) for n, d in zip(names, init.args.defaults)}
    for k, v in p5.TRAINCONFIG.items():
        assert dflt[k] == v, k
    space = git_show("hpc/joint/stage4/joint_space.py")
    assert "lr: Tuple[float, float] = (1e-4, 2e-3)" in space
    assert "one_minus_beta1: Tuple[float, float] = (1e-2, 1e-1)" in space
    assert "weight_decay: Tuple[float, float] = (1e-5, 1e-2)" in space
    assert "batch_size_npe: Tuple[int, ...] = (256, 512, 1024)" in space
    assert '"one_minus_beta2": 1e-3' in space
    assert p5.SPACE == dict(lr=(1e-4, 2e-3), one_minus_beta1=(1e-2, 1e-1),
                            weight_decay=(1e-5, 1e-2), batch_size_npe=(256, 512, 1024),
                            one_minus_beta2=1e-3)
    cfg = git_show("hpc/dsn/config.py")
    for txt in ("lr_range: Tuple[float, float] = (1e-4, 0.2)",
                "one_minus_beta1_range: Tuple[float, float] = (1e-2, 1e-1)",
                "one_minus_beta2_range: Tuple[float, float] = (1e-4, 1e-2)",
                "weight_decay_range: Tuple[float, float] = (1e-4, 1e-2)",
                "lr: float = 3e-4", "weight_decay: float = 1e-4",
                "max_epochs: int = 100", "patience: int = 10"):
        assert txt in cfg, txt
    pbs = git_show("hpc/joint/stage1/jobs/build_latent_bank.pbs")
    for k, v in (("N_TRACES", 64), ("WELLS_PER_DONOR", 2), ("N_WINDOWS", 8)):
        assert ': "${%s:=%d}"' % (k, v) in pbs
    assert p5.BENCH == dict(n_traces=64, wells_per_donor=2, n_windows=8)
    # torch AdamW defaults (torch's own signature)
    import inspect
    sig = inspect.signature(torch.optim.AdamW.__init__).parameters
    assert sig["lr"].default == p5.TORCH_ADAMW["lr"]
    assert tuple(sig["betas"].default) == p5.TORCH_ADAMW["betas"]
    assert sig["weight_decay"].default == p5.TORCH_ADAMW["weight_decay"]


def test_printed_numbers_recomputed(out):
    assert "half-life =     6.58" in line_with(out, "u1 = 0.1 (runner default")
    assert "%.2f" % (math.log(2) / -math.log(0.9)) == "6.58"
    assert "%.2f" % (1 - 0.99 ** 25) == "0.22"
    assert "1-0.99^t = 0.2222" in line_with(out, "t =    25")
    lr, wd = 2e-3, 1e-2
    assert "%.4f" % (1 - (1 - lr * wd) ** 250) == "0.0050"
    assert "a 0.20 % shrink" in out and "%.2f" % (100 * (1 - (1 - 1e-5) ** 200)) == "0.20"
    assert "1.0000, 1.0000, 0.1000, 0.0100" in out
    assert "coef 0.7071" in out and "%.2f" % (5 / (math.hypot(5, 5) + 1e-6)) == "0.71"
    assert "%.2f" % (5 / (math.hypot(1, 10) + 1e-6)) == "0.50"
    lo, hi = 1e-4, 2e-3
    assert "%.3f" % (math.log(1e-3 / lo) / math.log(hi / lo)) == "0.769"
    assert "%.3f" % (math.log(5e-4 / lo) / math.log(hi / lo)) == "0.537"
    assert "22/5/5 donors = 352/80/80 rows" in out
    assert "%.1f" % (29616 / 512) == "57.8"
    p5d = doc_text("P5_OPTIMISER_AND_SCHEDULE.md")
    for q in ("$29\\,616 / 512 = 57.8$", "77 % below the runner's", "54 % of the mass",
              "$352 / 80 / 80$ rows", "a 0.20 % shrink", "scaled by 0.71", "by 0.50",
              "5 120 / 10 240 / 20 480", "2000 / 2000 curves"):
        assert q in p5d, q
