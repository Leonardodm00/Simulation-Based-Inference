#!/usr/bin/env python3
"""
check_env.py -- verify the conda environment before submitting any job.

Run this once after creating the environment, and again on the cluster the
first time you use it there. It fails loudly rather than letting a missing
or mismatched package surface at job time.

    python check_env.py

Checks, in order of how expensive they are to discover later:
  1. Python version meets the >=3.10 floor that sbi, zuko and torch share.
  2. Every required package imports.
  3. The pinned versions of sbi and zuko match, since the code depends on
     API details that have moved between their releases.
  4. torch reports whether CUDA is actually usable, not merely compiled in.
  5. The three sbi API surfaces this code calls actually exist and accept
     the arguments it passes.
  6. A one-second end-to-end flow build, which catches an ABI mismatch
     between a conda numpy and a pip torch that import checks miss.

ASCII-only by policy (HPC transfer safety).
"""

from __future__ import annotations

import sys

PINNED = {"sbi": "0.27.0", "zuko": "1.6.0"}
REQUIRED = ["numpy", "scipy", "pandas", "pyarrow", "sklearn", "torch", "sbi", "zuko"]

failures = []
warnings = []


def report(ok, label, detail=""):
    print("  [%s] %-22s %s" % ("PASS" if ok else "FAIL", label, detail))


def main() -> int:
    print("=" * 66)
    print("Environment check for the SBI / NPE stage")
    print("=" * 66)

    # -- 1. python ----------------------------------------------------------
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    report(ok, "python >= 3.10", "%d.%d.%d" % (v.major, v.minor, v.micro))
    if not ok:
        failures.append("python %d.%d is below the 3.10 floor" % (v.major, v.minor))
        print("\nFATAL: nothing else can be checked. Recreate the environment.")
        return 1

    # -- 2 and 3. imports and pins -------------------------------------------
    import importlib
    import importlib.metadata as md

    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            report(False, name, "import failed: %s" % exc)
            failures.append("%s does not import" % name)
            continue
        dist = {"sklearn": "scikit-learn"}.get(name, name)
        try:
            ver = md.version(dist)
        except Exception:  # noqa: BLE001
            ver = "?"
        if name in PINNED and ver != PINNED[name]:
            report(False, name, "%s (expected %s)" % (ver, PINNED[name]))
            failures.append("%s is %s, expected the pinned %s" % (name, ver, PINNED[name]))
        else:
            report(True, name, ver)

    if failures:
        print("\nStopping: fix the above before the runtime checks.")
        return 1

    # -- 4. cuda -------------------------------------------------------------
    import torch

    if torch.cuda.is_available():
        report(True, "cuda", "%d device(s), %s, torch cuda %s"
               % (torch.cuda.device_count(), torch.cuda.get_device_name(0),
                  torch.version.cuda))
    else:
        report(True, "cuda", "not available -- CPU only (fine for the smoke tests)")
        warnings.append("no GPU visible; set NPEConfig(device='cpu')")

    # -- 5. sbi api surfaces -------------------------------------------------
    import inspect

    try:
        from sbi.neural_nets import posterior_nn
        from sbi.neural_nets.factory import model_builders
        from sbi.inference import NPE  # noqa: F401
        from sbi.inference.posteriors.ensemble_posterior import EnsemblePosterior

        check_model = "zuko_nsf" in model_builders
        report(check_model, "model 'zuko_nsf'",
               "present" if check_model else "MISSING from model_builders")
        if not check_model:
            failures.append("the 'zuko_nsf' model string is not registered")

        sig = inspect.signature(posterior_nn)
        has_zscore = "z_score_theta" in sig.parameters
        report(has_zscore, "posterior_nn args",
               "z_score_theta present" if has_zscore else "z_score_theta MISSING")
        if not has_zscore:
            failures.append("posterior_nn has no z_score_theta parameter")

        src = inspect.getsource(EnsemblePosterior.log_prob)
        is_mixture = "logsumexp" in src
        report(is_mixture, "ensemble rule",
               "arithmetic mixture (logsumexp)" if is_mixture
               else "NOT logsumexp -- combination rule changed upstream")
        if not is_mixture:
            failures.append("EnsemblePosterior.log_prob no longer uses logsumexp; "
                            "the ensemble would be a product of experts")
    except Exception as exc:  # noqa: BLE001
        report(False, "sbi api", str(exc))
        failures.append("sbi API check raised: %s" % exc)

    # -- 6. end to end build --------------------------------------------------
    try:
        import numpy as np
        from sbi.inference import NPE
        from sbi.utils import BoxUniform

        prior = BoxUniform(low=torch.zeros(2), high=torch.ones(2))
        builder = posterior_nn(model="zuko_nsf", z_score_theta="independent",
                               hidden_features=16, num_transforms=2)
        inf = NPE(prior=prior, density_estimator=builder, show_progress_bars=False)
        rng = np.random.default_rng(0)
        th = rng.uniform(0, 1, size=(256, 2)).astype("float32")
        x = (th @ rng.normal(size=(2, 3)) + 0.1 * rng.normal(size=(256, 3))).astype("float32")
        inf.append_simulations(torch.as_tensor(th), torch.as_tensor(x))
        inf.train(max_num_epochs=2, training_batch_size=64, show_train_summary=False)
        post = inf.build_posterior()
        s = post.sample((16,), x=torch.as_tensor(x[0]), show_progress_bars=False)
        ok = tuple(s.shape) == (16, 2) and bool(torch.isfinite(s).all())
        report(ok, "end-to-end build", "trained and sampled, shape %s" % (tuple(s.shape),))
        if not ok:
            failures.append("end-to-end build produced a bad sample")
    except Exception as exc:  # noqa: BLE001
        report(False, "end-to-end build", str(exc))
        failures.append("end-to-end build raised: %s" % exc)

    # -- summary --------------------------------------------------------------
    print("=" * 66)
    for w in warnings:
        print("  WARN  %s" % w)
    if failures:
        print("  %d PROBLEM(S):" % len(failures))
        for f in failures:
            print("    - %s" % f)
        print("\nEnvironment is NOT ready.")
        return 1
    print("  Environment OK. Next: python smoke_test_npe.py --fast")
    return 0


if __name__ == "__main__":
    sys.exit(main())
