"""Smoke test for the 3-class demonstration scripts.

Run:  python3 smoke_test_demo_classes.py

Covers `demo_classes_generate.py` (class balance, the background cohort,
determinism, and that the class structure really is in the label axes)
and `demo_classes_plot.py` (the feature maps, above all the
shift-invariance that makes the PSD embedding meaningful, and that each
figure file is actually written).

Generation tests need the DSN's compute_ifr_trace and SKIP loudly
without DSN_MAIN_DIR. The feature-map tests are pure numerics and always
run, on a synthetic array -- so the property that matters (shift
invariance) is checked even on a machine with no DSN checkout.

Pure ASCII, LF only.
"""

import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demo_classes_plot import (embed, load_demo, psd_features,        # noqa: E402
                               trace_features)

RESULTS = []


def report(name, status, detail):
    RESULTS.append(status)
    print("[%s] %-54s %s" % (status, name, detail))


def ok(name, cond, detail):
    report(name, "PASS" if cond else "FAIL", detail)


def _dsn_dir():
    d = os.environ.get("DSN_MAIN_DIR")
    if d and os.path.isfile(os.path.join(d, "generate_burst_data.py")):
        return d
    return None


# ---------------------------------------------------------------------------
# D1-D4 -- generation                                        [needs DSN dir]
# ---------------------------------------------------------------------------

def test_generation():
    d = _dsn_dir()
    if d is None:
        for t in ("D1", "D2", "D3", "D4"):
            report(t, "SKIP", "DSN_MAIN_DIR not resolvable")
        return None
    from demo_classes_generate import build_parser, build_spec, generate
    from bench_burst_provider import load_bench_provider

    args = build_parser().parse_args(["--n-windows", "2", "--T-win", "6"])
    spec = build_spec(args)
    prov = load_bench_provider(d)
    out = generate(spec, prov, n_per_class=4, base_seed=3, n_background=5)

    cls = out["cls"]
    counts = [int(np.sum(cls == k)) for k in range(spec.n_classes)]
    ok("D1a the class cohort is balanced",
       counts == [4, 4, 4] and int(np.sum(cls < 0)) == 5,
       "4 per class + 5 background (label -1)")
    ok("D1b phi stays strictly inside the open unit box",
       bool(np.all(out["phi"] > 0.0) and np.all(out["phi"] < 1.0)),
       "a boundary coordinate maps to +/- inf under the logit transform")
    ok("D1c shapes and non-negativity hold with nuisance off",
       out["x"].shape == (17, 2, spec.W) and bool(np.all(out["x"] >= 0.0)),
       "(n, J, W) float64, x >= 0")

    out2 = generate(spec, prov, n_per_class=4, base_seed=3, n_background=5)
    out3 = generate(spec, prov, n_per_class=4, base_seed=4, n_background=5)
    ok("D2a generation is reproducible at fixed seed",
       np.array_equal(out["x"], out2["x"])
       and np.array_equal(out["phi"], out2["phi"]),
       "same seed, same bytes")
    ok("D2b a different seed changes the set",
       not np.array_equal(out["x"], out3["x"]),
       "the realisation draw is live")

    # D3: the class structure must live on the LABEL axes only -- this is
    # what makes the demo a test of eq. (7) rather than of a coincidence.
    lab, free = list(spec.label_idx), list(spec.free_idx)
    sig = out["cls"] >= 0
    within = np.concatenate([out["phi"][sig & (cls == k)]
                             - out["phi"][sig & (cls == k)].mean(axis=0)
                             for k in range(spec.n_classes)])
    spread_lab = within[:, lab].std()
    spread_free = within[:, free].std()
    ok("D3 within-class spread is tighter on the label axes",
       spread_lab < spread_free,
       "label sd %.3f < free sd %.3f (tau_ov = %.2f vs uniform)"
       % (spread_lab, spread_free, spec.tau_ov))

    # D4: the background is uniform over the box, hence NOT class-structured
    bg = out["phi"][cls < 0]
    ok("D4 the background cohort spans the box on every axis",
       bool(bg.min() < 0.35 and bg.max() > 0.65),
       "range [%.2f, %.2f] -- it is the 'full space' the map needs"
       % (bg.min(), bg.max()))
    return out


# ---------------------------------------------------------------------------
# D5-D7 -- feature maps and figures                          [always run]
# ---------------------------------------------------------------------------

def _synthetic(n=12, J=2, W=256, fs=50.0, seed=0):
    """Traces with a burst at a RANDOM time, identical otherwise."""
    rng = np.random.default_rng(seed)
    t = np.arange(W)
    x = np.zeros((n, J, W))
    for i in range(n):
        for j in range(J):
            c = rng.integers(40, W - 40)
            x[i, j] = np.exp(-0.5 * ((t - c) / 6.0) ** 2)
    return x, fs


def test_features():
    x, fs = _synthetic()
    ok("D5a trace_features flattens to (n, J * W)",
       trace_features(x).shape == (12, 2 * 256), "no summary statistic")

    # The property the whole PSD embedding rests on: a time shift must not
    # move a point. Rolling every window by a different amount changes the
    # raw features completely and the spectral features not at all.
    x_rolled = np.stack([[np.roll(x[i, j], 37 * (i + 1) + 11 * j)
                          for j in range(x.shape[1])]
                         for i in range(x.shape[0])])
    f0 = psd_features(x, fs)
    f1 = psd_features(x_rolled, fs)
    raw_rel = (np.abs(trace_features(x_rolled) - trace_features(x)).max()
               / np.abs(trace_features(x)).max())
    ok("D5b psd_features is invariant to a per-window time shift",
       np.allclose(f0, f1, atol=1e-8),
       "max |dPSD| = %.2e, while raw features move by %.2f of full scale"
       % (np.abs(f1 - f0).max(), raw_rel))
    ok("D5c the PSD is finite everywhere",
       bool(np.all(np.isfinite(f0))),
       "the +eps inside the log covers empty windows")

    y, per = embed(f0, perplexity=25.0, pca_dim=0, seed=0)
    ok("D6a embed returns a 2-D map",
       y.shape == (12, 2), "(n, 2)")
    ok("D6b perplexity is clamped to the (n - 1) / 3 rule of thumb",
       per <= (12 - 1) / 3.0 + 1e-9,
       "asked 25, used %.2f on n = 12" % per)


def test_figures(out):
    if out is None:
        report("D7", "SKIP", "needs the generated set from D1-D4")
        return
    from demo_classes_plot import plot_traces, plot_tsne
    meta = {"fs": 50.0, "nuisance": False}
    with tempfile.TemporaryDirectory() as td:
        p1 = plot_traces(out, meta, os.path.join(td, "full.png"), n_show=2)
        p2 = plot_traces(out, meta, os.path.join(td, "zoom.png"), n_show=2,
                         t_lo=0.0, t_hi=3.0)
        p3 = plot_tsne(out, meta, os.path.join(td, "tsne.png"),
                       source="psd", perplexity=25.0)
        sizes = [os.path.getsize(p) for p in (p1, p2, p3)]
        ok("D7 every figure is written and non-empty",
           all(s > 5000 for s in sizes),
           "sizes %s bytes" % sizes)

    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(path, meta=np.array(repr(meta)), **out)
        arrays, meta2 = load_demo(path)
        ok("D8 the demo file round-trips",
           np.array_equal(arrays["x"], out["x"]) and meta2["fs"] == 50.0,
           "arrays and meta survive save/load")
    finally:
        os.unlink(path)


def main():
    print("=" * 82)
    print("Smoke test: 3-class demonstration (generate + plot)")
    print("=" * 82)
    out = test_generation()
    test_features()
    test_figures(out)
    print("-" * 82)
    n_fail = RESULTS.count("FAIL")
    print("%d passed, %d failed, %d skipped"
          % (RESULTS.count("PASS"), n_fail, RESULTS.count("SKIP")))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
