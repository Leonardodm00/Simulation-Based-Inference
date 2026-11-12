#!/usr/bin/env python3
"""The data overview's nine figures, re-drawn from a data_view.npz by the
job's own figure code (hpc/pipeline/evaluation/ev_data_figures.make_all,
block 23) at a smaller resolution, into ../figures/ of this document.

The document's copies were drawn from the sandbox's run on banks rebuilt
with davinci's seeds (2026-11-12). Run it on davinci's file to see the same
figures drawn from davinci's banks (or open the job's own PNGs in
<out_root>/data/figures/):

    python3 make_job_figures.py \
        --data-view /davinci-1/home/ldellamea/runs/synth/evaluation/data/data_view.npz \
        [--out ../figures] [--dpi 110]

It copies data_view.npz into a temporary folder (make_all writes beside
its input) and copies the PNGs to --out under the job's names.

Pure ASCII, LF only.
"""

import argparse
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "hpc", "pipeline", "evaluation")))
import ev_data_figures  # noqa: E402  (its _evalpath puts hpc/joint/stage1 on the path)

CONFIG = {"dpi": 110, "formats": ("png",)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-view", required=True)
    p.add_argument("--out", default=os.path.join(HERE, "..", "figures"))
    p.add_argument("--dpi", type=int, default=CONFIG["dpi"])
    a = p.parse_args(argv)
    out = os.path.normpath(a.out)
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(a.data_view, os.path.join(tmp, "data_view.npz"))
        ecfg = types.SimpleNamespace(formats=CONFIG["formats"], dpi=a.dpi)
        figs = ev_data_figures.make_all(tmp, ecfg)
        for name, _caption in figs:
            shutil.copy(os.path.join(tmp, "figures", name + ".png"), os.path.join(out, name + ".png"))
            print("wrote %s" % os.path.join(out, name + ".png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
