"""Block 1 (e1_numbers.py): spec-derived tests.

Oracles (SPEC Block 1): every [RAN] Bk number of E1 equals what Bk prints;
each recomputed from the derivation E1 states and the FREEZE config; [REPO]
values equal the FREEZE file's fields; two runs identical.
"""
import json
import math
import re

import pytest

from conftest import doc_text, floats, git_show, line_with, run_script, script_output

CFG_PATH = "hpc/dsn/hpc/Config/refit_mea_joint_full_r2_l0_t82.json"


@pytest.fixture(scope="module")
def out():
    return script_output("e1_numbers.py")


@pytest.fixture(scope="module")
def cfg():
    return json.loads(git_show(CFG_PATH))


def test_exit_zero_and_ascii():
    res = run_script("e1_numbers.py")
    assert res.returncode == 0, res.stderr
    res.stdout.encode("ascii")          # raises if not plain ASCII (SPEC sec. 3)


def test_repo_values_equal_freeze_fields(out, cfg):
    coh, data = cfg["cohort"], cfg["data"]
    for key in ("w_size", "gaussian_window", "n_subsets", "electrodes_per_subset",
                "fs_raw", "grid_width"):
        ln = line_with(out, "cohort.%s " % key)
        assert floats(ln.split("=")[1])[0] == pytest.approx(coh[key], rel=0, abs=0)
    for key in ("window_s", "train_stride_s"):
        ln = line_with(out, "data.%s " % key)
        assert floats(ln.split("=")[1])[0] == data[key]
    assert floats(line_with(out, "backbone.embedding_size").split("=")[1])[0] == \
        cfg["backbone"]["embedding_size"]


def test_b1_window_grid_recomputed(out, cfg):
    coh, data = cfg["cohort"], cfg["data"]
    dt, sig = coh["w_size"], coh["gaussian_window"]
    t_win, stride = data["window_s"], data["train_stride_s"]
    T_rec, G = 1200.0, 35                      # [KB] E1 S3.2 (35 wells, 1200 s)
    f_s = 1.0 / dt
    assert "f_s = 1/dt = 1/0.01 = 100.0 Hz" in out and f_s == pytest.approx(100.0)
    W = round(t_win * f_s)
    assert W == 18000 and "= 18000 samples" in out
    assert sig / dt == pytest.approx(2.0) and "= 2.0 bins" in out
    # windows back to back from t = 0: k windows need (k-1)*stride + T_win <= T_rec
    k = max(j for j in range(1, 100) if (j - 1) * stride + t_win <= T_rec)
    assert k == 6
    left = T_rec - ((k - 1) * stride + t_win)
    assert left == 120.0
    assert re.search(r"\+ 1 = 6; 120 s left over", out)
    n_win = coh["n_subsets"] * k
    assert n_win == 54 and "= 54 windows per culture" in out
    assert G * n_win == 1890 and "35 x 54 = 1890 windows" in out
    assert G * coh["n_subsets"] == 315 and "= 315 subregion traces" in out
    raw = coh["fs_raw"] * T_rec
    assert round(raw) == 12132108 and "-> 12132108" in out
    pooled = coh["n_subsets"] * coh["electrodes_per_subset"]
    share = 100.0 * pooled / coh["grid_width"] ** 2
    assert pooled == 81 and coh["grid_width"] ** 2 == 2304
    assert "(%.2f %%)" % share in out and "%.2f" % share == "3.52"


def test_b2_parameter_vector(out):
    assert "d_theta = 23 neuron/synapse + 3 kernel = 26" in out
    assert "is 17 ln + 6 linear = 23" in out
    # E1 eq. (E1.1): log iff a > 0 and b >= 10 a (and not a kernel axis); p0_conn
    # [0.1, 1.0] satisfies the first two conditions exactly at the boundary
    assert 1.0 >= 10 * 0.1
    assert "log10(1.0/0.1) = 1.000000 -> ln" in out
    assert "log10(0.6/0.1) = %.6f -> linear" % math.log10(6.0) in out


def test_b3_kept_share(out):
    kept = 29616 / 86251
    assert "= %.4f (%.1f %%); drops %.1f %%" % (kept, 100 * kept, 100 - 100 * kept) in out
    assert "%.4f" % kept == "0.3434"


def test_b4_gate_floor(out):
    # B = 1/floor - 1; 1/0.001996 = 501.002..., so B = 500 to the printed precision
    b = 1.0 / 0.001996 - 1.0
    assert round(b) == 500
    assert "= 500.00;" in out
    assert "%.6f" % (1 / 501) == "0.001996"


def test_b5_class_roots(out, cfg):
    roots = cfg["cohort"]["class_roots"]
    names = cfg["cohort"]["class_names"]
    for key in roots:
        ln = line_with(out, "class key '%s'" % key)
        assert repr(names[int(key)]) in ln and "%d roots" % len(roots[key]) in ln
    assert names == ["control", "pathological"]


def test_e1_quotes_match_output(out):
    """Every [RAN] number E1 quotes, as quoted, against the printed value."""
    e1 = doc_text("E1_THE_PROBLEM.md")
    for quoted in ("$f_s = 1/\\Delta t = 100$ Hz", "$\\sigma_{\\rm sm}/\\Delta t = 2$ bins",
                   "\\mathrm{round}(180 \\times 100) = 18000$", "12,132,108",
                   "$9 \\times 9 = 81$ of its 2304 grid sites, 3.52 %",
                   "$29{,}616 / 86{,}251 = 0.3434$", "$23 + 3 = 26$",
                   "$\\log_{10}(1.0/0.1) = 1.000000$", "17 log and 6 linear"):
        assert quoted in e1, quoted
    assert "1/501" in e1 and "500 permutations" in e1


def test_two_runs_identical(out):
    res = run_script("e1_numbers.py")
    assert res.stdout == out
