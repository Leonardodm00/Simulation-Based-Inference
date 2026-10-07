"""Block 2 (e2_numbers.py): spec-derived tests.

Oracles (SPEC Block 2): (E2.2) and (E2.5) on the B1 linear-Gaussian toy,
closed form against Monte Carlo within the script's MC standard error; the B2
unit-interval toy's closed forms (E_U[ln(theta(1-theta))] = -2 per axis);
every [RAN] Bk number of E2 equals what Bk prints; the B5 anchor agrees with
P4 eq. (P4.10); refusal to import hpc/joint/stage1 when it differs from FREEZE;
two identical runs. SPEC sec. 3: [REPO] inputs are never read from the working
tree without the identity check.
"""
import math
import os

import numpy as np
import pytest
from scipy import integrate, stats

from conftest import doc_text, floats, line_with, run_script, script_output


@pytest.fixture(scope="module")
def out():
    return script_output("e2_numbers.py")


# ---------------------------------------------------------------- B1
def _toy_closed(sig=0.5, a=0.6, b=0.6, tau=2.0):
    """theta ~ N(0,1), z = theta + sig eps; q = N(a z, b^2); reference N(0, tau^2).
    Every term from Gaussian algebra (independent derivation)."""
    post_var = sig ** 2 / (1 + sig ** 2)
    m = 1 / (1 + sig ** 2)
    ez2 = 1 + sig ** 2
    h_cond = 0.5 * math.log(2 * math.pi * math.e * post_var)
    # E_z KL(N(m z, s2) || N(a z, b2)) = ln(b/s) + (s2 + (m-a)^2 E z^2)/(2 b2) - 1/2
    ekl = 0.5 * math.log(b * b / post_var) + (post_var + (m - a) ** 2 * ez2) / (2 * b * b) - 0.5
    # E[-ln q(theta | z)], E (theta - a z)^2 = 1 - 2a + a^2 (1 + sig^2)
    loss = 0.5 * math.log(2 * math.pi * b * b) + (1 - 2 * a + a * a * ez2) / (2 * b * b)
    mi = 0.5 * math.log(1 + 1 / sig ** 2)
    h_theta = 0.5 * math.log(2 * math.pi * math.e)
    kl_marg = math.log(tau) + 1 / (2 * tau ** 2) - 0.5      # KL(N(0,1) || N(0,tau^2))
    l0_ref = h_theta + kl_marg
    return dict(h_cond=h_cond, ekl=ekl, loss=loss, mi=mi, h_theta=h_theta,
                kl_marg=kl_marg, l0_ref=l0_ref)


def test_b1_identities_closed_form(out):
    c = _toy_closed()
    # (E2.2): loss = E H[theta|z] + E KL
    assert c["loss"] == pytest.approx(c["h_cond"] + c["ekl"], abs=1e-12)
    # (E2.5) case 1 and case 2
    assert c["h_theta"] - c["loss"] == pytest.approx(c["mi"] - c["ekl"], abs=1e-12)
    assert c["l0_ref"] - c["loss"] == pytest.approx(c["mi"] + c["kl_marg"] - c["ekl"], abs=1e-12)
    # printed values (6 decimals)
    assert "E H[theta|z]           closed %.6f" % c["h_cond"] in out
    assert "E KL(post || q)        closed %.6f" % c["ekl"] in out
    assert "E[-log q] (the loss)   closed %.6f" % c["loss"] in out
    assert "I(theta; z) = 0.5 ln(1 + 1/sigma^2) = %.6f" % c["mi"] in out
    assert "Delta = L0 - loss = %.6f" % (c["h_theta"] - c["loss"]) in out
    assert "(KL(p_ev||ref) = %.6f)" % c["kl_marg"] in out


def test_b1_monte_carlo_within_stated_se(out):
    c = _toy_closed()
    ln = line_with(out, "E[-log q] (the loss)")
    closed, mc, se = floats(ln.split("closed")[1])
    assert closed == pytest.approx(c["loss"], abs=1e-6)
    # the script's own tolerance is 5 standard errors; check it independently
    assert abs(mc - c["loss"]) < 5 * se
    ln = line_with(out, "E KL(post || q)")
    _, mc_kl = floats(ln.split("closed")[1])
    assert abs(mc_kl - c["ekl"]) < 5 * se       # same draws, same order of SE (tolerance: 5 SE)


# ---------------------------------------------------------------- B2
def test_b2_unit_interval_closed_forms(out):
    # E_U[ln(theta (1 - theta))] = 2 * int_0^1 ln u du = -2 exactly
    val, _ = integrate.quad(lambda u: math.log(u * (1 - u)), 0, 1)
    assert val == pytest.approx(-2.0, abs=1e-9)
    assert "E_U[ln(theta (1 - theta))] = -2.0000 per axis; x 26 = -52.0" in out
    # H(Beta(2,1)) = 1/2 - ln 2; I = ln 2 - 1/2
    h = stats.beta(2, 1).entropy()
    assert h == pytest.approx(0.5 - math.log(2), abs=1e-12)
    assert "I(theta; z) = %.6f nats" % (math.log(2) - 0.5) in out
    # filtered bank: KL(2 theta || U) = ln 2 - 1/2; bound -ln(1/2) = ln 2; slack -1/2
    assert "ln 2 - 1/2 = %.6f" % (math.log(2) - 0.5) in out
    assert "bound -ln Pr(pass) = %.6f" % math.log(2) in out
    assert "slack E[ln Pr(pass|theta)] = -0.500000" in out


# ---------------------------------------------------------------- B3
def test_b3_bound(out):
    kept = 29616 / 86251
    assert "-ln Pr(pass) = %.6f nats/row" % (-math.log(kept)) in out
    assert "%.3f" % (-math.log(kept)) == "1.069"


# ---------------------------------------------------------------- B4
def test_b4_bench_entropy_consistent(out):
    ln = line_with(out, "H[p_Theta] (Monte Carlo")
    h_mc, h_se = floats(ln.split("=")[1])[:2]
    dec = floats(line_with(out, "decomposition ln C").split("=")[-1])[0]
    assert abs(h_mc - dec) < 5 * h_se
    assert "%.2f" % dec == "-5.19"
    ln = line_with(out, "identity flow on the bench prior")
    ell = floats(ln.split("E[ell_i] =")[1])[0]
    assert "%.2f" % ell == "0.54"
    assert "%.2f" % (ell - dec) == "5.73"


def test_b4_dsn_centres_regular_simplex(out):
    # E2 F-ba: DSN's _class_center_vectors gives centred cosines -1/(C-1) = -0.5 at C = 3
    ln = line_with(out, "DSN `_class_center_vectors(3, 7, 'simplex')`")
    assert "centred cosines [-0.5, -0.5, -0.5]" in ln
    ln = line_with(out, "bench `simplex_centres(3, 7)`")
    assert "[-0.803, -0.4336, -0.189]" in ln


# ---------------------------------------------------------------- B5
def test_b5_anchor_p4_eq_4_10(out):
    # KL(standard logistic || standard normal) = E_logistic[ln logistic] - E_logistic[ln N]
    # = -H(logistic) + 0.5 ln(2 pi) + Var(logistic)/2 = -2 + 0.5 ln 2pi + pi^2/6
    kl = -2.0 + 0.5 * math.log(2 * math.pi) + math.pi ** 2 / 6
    assert "KL(standard logistic || standard normal) = %.4f per axis" % kl in out
    assert "x 26 = %.2f nats/row" % (26 * kl) in out
    p4 = doc_text("P4_FLOW_AXES.md")
    assert "14.66" in p4 and "0.5639" in p4          # P4 (P4.10)'s anchor, as quoted


def test_e2_quotes(out):
    e2 = doc_text("E2_NPE_AND_FLOWS.md")
    for q in ("$-5.19$ nats", "0.0006 nats", "0.54 nats/row", "5.73 nats above",
              "1.069 nats/row", "$\\ln 2 - 1/2 = 0.193$", "$-0.193$ nats/row",
              "$0.5639\\, d_\\theta = 14.66$", "$-52$ at $d_\\theta = 26$",
              "$-0.803$, $-0.434$, $-0.189$", "10.65, 8.05 and 6.97"):
        assert q in e2, q


# ---------------------------------------------------------------- freeze guard
def test_refuses_modified_stage1(clone):
    tools = os.path.join(clone, "hpc", "joint", "docs", "tools")
    with open(os.path.join(clone, "hpc", "joint", "stage1", "latent_sbi_simulator.py"), "a") as fh:
        fh.write("\n# modified by the spec test\n")
    res = run_script("e2_numbers.py", cwd=tools)
    assert res.returncode != 0
    assert "refusing to import" in (res.stdout + res.stderr)


def test_no_unchecked_working_tree_import(clone, tmp_path):
    """SPEC sec. 3: working-tree code is imported only after `git diff --quiet
    FREEZE -- <path>`. Importing hpc/joint/stage1 also executes
    hpc/joint/dsn_locate.py (outside the checked path); a modified copy of it
    must be refused, not executed."""
    flag = tmp_path / "executed"
    with open(os.path.join(clone, "hpc", "joint", "dsn_locate.py"), "a") as fh:
        fh.write("\nopen(%r, 'w').close()\n" % str(flag))
    tools = os.path.join(clone, "hpc", "joint", "docs", "tools")
    res = run_script("e2_numbers.py", cwd=tools)
    assert not (flag.exists() and res.returncode == 0), \
        "modified hpc/joint/dsn_locate.py was executed and the run exited 0"


def test_two_runs_identical(out):
    assert run_script("e2_numbers.py").stdout == out
