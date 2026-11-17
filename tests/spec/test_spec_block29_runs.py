"""Block 29: several searches in one recorded project, each with its own core
cap (specs/SPEC.md secs. 2.1, 3.1 and block 29).

Oracles (from the spec, before the code was read):
  O29.1 a list Q gives S1-S6, then search_<q>, best_fit_<q> per q in order;
        search_<q> depends on (sim_bank, real_bank), best_fit_<q> on search_<q>;
        no stage `search` or `best_fit`.
  O29.2 `searches` null: the eight stages of STAGE_ORDER with DEPS (regression).
  O29.3 the view of q: tune_root/<q>, runs_root/<q>, search.campaign = q.campaign;
        folders <tune_root>/<q>/<campaign>/ and <runs_root>/<q>/<dataset>_best/.
  O29.4 shard_globs(view): no real bank for S-A1, <banks_root>/<ds>_real/shard_*.npz
        for the others; the sim glob always <banks_root>/<ds>_sim/shard_*.npz.
  O29.5 done(search_q) = its SEARCH_COMPLETE.json exists; done(best_fit_q) = its
        BEST_FIT_COMPLETE.json names SEARCH_COMPLETE.json's finalist (block 7 (i)).
  O29.6 core_group_of: (max_cores, (search_q, best_fit_q)) for an int cap; None
        for S1-S6, for max_cores null, for every synthetic run stage.
  O29.7 fits(n) iff n fits under resources.max_cores AND n <= c - in-flight(G);
        check(n) raises CapExceeded iff n > c (naming the search and the cap) or
        n > the project's share; G's jobs count against the project's cap; other
        stages' jobs never count against c.
  O29.8 cores_in_flight(state, names=None) is the whole project's count; with
        names only those stages, counted the same way (done stages, finished
        jobs out; `after` groups the larger of the two).
  O29.9 status / dry run: `cores of search <q>: at most <c> (searches[].max_cores);
        in flight <x> in <k> job(s)` per capped search; none without a list.
  O29.13 cap_room: the dry runs' trial and member lines state min(the project's
        room, max_cores // ncpus) (LindDUP: 3, where the project alone gives 9).
  O29.11 tools: --search on a list; REFUSED lines and exit 2.
"""

import os
import re
import shutil

import pytest

import _spec_util as U

import config as C            # noqa: E402
import pipeline as P          # noqa: E402
import stages as S            # noqa: E402
import stages.common as CM    # noqa: E402
import stages.runs as R       # noqa: E402
import state as ST            # noqa: E402

WANT12 = ["real_extraction", "sim_redetection", "label_freeze", "bank_plan", "sim_bank",
          "real_bank", "search_A1", "best_fit_A1", "search_A2", "best_fit_A2", "search_A5",
          "best_fit_A5"]
EIGHT = ["real_extraction", "sim_redetection", "label_freeze", "bank_plan", "sim_bank",
         "real_bank", "search", "best_fit"]


@pytest.fixture
def root():
    r = U.tmpdir("spec_b29_")
    yield r
    shutil.rmtree(r, ignore_errors=True)


def _lin(root, mutate=None):
    path, d = U.linddup_copy(root, mutate)
    return C.load_config(path), d


# ----------------------------------------------------------- O29.1, O29.2 ---

def test_graph_of_a_list(root):
    cfg, d = _lin(root)
    g = S.default_graph(cfg)
    names = [s.name for s in g]
    assert names == WANT12
    by = dict((s.name, s) for s in g)
    want_deps = {"real_extraction": (), "sim_redetection": ("real_extraction",),
                 "label_freeze": (), "bank_plan": ("real_extraction", "sim_redetection",
                                                   "label_freeze"),
                 "sim_bank": ("bank_plan",), "real_bank": ("bank_plan",)}
    for n, deps in want_deps.items():
        assert tuple(by[n].deps) == deps, n
    for q in ("A1", "A2", "A5"):
        assert tuple(by["search_" + q].deps) == ("sim_bank", "real_bank")
        assert tuple(by["best_fit_" + q].deps) == ("search_" + q,)
    # the order is a dependency order (every dependency earlier in the list)
    pos = dict((n, i) for i, n in enumerate(names))
    for s in g:
        for dep in s.deps:
            assert pos[dep] < pos[s.name], (s.name, dep)


def test_graph_order_follows_the_list(root):
    def mut(d):
        d["searches"] = [{"name": "Z5", "campaign": "S-A5-only", "max_cores": None},
                         {"name": "B1", "campaign": "S-A1", "max_cores": 148}]
    cfg, _ = _lin(root, mut)
    names = [s.name for s in S.default_graph(cfg)]
    assert names[6:] == ["search_Z5", "best_fit_Z5", "search_B1", "best_fit_B1"]


def test_graph_null_is_the_eight(root):
    def mut(d):
        d["searches"] = None
        d["search"]["campaign"] = "S-A1"
        d["search"]["arm"] = "A1"
    cfg, _ = _lin(root, mut)
    g = S.default_graph(cfg)
    assert [s.name for s in g] == EIGHT
    deps = {"search": ("sim_bank", "real_bank"), "best_fit": ("search",)}
    for s in g:
        if s.name in deps:
            assert tuple(s.deps) == deps[s.name]
        assert getattr(s, "core_group", None) is None
    # the committed Giulia file (searches null) too
    gi = C.load_config(U.GIULIA)
    assert [s.name for s in S.default_graph(gi)] == EIGHT


# ----------------------------------------------------------- O29.3, O29.4 ---

def test_views_and_globs(root):
    cfg, d = _lin(root)
    views = dict(R.search_views(cfg))
    assert sorted(views) == ["A1", "A2", "A5"]
    camp = {"A1": "S-A1", "A2": "S-A2-only", "A5": "S-A5-only"}
    for q, v in views.items():
        vd = v.data
        assert vd["paths"]["tune_root"] == os.path.join(d["paths"]["tune_root"], q)
        assert vd["paths"]["runs_root"] == os.path.join(d["paths"]["runs_root"], q)
        assert vd["search"]["campaign"] == camp[q]
        # every other key unchanged
        for k in ("work_root", "banks_root", "sbix_dir", "sbi_dir"):
            assert vd["paths"][k] == d["paths"][k]
        assert vd["dataset"] == d["dataset"] and vd["replicate"] == d["replicate"]
        sim, real = CM.shard_globs(v)
        assert sim == os.path.join(d["paths"]["banks_root"], "linddup_sim", "shard_*.npz")
        if q == "A1":
            assert real is None
        else:
            assert real == os.path.join(d["paths"]["banks_root"], "linddup_real", "shard_*.npz")
    # the base configuration is untouched by the views
    assert "campaign" not in cfg.data["search"]
    assert cfg.data["paths"]["tune_root"] == d["paths"]["tune_root"]


# ------------------------------------------------------------------ O29.5 ---

def _write(path, obj):
    U.write_json(path, obj)


def test_done_predicates_on_the_views(root):
    cfg, d = _lin(root)
    by = dict((s.name, s) for s in S.default_graph(cfg))
    tune, runs = d["paths"]["tune_root"], d["paths"]["runs_root"]
    sc = os.path.join(tune, "A2", "S-A2-only", "SEARCH_COMPLETE.json")
    bf = os.path.join(runs, "A2", "linddup_best", "BEST_FIT_COMPLETE.json")
    assert not by["search_A2"].is_done(cfg)[0]
    assert not by["best_fit_A2"].is_done(cfg)[0]
    _write(sc, {"ended_by": "rounds", "finalist": {"trial_id": "a" * 64, "nll": 1.0}})
    assert by["search_A2"].is_done(cfg)[0]
    assert not by["search_A1"].is_done(cfg)[0] and not by["search_A5"].is_done(cfg)[0]
    _write(bf, {"finalist": {"trial_id": "a" * 64}})
    assert by["best_fit_A2"].is_done(cfg)[0]
    _write(bf, {"finalist": {"trial_id": "b" * 64}})
    ok, why = by["best_fit_A2"].is_done(cfg)
    assert not ok and why                      # the reason names the condition
    # the single-search folders are not read
    _write(os.path.join(tune, "S-A2-only", "SEARCH_COMPLETE.json"), {"ended_by": "rounds"})
    assert not by["search_A5"].is_done(cfg)[0]


# ------------------------------------------------------------------ O29.6 ---

def test_core_groups(root):
    cfg, _ = _lin(root)
    g = S.default_graph(cfg)
    for s in g[:6]:
        assert R.core_group_of(s) is None and getattr(s, "core_group", None) is None
    for q in ("A1", "A2", "A5"):
        for n in ("search_" + q, "best_fit_" + q):
            s = [x for x in g if x.name == n][0]
            assert R.core_group_of(s) == (148, ("search_" + q, "best_fit_" + q))
            assert s.core_group == R.core_group_of(s)

    def mut(d):
        d["searches"][1]["max_cores"] = None
    cfg2, _ = _lin(root, mut)
    g2 = dict((s.name, s) for s in S.default_graph(cfg2))
    assert R.core_group_of(g2["search_A2"]) is None and R.core_group_of(g2["best_fit_A2"]) is None
    assert R.core_group_of(g2["search_A1"]) == (148, ("search_A1", "best_fit_A1"))
    # the synthetic kind: no group, the alt best fit included
    sy = C.load_config(U.SYNTH, synthetic_ok=True)
    for s in S.default_graph(sy):
        assert R.core_group_of(s) is None


# ------------------------------------------------------- O29.7, O29.8 ---

class _Cfg(object):
    pass


def _state(root, stages):
    """A State whose stages hold the given jobs: {name: (status, [(id, ncpus,
    finished, after)])}."""
    st = ST.State.new(os.path.join(root, "wr"), "linddup", "x", "y")
    for name, (status, jobs) in stages.items():
        rec = st.stage(name)
        rec["status"] = status
        for jid, n, fin, after in jobs:
            j = {"id": jid, "role": "r", "cmd": [], "finished": fin, "exit_status": None,
                 "walltime_s": None}
            if n is not None:
                j["ncpus"] = n
            if after:
                j["after"] = after
            rec["jobs"].append(j)
    return st


def _jobs(prefix, k, n=48):
    return [("%s%d" % (prefix, i), n, False, None) for i in range(k)]


G1 = (148, ("search_A1", "best_fit_A1"))


def test_cores_in_flight_names(root):
    st = _state(root, {
        "search_A1": ("running", _jobs("a", 2) + [("af", 48, True, None)]),
        "best_fit_A1": ("running", _jobs("m", 1)),
        "search_A2": ("running", _jobs("b", 3)),
        "sim_bank": ("running", [("arr", 20, False, None), ("gate", 8, False, "arr")]),
        "search_A5": ("done", _jobs("c", 2)),          # a done stage holds nothing
    })
    whole = P.cores_in_flight(st)
    assert whole == P.cores_in_flight(st, names=None)
    # 2*48 + 48 + 3*48 + max(20, 8) = 308 cores in 7 counted jobs (the gate counted too)
    assert whole[0] == 96 + 48 + 144 + 20 and whole[2] == []
    g = P.cores_in_flight(st, names=G1[1])
    assert g[0] == 144 and g[1] == 3
    assert P.cores_in_flight(st, names="search_A2")[0] == 144
    assert P.cores_in_flight(st, names=("search_A5", "best_fit_A5"))[0] == 0
    # `after`: the larger of the job and the sum of those after it
    assert P.cores_in_flight(st, names=("sim_bank",))[0] == 20


def _sub(root, st, group, cap=444, drv=4, stage="search_A1"):
    cfg, _ = _lin(root, lambda d: (d["resources"].__setitem__("max_cores", cap),
                                   d["driver"].__setitem__("ncpus", drv)))
    return P.Submitter(None, st, stage, lambda t: None, cfg, group=group)


def test_submitter_group_fits(root):
    st = _state(root, {"search_A1": ("running", _jobs("a", 2)),
                       "search_A2": ("running", _jobs("b", 3)),
                       "best_fit_A5": ("running", _jobs("c", 3))})
    sub = _sub(root, st, G1)
    assert sub.fits(48)                 # group 96 + 48 = 144 <= 148; project 384 + 48 <= 440
    assert sub.fits(52)                 # 96 + 52 = 148: the bound is inclusive
    assert not sub.fits(53)             # 149 > 148
    st.stage("best_fit_A1")["jobs"].extend(
        {"id": "m0", "role": "r", "cmd": [], "finished": False, "ncpus": 48} for _ in [0])
    st.stage("best_fit_A1")["status"] = "running"
    assert not sub.fits(48)             # the best fit's job counts in the group: 144 + 48
    assert sub.fits(4)                  # 148
    # the other searches' jobs never count against A1's cap
    st2 = _state(root, {"search_A1": ("running", []),
                        "search_A2": ("running", _jobs("b", 3)),
                        "search_A5": ("running", _jobs("c", 3))})
    sub2 = _sub(root, st2, G1)
    assert sub2.fits(48) and sub2.group_free() == 148
    # the project's cap still binds: 9 jobs of 48 elsewhere = 432; 432 + 48 > 440
    st3 = _state(root, {"search_A1": ("running", []),
                        "search_A2": ("running", _jobs("b", 3)),
                        "search_A5": ("running", _jobs("c", 3)),
                        "best_fit_A2": ("running", _jobs("d", 3))})
    sub3 = _sub(root, st3, G1)
    assert not sub3.fits(48) and sub3.fits(8)
    # a finished job and a done stage hold nothing
    st4 = _state(root, {"search_A1": ("running", [("a0", 48, True, None)] + _jobs("a", 2)),
                        "best_fit_A1": ("done", _jobs("m", 3))})
    assert _sub(root, st4, G1).fits(48)


def test_submitter_group_check(root):
    st = _state(root, {})
    sub = _sub(root, st, G1)
    assert sub.check(148) is True
    with pytest.raises(P.CapExceeded) as e:
        sub.check(149)
    msg = str(e.value)
    assert "A1" in msg and "148" in msg              # names the search and its max_cores
    with pytest.raises(P.CapExceeded):
        sub.fits(149)                                 # fits asks check first: it can never start
    # without a group: the project's share alone (440)
    nog = _sub(root, st, None)
    assert nog.fits(440) and nog.check(440) is True
    with pytest.raises(P.CapExceeded):
        nog.check(441)
    assert nog.fits(None) and sub.fits(None)          # an unknown size always fits


def test_submitter_null_group_project_only(root):
    """A search with max_cores null is limited by the project's cap only."""
    st = _state(root, {"search_A2": ("running", _jobs("b", 8))})        # 384 in flight
    sub = _sub(root, st, None, stage="search_A2")
    assert sub.fits(48) and not sub.fits(49 + 8)                         # 432 ok; 441 > 440


def test_submit_guard_raises_when_group_full(root):
    st = _state(root, {"search_A1": ("running", _jobs("a", 3))})

    class NoPBS(object):
        def submit(self, *a, **k):
            raise AssertionError("submitted past the group's cap")

        def select(self, *a, **k):
            return []

        def status(self, *a, **k):
            raise AssertionError("no status expected")
    sub = P.Submitter(NoPBS(), st, "search_A1", lambda t: None,
                      _lin(root)[0], group=G1)
    with pytest.raises(P.CapExceeded):
        sub.submit(["qsub", "-l", "select=1:ncpus=48", "x.pbs"], role="trial", ncpus=48)


# ------------------------------------------------------ O29.9, O29.13 ---

def _dry(root, path):
    spool = os.path.join(root, "spool")
    os.makedirs(spool, exist_ok=True)
    env = U.env_with_fake_pbs(spool)
    return U.run([U.PYTHON, os.path.join(U.PIPE, "pipeline.py"), "run", "--config", path,
                  "--dry-run"], env=env, cwd=root, timeout=300)


def test_dry_run_lines_of_a_list(root):
    path, d = U.linddup_copy(root)
    rc, out = _dry(root, path)
    assert rc == 0, out[-3000:]
    for q in ("A1", "A2", "A5"):
        assert ("cores of search %s: at most 148 (searches[].max_cores); in flight 0 in 0 "
                "job(s)" % q) in out, out[-3000:]
    # cap_room: min(440 // 48 = 9, 148 // 48 = 3) = 3 for the trials and the members
    trial_lines = [ln for ln in out.splitlines() if "trial(s) at once" in ln]
    member_lines = [ln for ln in out.splitlines() if "member(s) at once" in ln]
    assert len(trial_lines) == 3 and len(member_lines) == 3, out[-3000:]
    for ln in trial_lines:
        assert "at most 3 trial(s) at once" in ln and "room for 3 of 48" in ln, ln
    for ln in member_lines:
        assert "at most 3 member(s) at once" in ln and "room for 3 of 48" in ln, ln
    assert "REFUSE" not in out


def test_dry_run_lines_without_a_list(root):
    path, d = U.giulia_copy(root)
    rc, out = _dry(root, path)
    assert rc == 0, out[-3000:]
    assert "cores of search" not in out
    room = (d["resources"]["max_cores"] - d["driver"]["ncpus"]) // d["search"]["trial"]["ncpus"]
    n = min(room, d["search"]["max_concurrent_trials"])
    assert re.search(r"at most %d trial\(s\) at once" % n, out), out[-3000:]


def test_status_lines_of_a_list_with_jobs_in_flight(root):
    path, d = U.linddup_copy(root)
    wr = d["paths"]["work_root"]
    os.makedirs(wr)
    cfg = C.load_config(path)
    st = ST.State.new(wr, "linddup", path, cfg.sha256)
    rec = st.stage("search_A2")
    rec["status"] = "running"
    rec["jobs"] = [{"id": "11.srv", "role": "trial x", "cmd": [], "finished": False,
                    "ncpus": 48},
                   {"id": "12.srv", "role": "trial y", "cmd": [], "finished": False,
                    "ncpus": 48}]
    lines = P.search_lines(cfg, st)
    assert ("cores of search A2: at most 148 (searches[].max_cores); in flight 96 in 2 job(s)"
            in lines), lines
    assert ("cores of search A1: at most 148 (searches[].max_cores); in flight 0 in 0 job(s)"
            in lines), lines
    # nothing for a configuration without a list
    gi = C.load_config(U.GIULIA)
    assert P.search_lines(gi, st) == []


# ------------------------------------------------------------------ O29.11 ---

def _tool(root, tool, path, *extra):
    spool = os.path.join(root, "spool")
    os.makedirs(spool, exist_ok=True)
    env = U.env_with_fake_pbs(spool)
    return U.run([U.PYTHON, os.path.join(U.TOOLS, tool), "--config", path] + list(extra),
                 env=env, cwd=root, timeout=300)


REFUSED_LIST = "REFUSED: this configuration has the searches A1, A2, A5: give --search <name>"


@pytest.mark.parametrize("tool", ["search_report.py", "search_reset.py"])
def test_search_tools_need_search_on_a_list(root, tool):
    path, d = U.linddup_copy(root)
    for extra in ([], ["--search", "A9"]):
        rc, out = _tool(root, tool, path, *extra)
        assert rc == 2, (extra, out)
        assert REFUSED_LIST in out, (extra, out)


@pytest.mark.parametrize("tool", ["search_report.py", "search_reset.py"])
def test_search_tools_refuse_search_on_null(root, tool):
    path, d = U.giulia_copy(root)
    rc, out = _tool(root, tool, path, "--search", "A1")
    assert rc == 2 and "REFUSED:" in out, out


def test_search_report_on_a_view(root):
    path, d = U.linddup_copy(root)
    camp = os.path.join(d["paths"]["tune_root"], "A5", "S-A5-only")
    U.write_json(os.path.join(camp, "trials", "t" * 64 + ".json"),
                 {"trial_id": "t" * 64, "status": "ok", "nll": 1.25, "config": {"lr": 0.001}})
    rc, out = _tool(root, "search_report.py", path, "--search", "A5")
    assert rc == 0, out
    assert out.rstrip().splitlines()[-1].startswith("SEARCH REPORT OK:"), out
    assert "campaign folder %s" % camp in out, out
    # the same file read through another search's view: not seen
    rc2, out2 = _tool(root, "search_report.py", path, "--search", "A2")
    assert rc2 == 0 and "t" * 12 not in out2


def test_search_reset_on_a_view_lists_its_folder(root):
    path, d = U.linddup_copy(root)
    camp = os.path.join(d["paths"]["tune_root"], "A2", "S-A2-only")
    U.write_json(os.path.join(camp, "trials", "u" * 64 + ".json"),
                 {"trial_id": "u" * 64, "status": "ok", "nll": 1.0})
    rc, out = _tool(root, "search_reset.py", path, "--search", "A2")
    assert rc == 0, out
    assert camp in out and re.search(r"^CODE [0-9a-f]{8} ", out, re.M), out
    assert "--retry search_A2" in out, out
    assert os.path.isfile(os.path.join(camp, "trials", "u" * 64 + ".json"))   # nothing moved


@pytest.mark.parametrize("tool,extra", [("perrow_check.py", []),
                                        ("purge_outputs.py", ["--what", "search"])])
def test_perrow_and_purge_refuse_a_list(root, tool, extra):
    path, d = U.linddup_copy(root)
    rc, out = _tool(root, tool, path, *extra)
    assert rc == 2, out
    refused = [ln for ln in out.splitlines() if ln.startswith("REFUSED:")]
    assert len(refused) == 1 and "several searches" in refused[0], out


def test_perrow_and_purge_unchanged_on_null(root):
    path, d = U.giulia_copy(root)
    rc, out = _tool(root, "purge_outputs.py", path, "--what", "search")
    assert rc == 0 and "REFUSED" not in out and re.search(r"^CODE [0-9a-f]{8}", out, re.M), out
    rc, out = _tool(root, "perrow_check.py", path)
    assert rc == 3 and "REFUSED" not in out, out          # no per-row file to read: exit 3


# ------------------------------------- block 7 (j) on the views of a list ---

def test_identity_of_each_view_matches_its_configuration():
    """Block 7 (j)'s contract on the committed files at v11: for every search
    (giulia's single one, linddup's three views, synth's five) the K_cfg part of
    the identity the configuration gives compares equal to the configuration,
    and each view's campaign is its search's."""
    import stages.search as SS
    seen = []
    for path in (U.GIULIA, U.LINDDUP, U.SYNTH):
        cfg = C.load_config(path, synthetic_ok=True)
        for name, view in R.search_views(cfg):
            ident = dict(SS.config_identity(view))
            assert SS.identity_refusal(ident, view) is None
            if name is not None:
                q = [x for x in (cfg.data.get("searches") or cfg.data["synthetic"]["searches"])
                     if x["name"] == name][0]
                assert ident["campaign"] == q["campaign"]
            seen.append((os.path.basename(path), name))
    assert len(seen) == 1 + 3 + 5, seen
