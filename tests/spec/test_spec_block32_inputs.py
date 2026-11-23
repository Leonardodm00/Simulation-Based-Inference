"""Block 32: the inputs check on configuration v11 (specs/SPEC.md block 32).

The tree is the smoke suite's lcdup15-like fixture (hpc/pipeline/tests/
test_smoke_inputs_check.py, which builds it from the spec's description); the
expected rows and counts below are the spec's, computed here by hand.

Oracles (from the spec, before the code was read):
  O32.1 `cohort map`: null -> NOTE `no cohort map: donor = well`; read by the
        Sbi-extractor checkout's cohort_map.read_cohort_map against the
        manifest -> OK `<n> wells, <k> excluded; <D> donors; <G> pairing groups;
        included per class <class>:<count>, ...` (the committed map: 35 wells, 5
        excluded, 2 donors, 4 groups, control:17, pathological:13 -- counted here
        from the CSV); its error -> FAIL; the module absent or raising
        NotImplementedError -> FAIL `the Sbi-extractor checkout has no Block 9:
        pull it`; the manifest absent -> NOTE; the map file absent -> FAIL.
  O32.2 `topology metadata`: folders without topology_meta.json -> FAIL with
        their count and the first ten paths relative to sim.sim_main; else OK.
  O32.3 `partial topologies` (drop true): K(t) = _resolved_n_workers x
        n_params_per_worker; f(t, k) = iter_*.json + non-blank lines of
        _failures.jsonl; f < K partial (NOTE `<n> partial topologies in <m>
        task(s), <s> simulations: the bank plan drops them whole`); f > K, or a
        K that is not two positive integers -> FAIL; drop false -> NOTE `not
        checked (drop_partial_topologies false)`.
  O32.4 `searches`: one NOTE naming each search, its campaign and cap, and the
        caps' sum against resources.max_cores; no row without a list.
  O32.5 last line `INPUTS CHECK: <f> FAIL, <w> WARN`; exit 0 without a FAIL, 1 with.
"""

import csv
import json
import os
import sys

import pytest

import _spec_util as U

sys.path.insert(0, U.PTESTS)
T = pytest.importorskip("test_smoke_inputs_check")


@pytest.fixture
def tree():
    try:
        r = T.fresh()
    except Exception as exc:          # the fixture needs the Sbi-extractor checkout
        pytest.skip("fixture: %s" % exc)
    yield r
    T.cleanup(r)


def rows(doc):
    out = {}
    for r in doc["checks"]:
        out.setdefault(r["check"], []).append((r["status"], r["text"]))
    return out


def failing(doc):
    return sorted(set(r["check"] for r in doc["checks"] if r["status"] == "FAIL"))


def only(doc, label):
    got = rows(doc).get(label, [])
    assert len(got) == 1, (label, got)
    return got[0]


def _map_counts():
    rws = list(csv.DictReader(open(U.COHORT_MAP, newline="")))
    inc = [r for r in rws if r["include"].strip().lower() in ("yes", "true", "1")]
    per = {}
    for r in inc:
        c = r["out_dir"].split("/")[0]
        per[c] = per.get(c, 0) + 1
    return (len(rws), len(rws) - len(inc), len(set(r["donor"] for r in inc)),
            len(set(r["rep_group"] for r in inc)), per)


def test_consistent_tree(tree):
    rc, lines, doc = T.run(tree)
    assert rc == 0 and lines[-1] == "INPUTS CHECK: 0 FAIL, 0 WARN", lines[-5:]
    n, k, D, G, per = _map_counts()
    assert (n, k, D, G, per) == (35, 5, 2, 4, {"control": 17, "pathological": 13})
    st, text = only(doc, "cohort map")
    assert st == "OK" and text.startswith(
        "%d wells, %d excluded; %d donors; %d pairing groups; included per class "
        "control:%d, pathological:%d" % (n, k, D, G, per["control"], per["pathological"])), text
    st, text = only(doc, "topology metadata")
    assert st == "OK", text
    st, text = only(doc, "partial topologies")
    assert st == "NOTE" and text.startswith("0 partial topologies in 0 task(s), 0 simulations: "
                                            "the bank plan drops them whole"), text
    st, text = only(doc, "searches")
    assert st == "NOTE", text
    for q, c in (("A1", "S-A1"), ("A2", "S-A2-only"), ("A5", "S-A5-only")):
        assert "%s: %s, max_cores 148" % (q, c) in text, text
    assert "444" in text, text                                  # 3 x 148 against 444


def test_cohort_map_null(tree):
    T.edit_json(T._cfg(tree), lambda d: d["real"].__setitem__("cohort_map", None))
    rc, lines, doc = T.run(tree)
    assert only(doc, "cohort map") == ("NOTE", "no cohort map: donor = well")
    assert rc == 0


def test_cohort_map_file_absent(tree):
    os.remove(os.path.join(tree, "linddup_cohort_map.csv"))
    rc, lines, doc = T.run(tree)
    assert failing(doc) == ["cohort map"] and rc == 1, failing(doc)


def test_cohort_map_module_absent_or_stub(tree):
    sbix = json.load(open(T._cfg(tree)))["paths"]["sbix_dir"]
    mod = os.path.join(sbix, "export_windows", "cohort_map.py")
    assert os.path.isfile(mod)
    with open(mod, "w") as fh:
        fh.write("class CohortMapError(ValueError):\n    pass\n\n\n"
                 "def read_cohort_map(path, manifest):\n    raise NotImplementedError('stub')\n")
    rc, lines, doc = T.run(tree)
    st, text = only(doc, "cohort map")
    assert st == "FAIL" and "the Sbi-extractor checkout has no Block 9: pull it" in text, text
    os.remove(mod)
    rc, lines, doc = T.run(tree)
    st, text = only(doc, "cohort map")
    assert st == "FAIL" and "the Sbi-extractor checkout has no Block 9: pull it" in text, text
    assert failing(doc) == ["cohort map"], failing(doc)


def test_cohort_map_without_manifest(tree):
    man = os.path.join(tree, "extracted", "cohort_manifest.json")
    os.remove(man)
    os.remove(man + ".sha256")
    rc, lines, doc = T.run(tree)
    assert only(doc, "cohort map")[0] == "NOTE"


def _rewrite_map(tree, fn):
    p = os.path.join(tree, "linddup_cohort_map.csv")
    with open(p, newline="") as fh:
        rws = list(csv.DictReader(fh))
        fields = list(rws[0].keys())
    rws = fn(rws)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rws:
            w.writerow(r)


def test_cohort_map_mismatches(tree):
    _rewrite_map(tree, lambda rws: rws[1:])                      # a manifest well without a row
    rc, lines, doc = T.run(tree)
    assert failing(doc) == ["cohort map"], failing(doc)


def test_cohort_map_row_of_no_well(tree):
    def add(rws):
        extra = dict(rws[0])
        extra["out_dir"] = "control/DATA_C_Batch9/ptrain_Z9"
        return rws + [extra]
    _rewrite_map(tree, add)
    rc, lines, doc = T.run(tree)
    assert failing(doc) == ["cohort map"], failing(doc)


def _drop_meta(tree, n, k):
    os.remove(os.path.join(T._topo(tree, n, k), "topology_meta.json"))
    T._old(T._topo(tree, n, k))


def test_topology_metadata_missing(tree):
    _drop_meta(tree, 0, 1)
    rc, lines, doc = T.run(tree)
    st, text = only(doc, "topology metadata")
    assert st == "FAIL" and text.startswith("1 topology folder(s) without topology_meta.json"), text
    rel = os.path.relpath(T._topo(tree, 0, 1), os.path.join(tree, T.SIM))
    assert rel in text, (rel, text)
    assert failing(doc) == ["topology metadata"] and rc == 1, failing(doc)


def test_topology_metadata_first_ten(tree):
    """12 folders without metadata: the count 12, the first ten paths named."""
    td = T._task(tree, 0)
    made = []
    for k in range(3, 15):
        p = os.path.join(td, "topo_%05d" % k)
        os.makedirs(p)
        made.append(os.path.relpath(p, os.path.join(tree, T.SIM)))
    T._old(td, *[os.path.join(td, "topo_%05d" % k) for k in range(3, 15)])
    rc, lines, doc = T.run(tree)
    st, text = only(doc, "topology metadata")
    assert st == "FAIL" and text.startswith("12 topology folder(s)"), text
    named = [m for m in made if m in text]
    assert len(named) == 10 and named == made[:10], (named, text)


def _iter_json(tree, n, k):
    top = T._topo(tree, n, k)
    return sorted(f for f in os.listdir(top) if f.startswith("iter_") and f.endswith(".json"))


def test_partial_topology_counted(tree):
    # task 0 (K = 2 x 2 = 4): topology 1 keeps 2 of its 4 runs (json and npz)
    top = T._topo(tree, 0, 1)
    for i in (2, 3):
        os.remove(os.path.join(top, "iter_%05d.json" % i))
        os.remove(os.path.join(top, "iter_%05d.npz" % i))
    T._old(top)
    rc, lines, doc = T.run(tree)
    st, text = only(doc, "partial topologies")
    assert st == "NOTE" and text.startswith("1 partial topologies in 1 task(s), 2 simulations: "
                                            "the bank plan drops them whole"), text
    assert rc == 0 and failing(doc) == []


def test_failures_lines_count_as_finished(tree):
    top = T._topo(tree, 1, 0)                       # task 1: K = 3 x 1 = 3
    for i in (1, 2):
        os.remove(os.path.join(top, "iter_%05d.json" % i))
    with open(os.path.join(top, "_failures.jsonl"), "w") as fh:
        fh.write('{"iter": 1}\n\n   \n{"iter": 2}\n')          # two non-blank lines
    T._old(top)
    rc, lines, doc = T.run(tree)
    st, text = only(doc, "partial topologies")
    assert st == "NOTE" and text.startswith("0 partial topologies"), text   # f = 1 + 2 = K
    with open(os.path.join(top, "_failures.jsonl"), "a") as fh:
        fh.write('{"iter": 3}\n')                                         # f = 4 > K = 3
    T._old(top)
    rc, lines, doc = T.run(tree)
    assert failing(doc) == ["partial topologies"] and rc == 1, failing(doc)


def test_more_runs_than_dispatched_refused(tree):
    top = T._topo(tree, 0, 0)
    with open(os.path.join(top, "iter_00099.json"), "w") as fh:
        json.dump({"iter_idx": 99}, fh)
    T._old(top)
    rc, lines, doc = T.run(tree)
    assert failing(doc) == ["partial topologies"], failing(doc)


@pytest.mark.parametrize("change", [
    lambda ja: ja.pop("n_params_per_worker"),
    lambda ja: ja.__setitem__("_resolved_n_workers", True),
    lambda ja: ja.__setitem__("_resolved_n_workers", 2.0),
    lambda ja: ja.__setitem__("n_params_per_worker", 0),
    lambda ja: ja.__setitem__("n_params_per_worker", "2"),
])
def test_dispatch_size_must_be_positive_integers(tree, change):
    p = os.path.join(T._task(tree, 0), "job_args.json")
    T.edit_json(p, change)
    T._old(p, T._task(tree, 0))
    rc, lines, doc = T.run(tree)
    assert "partial topologies" in failing(doc), failing(doc)


def test_drop_false_not_checked(tree):
    T.edit_json(T._cfg(tree), lambda d: d["bank"].__setitem__("drop_partial_topologies", False))
    top = T._topo(tree, 0, 0)
    with open(os.path.join(top, "iter_00099.json"), "w") as fh:
        json.dump({"iter_idx": 99}, fh)                              # would refuse when true
    T._old(top)
    rc, lines, doc = T.run(tree)
    assert only(doc, "partial topologies") == ("NOTE", "not checked (drop_partial_topologies false)")
    assert rc == 0


def test_no_searches_row_without_a_list(tree):
    def null(d):
        d["searches"] = None
        d["search"]["campaign"] = "S-A1"
        d["search"]["arm"] = "A1"
    T.edit_json(T._cfg(tree), null)
    rc, lines, doc = T.run(tree)
    assert "searches" not in rows(doc) and rc == 0, rows(doc).get("searches")
