"""Chain C1 (the real-data project) -- the tester's stand-in harness.

Not pipeline code and not a test module: a fixture for the chain tests
`tests/spec/test_spec_chain_C1_*.py` of the integration tester (audit
D-131, run t2-C1). It builds a stand-in davinci on which a RECORDED data set
(no `kind`, the eight stages S1-S8 of SPEC.md sec. 2.1) runs end to end
through its real entry point `hpc/pipeline/run_pipeline.sh`: the sequencer's
driver jobs (D-098) under the fake PBS of `hpc/pipeline/tests/fake_pbs/`.

S1-S8 all run (none pre-made):
- S1, S2: the repository's own stubs of Sbi-extractor's two launchers
  (`hpc/pipeline/tests/standin.py`), changed here in two places so that the
  chain hands real artefacts on: the aggregation job writes the bank
  fixture's cohort manifest (not a stub JSON) and S2's launcher freezes
  that manifest as Sbi-extractor's real launcher does (lines 114-137 of
  `sim_reextract/launch_sim_reextract.sh` at e9aeb55: copy with its
  sidecar, or refuse a frozen copy of another manifest); S2's gate job
  writes the fixture's G2 record. The real launchers need the recordings and
  the ANN tools (davinci).
- S3: Sbi-extractor's REAL freeze tool on the fixture's simulation tree.
- S4-S6: Sbi-extractor's REAL bank tools; S7-S8 SBI's REAL tuner and runner.

Stdlib + the fixture's numpy. Pure ASCII, LF only.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PIPE = os.path.join(REPO, "hpc", "pipeline")
PTESTS = os.path.join(PIPE, "tests")
FAKE = os.path.join(PTESTS, "fake_pbs")
for _p in (PIPE, PTESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import standin as ST          # noqa: E402  (the repository's stand-in helpers)

PYTHON = sys.executable
TAG = "synth"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sidecar(path):
    h = sha256_file(path)
    with open(path + ".sha256", "w") as fh:
        fh.write("%s  %s\n" % (h, os.path.basename(path)))
    return h


def tree_snapshot(root, skip_dirs=()):
    """{relative path: (size, mtime_ns, sha256 for files <= 2 MB else None)}
    of every entry under `root` (links not followed)."""
    snap = {}
    if not os.path.exists(root):
        return snap
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in skip_dirs]
        rel_d = os.path.relpath(dirpath, root)
        st = os.lstat(dirpath)
        snap["D " + rel_d] = (None, st.st_mtime_ns, None)
        for f in filenames:
            p = os.path.join(dirpath, f)
            st = os.lstat(p)
            h = None
            if os.path.isfile(p) and not os.path.islink(p) and st.st_size <= (2 << 20):
                h = sha256_file(p)
            snap["F " + os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns, h)
    return snap


class ChainStandin(object):
    """A recorded data set on the stand-in, raw inputs only (S1-S8 to run)."""

    def __init__(self, root=None, search=None, best_fit=None, driver=None, extra=None,
                 split=None, loader=None, cap=64):
        self.root = root or tempfile.mkdtemp(prefix="chain_c1_")
        clone = ST.sbix_clone()
        if clone is None:
            raise RuntimeError("no Sbi-extractor checkout (set SBIX_CLONE)")
        self.clone = clone
        self.sbix = ST.build_sbix(self.root, clone)
        self.fakebin = ST.fake_conda(self.root)
        # the bank fixture with 12 more topologies per task (block 8's data set)
        extra_t = dict((i, dict((k, [0, 1]) for k in range(3, 15))) for i in range(3))
        self.ds = ST.write_dataset(self.root, self.sbix, tag=TAG, extra_topos=extra_t)
        ST.stub_launchers(self.sbix, tag=TAG)
        self.golden = os.path.join(self.root, "golden")
        os.makedirs(self.golden)
        self._arts = {
            "real_extraction": os.path.join(self.ds["extract_root"], "cohort_manifest.json"),
            "sim_redetection": os.path.join(self.ds["out_root"], "REEXTRACTION_RECORD.json"),
            "label_freeze": os.path.join(self.sbix, self.ds["freeze_out"]),
        }
        # keep the fixture's manifest and G2 record: the stubs' jobs write them
        shutil.copy2(self._arts["real_extraction"], os.path.join(self.golden, "cohort_manifest.json"))
        shutil.copy2(self._arts["sim_redetection"],
                     os.path.join(self.golden, "REEXTRACTION_RECORD.json"))
        shutil.copy2(self._arts["label_freeze"], os.path.join(self.golden, "label_freeze.json"))
        self._patch_stubs()
        # S1-S3's artefacts and the frozen copy removed: the chain makes them
        for p in self._arts.values():
            for q in (p, p + ".sha256"):
                if os.path.isfile(q):
                    os.remove(q)
        frozen_dir = os.path.join(self.sbix, "artifacts", "cohort_manifest_%s" % TAG)
        shutil.rmtree(frozen_dir, ignore_errors=True)
        self.frozen = os.path.join(frozen_dir, "cohort_manifest.json")
        cfg = ST.synthetic_config(self.root, self.sbix, REPO, self.ds)
        with open(cfg["real"]["cohort_config"], "w") as fh:
            fh.write('{"stub": true}\n')
        os.makedirs(cfg["sim"]["tools"], exist_ok=True)
        cfg["split"] = split or {"fractions": [0.6, 0.2, 0.2]}
        cfg["search"].update({
            "shape_anchors": {"p": 23, "d_theta": 23, "embedding_dim": 8},
            "space_overrides": {"depth_exponent": [3, 3], "width_multiplier": [1.5, 1.5],
                                "embedding_size": [8, 8], "hidden_features": [24, 24],
                                "num_transforms": [3, 3], "batch_size_npe": [16, 32],
                                "head_pool_ops": [0, 1]},
            "n_initial_points": 2, "gp_rounds": 1, "trials_per_round": 2,
            "max_concurrent_trials": 2, "gp_seed": 3, "train_seed": 0,
            "budget_gate": {"probe_steps": 2, "probe_val_rows": 10, "safety_factor": 1.5},
            "max_total_walltime_h": 400})
        cfg["search"]["trial"].update({"passes": 1.0, "epochs": 2, "ncpus": 1, "mem": "8gb",
                                       "walltime": "00:20:00", "queue": "test",
                                       "max_resubmits": 1})
        cfg["best_fit"].update({"split_seed": 1, "member_seeds": [1, 2], "passes": 1.0,
                                "epochs": 2, "patience_epochs": 10, "n_posterior_draws": 16,
                                "ncpus": 1, "mem": "8gb", "walltime": "00:20:00",
                                "queue": "test", "max_resubmits": 1})
        cfg["sim"].update({"ncpus": 2, "concurrency": 4, "test_walltime": "00:10:00",
                           "walltime": "00:20:00"})
        cfg["bank"]["verify"] = {"ncpus": 1, "mem": "1gb", "walltime": "00:10:00"}
        cfg["loader"] = loader or {"buffer_shards": 2}
        # the sequencer's jobs: a 3-minute walltime (a budget of 135 s), so the
        # chain hands over between drivers several times on the way (D-098)
        cfg["driver"] = driver or {"queue": "test", "ncpus": 1, "mem": "1gb",
                                   "walltime": "00:03:00", "poll_seconds": 1}
        cfg["resources"] = {"max_cores": int(cap)}
        if search:
            for k, v in search.items():
                if isinstance(v, dict) and isinstance(cfg["search"].get(k), dict):
                    cfg["search"][k].update(v)
                else:
                    cfg["search"][k] = v
        if best_fit:
            cfg["best_fit"].update(best_fit)
        if extra:
            extra(cfg)
        self.cfg = cfg
        self.path = os.path.join(self.root, "%s.json" % TAG)
        self.write_cfg()
        self.spool = os.path.join(self.root, "spool")
        self.work_root = cfg["paths"]["work_root"]

    # ------------------------------------------------------------------ setup
    def write_cfg(self):
        with open(self.path, "w") as fh:
            json.dump(self.cfg, fh, indent=1, sort_keys=True)

    def _patch_stubs(self):
        g_man = os.path.join(self.golden, "cohort_manifest.json")
        g_rec = os.path.join(self.golden, "REEXTRACTION_RECORD.json")
        p1 = os.path.join(self.sbix, "extractor", "launch_stage_d.sh")
        t = open(p1).read()
        old = """printf '{"manifest_version": 3, "n_wells": 2, "stub": true}\\n' > "$ROOT/cohort_manifest.json\""""
        assert old in t, "the S1 stub changed: %s" % p1
        t = t.replace(old, 'cp "%s" "$ROOT/cohort_manifest.json"' % g_man)
        with open(p1, "w") as fh:
            fh.write(t)
        p2 = os.path.join(self.sbix, "sim_reextract", "launch_sim_reextract.sh")
        t = open(p2).read()
        anchor = ('[ -f "$COHORT_MANIFEST.sha256" ] || { echo "ABORT: $COHORT_MANIFEST has no '
                  '.sha256 sidecar; refusing to freeze an unchecked copy"; exit 3; }\n')
        assert anchor in t, "the S2 stub changed: %s" % p2
        freeze = (anchor +
                  '# the freeze of Sbi-extractor\'s real launcher (e9aeb55, lines 114-137)\n'
                  'FROZEN_DIR="$(cd .. && pwd)/artifacts/cohort_manifest_${COHORT_TAG}"\n'
                  'FROZEN="$FROZEN_DIR/cohort_manifest.json"\n'
                  'if [ ! -f "$FROZEN" ]; then\n'
                  '    mkdir -p "$FROZEN_DIR"\n'
                  '    cp "$COHORT_MANIFEST" "$FROZEN" && cp "$COHORT_MANIFEST.sha256" "$FROZEN.sha256" '
                  '|| { echo "ABORT: could not freeze the manifest"; exit 2; }\n'
                  '    echo "[launch] froze $COHORT_MANIFEST -> $FROZEN"\n'
                  'else\n'
                  '    got=$(sha256sum "$FROZEN" | cut -d" " -f1); cm=$(sha256sum "$COHORT_MANIFEST" '
                  '| cut -d" " -f1)\n'
                  '    [ "$cm" = "$got" ] || { echo "ABORT: the frozen $FROZEN (sha256 ${got:0:16}) is '
                  'not $COHORT_MANIFEST (sha256 ${cm:0:16})"; exit 3; }\n'
                  '    echo "[launch] frozen manifest: $FROZEN"\n'
                  'fi\n')
        t = t.replace(anchor, freeze)
        old2 = """printf '{"record": "sim_reextraction", "stub": true, "tasks": []}\\n' > "$OUT_ROOT/REEXTRACTION_RECORD.json\""""
        assert old2 in t, "the S2 stub's gate changed: %s" % p2
        t = t.replace(old2, 'cp "%s" "$OUT_ROOT/REEXTRACTION_RECORD.json"' % g_rec)
        with open(p2, "w") as fh:
            fh.write(t)

    def restore_inputs(self):
        """S1-S3's artefacts back as the fixture made them (with sidecars) and
        the frozen manifest: a data set whose S1-S3 are done (the Giulia
        situation of block 8)."""
        g = self.golden
        for name, src in (("real_extraction", "cohort_manifest.json"),
                          ("sim_redetection", "REEXTRACTION_RECORD.json"),
                          ("label_freeze", "label_freeze.json")):
            dst = self._arts[name]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(g, src), dst)
            write_sidecar(dst)
        os.makedirs(os.path.dirname(self.frozen), exist_ok=True)
        shutil.copy2(self._arts["real_extraction"], self.frozen)
        shutil.copy2(self._arts["real_extraction"] + ".sha256", self.frozen + ".sha256")

    # --------------------------------------------------------------- running
    def env(self, **extra):
        env = dict(os.environ)
        env["PATH"] = FAKE + os.pathsep + self.fakebin + os.pathsep + env.get("PATH", "")
        env["FAKE_PBS_SPOOL"] = self.spool
        env["FAKE_PBS_DELAY"] = "0.3"
        env["SKIP_CONDA"] = "1"
        env["PIPELINE_SKIP_CONDA"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = PIPE + os.pathsep + env.get("PYTHONPATH", "")
        for k in ("DRYRUN", "WALLTIME", "TEST_INDEX", "RESUME", "COHORT_TAG", "STUB_REFUSE",
                  "STUB_ONE_TASK", "STUB_TRACK", "STUB_N_TASKS", "STUB_NEVER_START",
                  "STUB_FAIL_TASK", "PIPELINE_JOBS_DIR", "WORK_ROOT", "CHAIN_NAME",
                  "AVOID_FILE", "AVOID_HOLD_S", "PROFILE", "QUEUE", "CONCURRENCY", "PBS_JOBID"):
            env.pop(k, None)
        env.update(extra)
        return env

    def sh(self, args, timeout=600, **env_extra):
        """bash hpc/pipeline/run_pipeline.sh --config <file> <args>: (rc, output)."""
        cmd = ["bash", os.path.join(PIPE, "run_pipeline.sh"), "--config", self.path] + list(args)
        p = subprocess.run(cmd, cwd=self.root, env=self.env(**env_extra), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
        return p.returncode, p.stdout

    def tool(self, rel, args, timeout=600, **env_extra):
        """python3 hpc/pipeline/<rel> <args>: (rc, output)."""
        cmd = [PYTHON, os.path.join(PIPE, rel)] + list(args)
        p = subprocess.run(cmd, cwd=self.root, env=self.env(**env_extra), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
        return p.returncode, p.stdout

    # ------------------------------------------------------------ observing
    def state(self):
        p = os.path.join(self.work_root, "state.json")
        if not os.path.isfile(p):
            return None
        for _ in range(20):
            try:
                with open(p) as fh:
                    return json.load(fh)
            except ValueError:
                time.sleep(0.2)
        raise RuntimeError("state.json does not parse")

    def statuses(self):
        s = self.state()
        return {} if s is None else dict((k, v["status"]) for k, v in s["stages"].items())

    def log_text(self):
        p = os.path.join(self.work_root, "pipeline.log")
        return open(p).read() if os.path.isfile(p) else ""

    def jobs(self):
        out = []
        for p in glob.glob(os.path.join(self.spool, "*.json")):
            b = os.path.basename(p)[:-5]
            if b.isdigit():
                for _ in range(20):
                    try:
                        with open(p) as fh:
                            out.append(json.load(fh))
                        break
                    except ValueError:
                        time.sleep(0.1)
        return sorted(out, key=lambda r: r["num"])

    def live_jobs(self):
        return [j for j in self.jobs() if j.get("state") not in ("F",)]

    def chain_end(self):
        p = os.path.join(self.work_root, "driver", "CHAIN_END")
        return open(p).read().strip() if os.path.isfile(p) else None

    def wait_for(self, pred, timeout, every=1.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            v = pred()
            if v:
                return v
            time.sleep(every)
        return None

    def wait_chain_end(self, timeout=1800):
        """Until the chain has ended on purpose (driver/CHAIN_END) and no job
        of the fake PBS is live; returns the CHAIN_END line or None."""
        end = self.wait_for(lambda: self.chain_end(), timeout, every=2.0)
        if end is None:
            return None
        self.wait_for(lambda: not self.live_jobs(), 300, every=1.0)
        return end

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def job_argv(rec):
    """The argv file of a run_argv.pbs job, read back (one token per line)."""
    f = (rec.get("vars") or {}).get("ARGV_FILE")
    if not f or not os.path.isfile(f):
        return None
    with open(f) as fh:
        return [line.rstrip("\n") for line in fh if line.rstrip("\n") != ""]


def flag_value(argv, flag, n=1):
    """The value(s) after the LAST occurrence of `flag` in argv (or None)."""
    if flag not in argv:
        return None
    i = len(argv) - 1 - argv[::-1].index(flag)
    vals = argv[i + 1:i + 1 + n]
    return vals[0] if n == 1 else vals


def runner_argv_from_log(text):
    """The runner's command line as the tuner prints it in a trial's job log
    (the line holding run_joint_arms.py), split on blanks; None if absent."""
    for line in text.splitlines():
        if "run_joint_arms.py" in line and "--arm" in line:
            i = line.find("run_joint_arms.py")
            j = line.rfind(" ", 0, i)
            return line[j + 1:].split()
    return None
