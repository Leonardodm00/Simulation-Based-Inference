#!/usr/bin/env python3
"""The stand-in davinci for the pipeline's tests (blocks 2, 3 and 8): a
copy of the Sbi-extractor tools, a fake conda, and a synthetic data set
built from Sbi-extractor's own bank fixture (`export_windows/
bank_plan_fixture.py`: three simulated tasks, a G2-style record, a label
freeze, a frozen cohort manifest, a stand-in simulator registry and the
real arm's archives).

Not pipeline code: a test fixture. The Sbi-extractor checkout is taken
from the SBIX_CLONE environment variable, else from a few known places;
`sbix_clone()` returns None when there is none (the tests then skip).

Pure ASCII, LF only; stdlib plus the fixture's numpy.
"""

import hashlib
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_KNOWN = ("/home/claude/leonardodm00/sbi-extractor",
          os.path.expanduser("~/repos/Sbi-extractor"),
          os.path.expanduser("~/Sbi-extractor"))

TOOL_DIRS = ("export_windows", "sim_reextract", "extractor")


def sbix_clone():
    p = os.environ.get("SBIX_CLONE")
    cands = [p] if p else list(_KNOWN)
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "export_windows", "launch_bank.sh")):
            return c
    return None


def _ignore(dirname, names):
    return [n for n in names if n == "__pycache__" or n.endswith((".pyc", ".npz", ".log"))
            or n in ("profiles_local",)]


def build_sbix(root, clone):
    """A stand-in Sbi-extractor checkout under `root`: the root modules,
    env.sh, the three tool folders; an empty artifacts/."""
    sbix = os.path.join(root, "sbix")
    os.makedirs(sbix)
    for f in os.listdir(clone):
        src = os.path.join(clone, f)
        if os.path.isfile(src) and (f.endswith(".py") or f == "env.sh"):
            shutil.copy2(src, os.path.join(sbix, f))
    for d in TOOL_DIRS:
        shutil.copytree(os.path.join(clone, d), os.path.join(sbix, d), ignore=_ignore)
    os.makedirs(os.path.join(sbix, "artifacts"))
    return sbix


def fake_conda(root):
    """A fake `conda` on PATH whose `activate` puts this interpreter's folder
    first on PATH (the Sbi-extractor job scripts' environment block): the
    builder of their own smoke test."""
    b = os.path.join(root, "fakebin")
    base = os.path.join(root, "fakebase")
    os.makedirs(b)
    os.makedirs(os.path.join(base, "etc", "profile.d"))
    with open(os.path.join(b, "conda"), "w") as fh:
        fh.write('#!/bin/bash\nif [ "$1" = info ] && [ "$2" = --base ]; then echo "%s"; '
                 'exit 0; fi\nif [ "$1" = shell.bash ] && [ "$2" = hook ]; then '
                 'echo "source %s/etc/profile.d/conda.sh"; exit 0; fi\n'
                 'echo "fake conda: $*" >&2\n' % (base, base))
    with open(os.path.join(base, "etc", "profile.d", "conda.sh"), "w") as fh:
        fh.write('conda() {\n  if [ "$1" = activate ]; then\n'
                 '    export PATH="%s:$PATH"; export CONDA_DEFAULT_ENV="$2"; return 0\n'
                 '  fi\n  command conda "$@"\n}\n' % os.path.dirname(sys.executable))
    os.chmod(os.path.join(b, "conda"), 0o755)
    return b


def _copy_with_sidecar(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    with open(dst + ".sha256", "w") as fh:
        fh.write("%s  %s\n" % (h, os.path.basename(dst)))
    return h


def write_dataset(root, sbix, tag="synth", extra_topos=None):
    """The synthetic data set as the pipeline expects it: S1's manifest at
    `<extract_root>/cohort_manifest.json`, S2's record at `<out_root>/
    REEXTRACTION_RECORD.json`, S3's freeze at `<sbix>/artifacts/label_axes_
    <tag>.json`, the frozen manifest copy under artifacts/, the registry as
    `sim_main`. Returns the paths the configuration needs."""
    # the fixture module from the ORIGINAL checkout (one import per process), with
    # the DSN tree its imports need named explicitly
    clone = sbix_clone()
    os.environ.setdefault("SBI_HPC_DIR", os.path.join(_REPO, "hpc"))
    for p in (os.path.join(clone, "export_windows"), clone, os.path.join(clone, "sim_reextract")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import bank_plan_fixture as FX
    fx_root = os.path.join(root, "fx")
    os.makedirs(fx_root)
    fx = FX.write_fixture(fx_root, extra_topos=extra_topos)
    extract_root = os.path.join(root, "extracted")
    _copy_with_sidecar(fx["manifest"], os.path.join(extract_root, "cohort_manifest.json"))
    frozen = os.path.join(sbix, "artifacts", "cohort_manifest_%s" % tag, "cohort_manifest.json")
    _copy_with_sidecar(fx["manifest"], frozen)
    freeze_rel = "artifacts/label_axes_%s.json" % tag
    _copy_with_sidecar(fx["freeze"], os.path.join(sbix, freeze_rel))
    # sim_main holds both the campaign folders and the simulator's registry
    # modules, as Giulia_Astro does on davinci (the profile's sim_dir is the
    # same folder): the registry's two files are copied beside the campaigns
    sim_main = os.path.join(fx_root, "sim")
    for f in os.listdir(os.path.join(fx_root, "registry")):
        shutil.copy2(os.path.join(fx_root, "registry", f), os.path.join(sim_main, f))
    # the freeze tool reads each task's job_args.json for its conn_rule (the Giulia
    # tasks declare "flat"); the fixture's job_args has no such key: add it
    for t in fx["tasks"]:
        ja = os.path.join(t["campaign_dir"], "job_args.json")
        d = json.load(open(ja))
        d["conn_rule"] = "flat"
        with open(ja, "w") as fh:
            json.dump(d, fh, indent=2, sort_keys=True)
    return {"fixture": fx, "extract_root": extract_root, "out_root": os.path.dirname(fx["record"]),
            "sim_main": sim_main, "freeze_out": freeze_rel,
            "expected_p": len(FX.ACTIVE) + 1, "n_tasks": len(fx["tasks"])}


def premake_bank_record(sim_dir, body='{"pre_made": true}\n'):
    """BANK_RECORD.json + .sha256 and the certificate BANK_VERIFIED.json +
    .sha256 naming the record's sha256 -- S5's artefact of record since
    2026-10-18 (D-094): a simulated bank the stages treat as done."""
    if not os.path.isdir(sim_dir):
        os.makedirs(sim_dir)
    rec = os.path.join(sim_dir, "BANK_RECORD.json")
    with open(rec, "w") as fh:
        fh.write(body)
    h = hashlib.sha256(open(rec, "rb").read()).hexdigest()
    with open(rec + ".sha256", "w") as fh:
        fh.write("%s  BANK_RECORD.json\n" % h)
    cert = os.path.join(sim_dir, "BANK_VERIFIED.json")
    with open(cert, "w") as fh:
        json.dump({"certificate": "bank_verified", "record_sha256": h, "pre_made": True}, fh)
    hc = hashlib.sha256(open(cert, "rb").read()).hexdigest()
    with open(cert + ".sha256", "w") as fh:
        fh.write("%s  BANK_VERIFIED.json\n" % hc)
    return rec


def synthetic_config(root, sbix, sbi_dir, ds, tag="synth", bank_walltime="00:30:00"):
    """A pipeline configuration (config v6) on the stand-in, from the
    committed Giulia file's shape."""
    here = os.path.dirname(os.path.abspath(__file__))
    giulia = os.path.join(here, "..", "projects", "giulia", "giulia.json")
    sys.path.insert(0, os.path.join(here, ".."))
    import config as C
    with open(giulia, "rb") as fh:
        raw = json.loads(fh.read().decode("ascii"))
    cfg = C._strip_docs(raw)
    cfg["dataset"] = tag
    cfg["paths"] = {"sbix_dir": sbix, "sbi_dir": sbi_dir,
                    "work_root": os.path.join(root, "work_root"),
                    "banks_root": os.path.join(root, "banks"),
                    "runs_root": os.path.join(root, "runs"),
                    "tune_root": os.path.join(root, "tune")}
    cfg["real"] = {"cohort_config": os.path.join(root, "cohort_config.json"), "cohort_tag": tag,
                   "extract_root": ds["extract_root"]}
    cfg["sim"].update({"sim_main": ds["sim_main"], "out_root": ds["out_root"],
                       "tools": os.path.join(root, "tools"), "queue": "test"})
    cfg["freeze"]["out"] = ds["freeze_out"]
    # the fixture's iteration files carry no Weibull kernel axis: nothing to exclude
    cfg["freeze"]["exclude"] = {}
    cfg["bank"].update({"expected_p": int(ds["expected_p"]), "sample_seed": 11,
                        "bootstrap_resamples": 200, "bootstrap_seed": 12, "bytes_sample_rows": 50,
                        "rows_per_shard": 100, "queue": "test", "concurrency": 20,
                        "walltime": bank_walltime})
    return cfg


# ---------------------------------------------------------------------------
# Stubs of the S1 and S2 launchers (block 3's test): the real launchers'
# printed lines and job structure, the jobs through qsub (the fake PBS),
# the artefacts written by the jobs. The real launchers need the recordings
# and the ANN tools (davinci); their own suites cover them.
# ---------------------------------------------------------------------------

_STAGE_D_STUB = r'''#!/bin/bash
# STUB of Sbi-extractor/extractor/launch_stage_d.sh (the pipeline's tests)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="${1:?EXTRACT_ROOT}"
: "${CONFIG:?}"; : "${COHORT_TAG:?}"; : "${SBI_HPC_DIR:?}"
[ -f "$CONFIG" ] || { echo "ABORT: config not found: $CONFIG"; exit 2; }
if [ -f "$ROOT/cohort_manifest.json" ]; then echo "ABORT: $ROOT already holds a cohort_manifest.json; pick a fresh root."; exit 3; fi
if [ "${STUB_REFUSE:-}" = "flags" ]; then
    echo "NEW COHORT: wrote extraction_flags_${COHORT_TAG}.sh from the config's cohort block:"
    echo "       Review it, COMMIT it (git add extraction_flags_${COHORT_TAG}.sh && git commit), then re-run this launcher."
    echo "       Nothing was submitted."
    exit 3
fi
mkdir -p out
echo "########################################################################"
echo "# config       : $CONFIG"
echo "# cohort tag   : $COHORT_TAG"
echo "# extract root : $ROOT   (config declares: $ROOT)"
echo "# wells        : 2   -> array 0-1   (out/extraction_manifest_${COHORT_TAG}.tsv)"
echo "# env          : ${ENV_NAME:-sbi_export}"
if [ "${DRYRUN:-0}" = "1" ]; then
    # as the real launcher: both qsub lines printed, nothing submitted
    echo "  qsub -J 0-1 -N chsub_mea_array_${COHORT_TAG} -o out/chsub_mea_array_${COHORT_TAG}_^array_index^.log -v ENV_NAME=${ENV_NAME:-sbi_export} run_extractor_array_mea.pbs"
    echo "  qsub -W depend=afterok:<array id> -N cohort_manifest_${COHORT_TAG} -o out/cohort_manifest_${COHORT_TAG}.log -v EXTRACT_ROOT=${ROOT} run_cohort_manifest.pbs"
    echo; echo "(DRYRUN -- nothing was submitted. Re-run without DRYRUN=1.)"; exit 0
fi
cat > out/stub_array_${COHORT_TAG}.sh <<EOS
#!/bin/bash
echo "[array] well \${PBS_ARRAY_INDEX} extracted (stub)"
EOS
cat > out/stub_agg_${COHORT_TAG}.sh <<EOS
#!/bin/bash
mkdir -p "$ROOT"
if [ "${STUB_REFUSE:-}" = "agg" ]; then
    echo "REFUSED: 1 well(s) could not be used:"
    echo "  well 1 (array index 1): no archive"
    echo "[job] cohort_manifest_exit=1 finished=now"
    exit 1
fi
printf '{"manifest_version": 3, "n_wells": 2, "stub": true}\n' > "$ROOT/cohort_manifest.json"
h=\$(sha256sum "$ROOT/cohort_manifest.json" | cut -c1-64)
echo "\$h  cohort_manifest.json" > "$ROOT/cohort_manifest.json.sha256"
echo "wrote $ROOT/cohort_manifest.json  sha256 \${h:0:16}"
echo "[job] cohort_manifest_exit=0 finished=now"
EOS
echo "  qsub -J 0-1 -N chsub_mea_array_${COHORT_TAG} -o out/chsub_mea_array_${COHORT_TAG}_^array_index^.log -v ENV_NAME=${ENV_NAME:-sbi_export} run_extractor_array_mea.pbs"
arr=$(qsub -J 0-1 -N "chsub_mea_array_${COHORT_TAG}" -o "out/chsub_mea_array_${COHORT_TAG}_^array_index^.log" -v "ENV_NAME=${ENV_NAME:-sbi_export}" "out/stub_array_${COHORT_TAG}.sh") || { echo "ABORT: array qsub failed"; exit 2; }
echo "  array : $arr"
agg=$(qsub -W "depend=afterok:${arr}" -N "cohort_manifest_${COHORT_TAG}" -o "out/cohort_manifest_${COHORT_TAG}.log" -v "EXTRACT_ROOT=${ROOT}" "out/stub_agg_${COHORT_TAG}.sh") || { echo "ABORT: aggregation qsub failed"; exit 2; }
echo "  agg   : $agg   (runs once the array has ended, whatever its tasks' exit codes)"
echo
echo "watch:   qstat -u \$USER"
'''

_SIM_REEXTRACT_STUB = r'''#!/bin/bash
# STUB of Sbi-extractor/sim_reextract/launch_sim_reextract.sh (the pipeline's tests)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
MODE="${1:-}"
case "$MODE" in plan|test|array|gate) ;; *) echo "usage"; exit 2 ;; esac
: "${COHORT_TAG:?}"; : "${COHORT_MANIFEST:?}"; : "${SIM_MAIN:?}"; : "${OUT_ROOT:?}"; : "${TOOLS:?}"
: "${CAMPAIGN_GLOB:?}"; : "${QUEUE:?}"; : "${NCPUS:?}"; : "${CONCURRENCY:?}"; : "${SBI_HPC_DIR:?}"
[ -f "profiles/${COHORT_TAG}.sh" ] || { echo "ABORT: no profile for COHORT_TAG=$COHORT_TAG (profiles/${COHORT_TAG}.sh)"; exit 2; }
[ -f "$COHORT_MANIFEST" ] || { echo "ABORT: cohort manifest not found: $COHORT_MANIFEST"; exit 2; }
[ -f "$COHORT_MANIFEST.sha256" ] || { echo "ABORT: $COHORT_MANIFEST has no .sha256 sidecar; refusing to freeze an unchecked copy"; exit 3; }
mkdir -p out "$COHORT_TAG"
if [ "${STUB_TRACK:-0}" = "1" ]; then
    # D-113's tests: STUB_N_TASKS tasks whose completion is a file in
    # $OUT_ROOT/stub_tasks; the jobs start through the pipeline's job protocol
    # when PIPELINE_JOBS_DIR is set, as the real job scripts do. STUB_NEVER_START
    # (a list of tasks) exit 96 before any work, every time; STUB_FAIL_TASK=<i>:<k>
    # fails task i in its first k runs after it started. The gate refuses as
    # sim_reextract_gate.py does, one block per bad task.
    T="${STUB_N_TASKS:-3}"
    D="$OUT_ROOT/stub_tasks"
    mkdir -p "$D"
    todo=()
    for i in $(seq 0 $((T - 1))); do
        if [ "${RESUME:-0}" = "1" ] && [ -f "$D/task_$i.done" ]; then continue; fi
        todo+=("$i")
    done
    N=${#todo[@]}
    [ "${RESUME:-0}" = "1" ] && echo "[plan] tasks complete  : $((T - N)) kept as done (--resume)"
    echo "[plan] tasks to run    : $N (array 0-$((N - 1)))"
    [ "$MODE" = "plan" ] && { echo "[launch] plan only: $N task(s) in tasks.tsv; nothing submitted"; exit 0; }
    PROTO='if [ -n "${PIPELINE_JOBS_DIR:-}" ]; then
    source "$PIPELINE_JOBS_DIR/stage_job.sh" || exit 2
    pipeline_stage_job_start "${ENV_NAME:-sbi_export}" || exit $?
    trap pipeline_stage_job_end EXIT
fi'
    printf '%s\n' "${todo[@]}" > "out/stub_todo_${COHORT_TAG}.txt"
    cat > "out/stub_track_task_${COHORT_TAG}.sh" <<EOS
#!/bin/bash
$PROTO
if [ -n "\${PBS_ARRAY_INDEX:-}" ]; then
    i=\$(sed -n "\$((PBS_ARRAY_INDEX + 1))p" "$PWD/out/stub_todo_${COHORT_TAG}.txt")
else
    i="\${STUB_TASK:-\$(sed -n 1p "$PWD/out/stub_todo_${COHORT_TAG}.txt")}"
fi
case ",\${STUB_NEVER_START:-}," in *",\$i,"*) echo "RUN_ARGV ENV FAULT: host \$(hostname): stub"; exit 96 ;; esac
echo started > "$D/task_\$i.env"
n=\$(cat "$D/task_\$i.runs" 2>/dev/null || echo 0); n=\$((n + 1)); echo "\$n" > "$D/task_\$i.runs"
case "\${STUB_FAIL_TASK:-}" in "\$i":*) if [ "\$n" -le "\${STUB_FAIL_TASK#*:}" ]; then echo "[mea] task \$i failed (stub, run \$n)"; exit 1; fi ;; esac
touch "$D/task_\$i.done"
echo "[mea] task \$i: 12/12 iters processed (stub)"
EOS
    cat > "out/stub_track_gate_${COHORT_TAG}.sh" <<EOS
#!/bin/bash
$PROTO
mkdir -p "$OUT_ROOT"
bad=0
for i in \$(seq 0 $((T - 1))); do
    [ -f "$D/task_\$i.done" ] && continue
    bad=\$((bad + 1))
    echo "[gate] FAIL campaign_x/sweep_cpu_task000\$i"
    echo "[gate]      - no mea_manifest.json (the task did not finish)"
    [ -f "$D/task_\$i.env" ] || echo "[gate]      - no mea_env.json (the job script that ran did not write one)"
done
if [ "\$bad" -gt 0 ]; then
    echo "[gate] read 0 mea_iter_*.npz file(s) across $T task(s); \$bad task(s) bad"
    echo "[gate] REFUSED -- \$bad bad task(s), nothing written. Re-run those tasks (plan --resume, then the array) and run this gate again."
    echo "[job] sim_reextract_gate_exit=1 finished=now"
    exit 1
fi
printf '{"record": "sim_reextraction", "stub": true, "tasks": $T}\n' > "$OUT_ROOT/REEXTRACTION_RECORD.json"
h=\$(sha256sum "$OUT_ROOT/REEXTRACTION_RECORD.json" | cut -c1-64)
echo "\$h  REEXTRACTION_RECORD.json" > "$OUT_ROOT/REEXTRACTION_RECORD.json.sha256"
echo "wrote $OUT_ROOT/REEXTRACTION_RECORD.json  sha256 \${h:0:16}"
echo "[job] sim_reextract_gate_exit=0 finished=now"
EOS
    DEP=()
    if [ "$MODE" = "test" ]; then
        [ -n "${WALLTIME:-}" ] || { echo "ABORT: WALLTIME=hh:mm:ss is required"; exit 3; }
        j=$(qsub -N "g2_test" -l "walltime=${WALLTIME}" -o "out/stub_test_${COHORT_TAG}.log" -v "STUB_TASK=${todo[0]}" "out/stub_track_task_${COHORT_TAG}.sh") || { echo "ABORT: qsub failed"; exit 2; }
        echo "  job   : $j"; exit 0
    fi
    if [ "$MODE" = "array" ]; then
        [ "$N" -ge 1 ] || { echo "ABORT: tasks.tsv is empty (every task complete?); nothing to submit"; exit 3; }
        [ -n "${WALLTIME:-}" ] || { echo "ABORT: WALLTIME=hh:mm:ss is required"; exit 3; }
        if [ "$N" -gt 1 ]; then
            j=$(qsub -J "0-$((N - 1))%${CONCURRENCY}" -N "g2_mea" -l "walltime=${WALLTIME}" -o "out/stub_mea_${COHORT_TAG}_^array_index^.log" "out/stub_track_task_${COHORT_TAG}.sh") || { echo "ABORT: qsub failed"; exit 2; }
        else
            j=$(qsub -N "g2_mea" -l "walltime=${WALLTIME}" -o "out/stub_mea_${COHORT_TAG}_0.log" "out/stub_track_task_${COHORT_TAG}.sh") || { echo "ABORT: qsub failed"; exit 2; }
        fi
        echo "  job   : $j"
        DEP=(-W "depend=afterok:${j}")
    fi
    g=$(qsub ${DEP[@]+"${DEP[@]}"} -N "g2_gate" -o "out/sim_reextract_gate_${COHORT_TAG}.log" "out/stub_track_gate_${COHORT_TAG}.sh") || { echo "ABORT: the gate's qsub failed"; exit 2; }
    echo "  gate  : $g"
    exit 0
fi
N=2
if [ "${RESUME:-0}" = "1" ]; then
    echo "[plan] tasks complete  : 1 kept as done (--resume)"
    N=$(( ${STUB_ONE_TASK:-0} == 1 ? 0 : 1 ))
fi
echo "[plan] tasks to run    : $N (array 0-$((N - 1)))"
[ "$MODE" = "plan" ] && { echo "[launch] plan only: $N task(s) in tasks.tsv; nothing submitted"; exit 0; }
if [ "$MODE" = "test" ] || [ "$MODE" = "array" ]; then
    [ "$N" -ge 1 ] || { echo "ABORT: tasks.tsv is empty (every task complete?); nothing to submit"; exit 3; }
    [ -n "${WALLTIME:-}" ] || { echo "ABORT: WALLTIME=hh:mm:ss is required"; exit 3; }
    cat > "out/stub_task_${COHORT_TAG}.sh" <<EOS
#!/bin/bash
echo "[mea] 3 topos, 12/12 iters processed in 1 s (stub)"
EOS
    if [ "$MODE" = "test" ]; then
        arr=$(qsub -N "g2_test" -l "walltime=${WALLTIME}" -o "out/stub_test_${COHORT_TAG}.log" "out/stub_task_${COHORT_TAG}.sh") || { echo "ABORT: qsub failed"; exit 2; }
        echo "  job   : $arr"; exit 0
    fi
    arr=$(qsub -J "0-$((N - 1 > 0 ? N - 1 : 1))%${CONCURRENCY}" -N "g2_mea" -l "walltime=${WALLTIME}" -o "out/stub_mea_${COHORT_TAG}_^array_index^.log" "out/stub_task_${COHORT_TAG}.sh") || { echo "ABORT: qsub failed"; exit 2; }
    echo "  job   : $arr"
    DEP=(-W "depend=afterok:${arr}")
else
    DEP=()
fi
cat > "out/stub_gate_${COHORT_TAG}.sh" <<EOS
#!/bin/bash
mkdir -p "$OUT_ROOT"
if [ "${STUB_REFUSE:-}" = "gate" ]; then
    echo "[gate] FAIL campaign_x/sweep_task0001: 2 file(s) missing"
    echo "REFUSED: 1 task(s) incomplete (each named above)"
    echo "[job] sim_reextract_gate_exit=1 finished=now"
    exit 1
fi
printf '{"record": "sim_reextraction", "stub": true, "tasks": []}\n' > "$OUT_ROOT/REEXTRACTION_RECORD.json"
h=\$(sha256sum "$OUT_ROOT/REEXTRACTION_RECORD.json" | cut -c1-64)
echo "\$h  REEXTRACTION_RECORD.json" > "$OUT_ROOT/REEXTRACTION_RECORD.json.sha256"
echo "wrote $OUT_ROOT/REEXTRACTION_RECORD.json  sha256 \${h:0:16}"
echo "[job] sim_reextract_gate_exit=0 finished=now"
EOS
gate=$(qsub ${DEP[@]+"${DEP[@]}"} -N "g2_gate" -o "out/sim_reextract_gate_${COHORT_TAG}.log" "out/stub_gate_${COHORT_TAG}.sh") || { echo "ABORT: the gate's qsub failed"; exit 2; }
echo "  gate  : $gate"
'''


def fake_env_folder(folder, name, python):
    """A conda-like environment folder for the job protocol's tests (D-111,
    D-113): conda-meta/, bin/python (`python`, unless `hostname` is in
    FAKE_ENV_BROKEN_HOSTS: then it exits 1, an environment that cannot start
    on that node). As test_smoke_block7c's own helper."""
    p = os.path.join(folder, name)
    for d in ("conda-meta", "bin"):
        os.makedirs(os.path.join(p, d), exist_ok=True)
    with open(os.path.join(p, "bin", "python"), "w") as fh:
        fh.write("#!/bin/bash\nh=$(hostname 2>/dev/null)\n"
                 "case \",${FAKE_ENV_BROKEN_HOSTS:-},\" in *\",$h,\"*) "
                 "echo \"python: cannot start on $h\" >&2; exit 1 ;; esac\n"
                 "exec %s \"$@\"\n" % python)
    os.chmod(os.path.join(p, "bin", "python"), 0o755)
    return p


def stub_launchers(sbix, tag="synth"):
    """Replace the two launchers in the stand-in by the stubs and write the
    cohort's profile the S2 launcher requires."""
    for rel, body in (("extractor/launch_stage_d.sh", _STAGE_D_STUB),
                      ("sim_reextract/launch_sim_reextract.sh", _SIM_REEXTRACT_STUB)):
        p = os.path.join(sbix, rel)
        with open(p, "w") as fh:
            fh.write(body)
        os.chmod(p, 0o755)
    prof = os.path.join(sbix, "sim_reextract", "profiles", "%s.sh" % tag)
    with open(prof, "w") as fh:
        fh.write("# stand-in profile for COHORT_TAG=%s (the pipeline's tests)\n"
                 "PROFILE_WORK=\"%s\"\nPROFILE_FROZEN=\"cohort_manifest_%s\"\nPROFILE_JOB=\"g2\"\n"
                 % (tag, tag, tag))
    return prof
