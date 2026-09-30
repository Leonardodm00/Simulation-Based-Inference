#!/usr/bin/env python3
"""
smoke_test_verify_origin_manifest.py -- does verify_origin_manifest.py catch
what it exists to catch?

Every check runs the verifier as the smoke job does -- a separate python3
process, from a directory other than the manifest's -- on a small tree built
in a temporary directory, except V3, which replays the real drift: the
manifest committed at 49da750 against the files committed at 49da750 must
fail on exactly the seven rows migration step 5b (fe9cb20) left stale.

  V1  a clean tree passes: exit 0, "PASS 3 row(s)"
  V2  one edited file: exit 1, that row and no other, "SHA CHANGED"
  V3  the real drift at 49da750: exit 1, exactly the 7 step-5b rows
      (SKIP, not FAIL, if 49da750 is not in this clone's history)
  V4  a listed file removed: exit 1, "LISTED BUT ABSENT"
  V5  a file converted to CRLF: exit 1, "line endings only"
  V6  malformed manifests: short sha, one column, a duplicate row, a path
      with "..": each exit 1 with its own message
  V7  a manifest with no rows: exit 1, "no rows at all"
  V8  an absent manifest: exit 1, "ABSENT"
  V9  in a git checkout: a tracked file in no row is NAMED in a NOTE; an
      exempt name (README.md), an untracked file and a listed file are not;
      exit stays 0
  V10 outside any git checkout: the NOTE says the check was skipped; exit 0

Run (stdlib + git; no torch, numpy or conda needed), from anywhere:

    python3 smoke_test_verify_origin_manifest.py
    python3 smoke_test_verify_origin_manifest.py --verifier PATH   # another copy

Expect the last line "ALL 10 CHECKS PASSED" and exit 0 -- or "ALL 9 RUN
CHECKS PASSED, 1 SKIPPED" when this clone cannot see 49da750 (V3). Any FAIL
prints the verifier's own output under it and the exit is 1.

HPC note (hpc-python-compat): pure ASCII, LF only.
"""
import argparse
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
STEP5B_ROWS = sorted([
    "hpc/dsn_l3c_factorial.pbs", "hpc/launch_l3c_factorial.sh",
    "hpc/run_full_pipeline.pbs", "hpc/run_mea_joint_search.pbs",
    "hpc/run_plot_embeddings.pbs", "hpc/run_plot_traces.pbs",
    "hpc/run_refit.pbs",
])
DRIFT_COMMIT = "49da750"

RESULTS = []


def record(tag, ok, detail="", skipped=False):
    RESULTS.append((tag, ok, skipped))
    state = "SKIP" if skipped else ("PASS" if ok else "FAIL")
    print("%-4s %s%s" % (tag, state, ("  " + detail) if detail else ""))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "wb") as fh:
        fh.write(data)


def make_tree(root, extra_rows=""):
    """Three listed files (one nested) + README.md (exempt, unlisted)."""
    files = {
        "a.txt": b"alpha\n",
        "sub/b.pbs": b"#!/bin/bash\necho beta\n",
        "c.py": b"print('gamma')\n",
        "README.md": b"not mirrored\n",
    }
    for name, data in files.items():
        write(os.path.join(root, name), data)
    lines = ["# test manifest", "# columns: path\tsha256\tsource"]
    for name in ("a.txt", "sub/b.pbs", "c.py"):
        lines.append("%s\t%s\tMain/%s" % (name, sha(files[name]), name))
    text = "\n".join(lines) + "\n" + extra_rows
    write(os.path.join(root, "ORIGIN_MANIFEST.tsv"), text.encode("ascii"))
    return os.path.join(root, "ORIGIN_MANIFEST.tsv")


def run_verifier(verifier, manifest, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen([sys.executable, verifier, manifest],
                            cwd=tempfile.gettempdir(), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = proc.communicate()
    return proc.returncode, out.decode("utf-8", "replace")


def problem_lines(out):
    """Verifier lines that name a problem (not NOTE, PASS, FAIL summary)."""
    keep = []
    for ln in out.splitlines():
        body = ln[len("[manifest] "):] if ln.startswith("[manifest] ") else ln
        if body.startswith(("PASS ", "FAIL ", "NOTE", "  ")):
            continue
        keep.append(body)
    return keep


def git(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)


def show_on_fail(ok, out):
    if not ok:
        for ln in out.splitlines():
            print("       | " + ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier",
                    default=os.path.join(HERE, "verify_origin_manifest.py"))
    args = ap.parse_args()
    verifier = os.path.abspath(args.verifier)
    print("smoke_test_verify_origin_manifest.py -- verifier: %s" % verifier)
    if not os.path.isfile(verifier):
        print("FAIL: verifier not found")
        return 1

    tmp = tempfile.mkdtemp(prefix="vom_smoke_")
    # keep git from finding a repository ABOVE the fixtures
    noparent = {"GIT_CEILING_DIRECTORIES": tmp}
    try:
        # V1
        root = os.path.join(tmp, "v1")
        man = make_tree(root)
        rc, out = run_verifier(verifier, man, noparent)
        ok = rc == 0 and "[manifest] PASS 3 row(s)" in out and not problem_lines(out)
        record("V1", ok, "rc=%d" % rc)
        show_on_fail(ok, out)

        # V2
        root = os.path.join(tmp, "v2")
        man = make_tree(root)
        write(os.path.join(root, "sub/b.pbs"), b"#!/bin/bash\necho BETA\n")
        rc, out = run_verifier(verifier, man, noparent)
        probs = problem_lines(out)
        ok = (rc == 1 and len(probs) == 1 and probs[0].startswith("sub/b.pbs")
              and "SHA CHANGED" in probs[0] and "recorded" in probs[0]
              and "FAIL 3 row(s) checked, 1 problem(s)" in out)
        record("V2", ok, "rc=%d problems=%d" % (rc, len(probs)))
        show_on_fail(ok, out)

        # V3 -- the real drift, replayed from this clone's history
        top = git(["rev-parse", "--show-toplevel"], HERE)
        have = (top.returncode == 0 and git(
            ["cat-file", "-e", DRIFT_COMMIT + "^{commit}"], HERE).returncode == 0)
        if not have:
            record("V3", True, "commit %s not in this clone's history "
                   "(shallow clone, or not a checkout)" % DRIFT_COMMIT, skipped=True)
        else:
            # hpc/.gitattributes already pins LF; the two -c flags keep V3
            # about the drift even in a clone configured to override it
            arch = git(["-c", "core.autocrlf=false", "-c", "core.eol=lf",
                        "archive", "--format=tar", DRIFT_COMMIT, "hpc/dsn"],
                       top.stdout.decode().strip())
            root = os.path.join(tmp, "v3")
            os.makedirs(root)
            tf = tarfile.open(fileobj=io.BytesIO(arch.stdout))
            if hasattr(tarfile, "data_filter"):
                tf.extractall(root, filter="data")
            else:
                tf.extractall(root)
            tf.close()
            man = os.path.join(root, "hpc", "dsn", "ORIGIN_MANIFEST.tsv")
            rc, out = run_verifier(verifier, man, noparent)
            probs = problem_lines(out)
            named = sorted(p.split()[0] for p in probs)
            ok = (rc == 1 and named == STEP5B_ROWS
                  and all("SHA CHANGED" in p for p in probs))
            record("V3", ok, "rc=%d, %d row(s) reported at %s"
                   % (rc, len(probs), DRIFT_COMMIT))
            show_on_fail(ok, out)

        # V4
        root = os.path.join(tmp, "v4")
        man = make_tree(root)
        os.remove(os.path.join(root, "c.py"))
        rc, out = run_verifier(verifier, man, noparent)
        probs = problem_lines(out)
        ok = rc == 1 and len(probs) == 1 and "c.py" in probs[0] \
            and "LISTED BUT ABSENT" in probs[0]
        record("V4", ok, "rc=%d" % rc)
        show_on_fail(ok, out)

        # V5
        root = os.path.join(tmp, "v5")
        man = make_tree(root)
        write(os.path.join(root, "sub/b.pbs"), b"#!/bin/bash\r\necho beta\r\n")
        rc, out = run_verifier(verifier, man, noparent)
        probs = problem_lines(out)
        ok = rc == 1 and len(probs) == 1 and "line endings only" in probs[0]
        record("V5", ok, "rc=%d" % rc)
        show_on_fail(ok, out)

        # V6 -- four malformed manifests, one defect each
        cases = [
            ("short sha", "d.txt\t" + "a" * 63 + "\tMain/d.txt\n",
             "not 64 lowercase hex"),
            ("one column", "d.txt\n", "not a tab-separated row"),
            ("duplicate", "a.txt\t%s\tMain/a.txt\n" % sha(b"alpha\n"),
             "LISTED TWICE"),
            ("dotdot", "../x.txt\t%s\tMain/x.txt\n" % ("0" * 64),
             "leaves the manifest's directory"),
        ]
        bad = []
        for i, (label, extra, needle) in enumerate(cases):
            root = os.path.join(tmp, "v6_%d" % i)
            man = make_tree(root, extra_rows=extra)
            rc, out = run_verifier(verifier, man, noparent)
            if not (rc == 1 and needle in out):
                bad.append(label)
                show_on_fail(False, out)
        record("V6", not bad, "failed: %s" % bad if bad else "4 cases")

        # V7
        root = os.path.join(tmp, "v7")
        os.makedirs(root)
        man = os.path.join(root, "ORIGIN_MANIFEST.tsv")
        write(man, b"# only comments\n\n")
        rc, out = run_verifier(verifier, man, noparent)
        ok = rc == 1 and "no rows at all" in out
        record("V7", ok, "rc=%d" % rc)
        show_on_fail(ok, out)

        # V8
        rc, out = run_verifier(verifier, os.path.join(tmp, "nowhere.tsv"), noparent)
        ok = rc == 1 and "[manifest] ABSENT" in out
        record("V8", ok, "rc=%d" % rc)
        show_on_fail(ok, out)

        # V9 -- inside a git checkout
        root = os.path.join(tmp, "v9")
        man = make_tree(root)
        write(os.path.join(root, "extra_tracked.py"), b"x = 1\n")
        write(os.path.join(root, "sub/untracked.log"), b"noise\n")
        if git(["init", "-q"], root).returncode != 0:
            record("V9", False, "git init failed")
        else:
            git(["add", "a.txt", "sub/b.pbs", "c.py", "README.md",
                 "ORIGIN_MANIFEST.tsv", "extra_tracked.py"], root)
            rc, out = run_verifier(verifier, man, noparent)
            note = [ln for ln in out.splitlines() if ln.startswith("[manifest]   ")]
            named = [ln.split()[-1] for ln in note]
            ok = (rc == 0 and named == ["extra_tracked.py"]
                  and "1 tracked file(s) here are in no row" in out)
            record("V9", ok, "rc=%d, NOTE names %s" % (rc, named))
            show_on_fail(ok, out)

        # V10 -- outside any checkout (V1's tree has no .git, and the ceiling
        # stops the search above it)
        rc, out = run_verifier(verifier, os.path.join(tmp, "v1", "ORIGIN_MANIFEST.tsv"),
                               noparent)
        ok = rc == 0 and "unlisted-file check was skipped" in out
        record("V10", ok, "rc=%d" % rc)
        show_on_fail(ok, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for _, ok, sk in RESULTS if not ok and not sk)
    n_skip = sum(1 for _, _, sk in RESULTS if sk)
    if n_fail:
        print("FAILED: %d of %d checks" % (n_fail, len(RESULTS)))
        return 1
    if n_skip:
        print("ALL %d RUN CHECKS PASSED, %d SKIPPED"
              % (len(RESULTS) - n_skip, n_skip))
    else:
        print("ALL %d CHECKS PASSED" % len(RESULTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
