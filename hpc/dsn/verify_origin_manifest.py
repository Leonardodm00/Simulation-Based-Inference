#!/usr/bin/env python3
"""
verify_origin_manifest.py -- is every row of ORIGIN_MANIFEST.tsv still true?

ORIGIN_MANIFEST.tsv is this directory's provenance record: one row per file
mirrored from the Deep-Summary-Network Main/ tree (plus cohort.py, new in
Stage D), giving the file's sha256 and its source path. Nothing checked it,
so it drifted: migration step 5b (fe9cb20, 2026-09-20) changed one line in
seven job scripts under hpc/, and their rows kept the step-1 sha until
2026-09-28. The smoke job printed only a row COUNT (manifest_rows=196), and a
stale row does not change a count.

Run, from anywhere:
    python3 verify_origin_manifest.py            # the manifest next to this file
    python3 verify_origin_manifest.py PATH.tsv   # another one; its rows resolve
                                                 # against PATH.tsv's directory

Exit 0 and one PASS line if every row's file exists and hashes as recorded.
Exit 1 and one line per problem otherwise. Standard library only; git is used,
when present, for the NOTE below and for nothing else. It therefore runs in
any environment, including one without torch or numpy.

What is checked is the CURRENT sha of each listed file against its row, not
the source blob: checking the source needs the retired DSN repository (the
command in the manifest header does that). A row after a "# EDITED" note
records the edited file's sha, so an edit that refreshed its row passes and
an edit that did not is reported as SHA CHANGED. The notes are not parsed.

A mismatch that disappears when CRLF is turned into LF is reported as such,
and is still a failure: the file on disk is not the recorded bytes, and on
Linux a CRLF job script does not run. A git checkout does not produce it --
hpc/.gitattributes pins eol=lf for everything under hpc/, on every
platform -- so it means a file was written outside git: by an editor, or by
a copy tool in text mode.

NOTE (not a failure): files that git tracks under this directory, that no row
claims and that are not among the five this repository added on purpose
(NOT_MIRRORED). Machine state (out/, caches, generated configs) is gitignored
and so never appears.

Its own test: python3 smoke_test_verify_origin_manifest.py (same directory).

HPC note (hpc-python-compat): pure ASCII, LF only.
"""
import hashlib
import os
import re
import subprocess
import sys

# In this directory, not from the DSN, and deliberately in no row: the record
# itself, this directory's README, the smoke job committed in migration step
# 1b, this script and its smoke test. Paths are relative to the manifest's
# directory.
NOT_MIRRORED = frozenset((
    "ORIGIN_MANIFEST.tsv",
    "README.md",
    "hpc/run_smoke_all.pbs",
    "smoke_test_verify_origin_manifest.py",
    "verify_origin_manifest.py",
))

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_of(path, crlf_to_lf=False):
    with open(path, "rb") as fh:
        data = fh.read()
    if crlf_to_lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_rows(manifest_path):
    """Return (rows, problems): rows is a list of (name, recorded_sha)."""
    rows, problems, seen = [], [], {}
    with open(manifest_path, "rb") as fh:
        raw_lines = fh.read().decode("utf-8", "replace").split("\n")
    for lineno, raw in enumerate(raw_lines, 1):
        row = raw.rstrip("\r")
        if not row.strip() or row.lstrip().startswith("#"):
            continue
        parts = row.split("\t")
        if len(parts) < 2:
            problems.append("line %d: not a tab-separated row: %r"
                            % (lineno, row[:60]))
            continue
        name, recorded = parts[0], parts[1]
        if not _SHA256.match(recorded):
            problems.append("line %d: %s: sha256 column is not 64 lowercase "
                            "hex digits: %r" % (lineno, name, recorded[:70]))
            continue
        if os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
            problems.append("line %d: %s: path leaves the manifest's "
                            "directory" % (lineno, name))
            continue
        if name in seen:
            problems.append("line %d: %s LISTED TWICE (first on line %d)"
                            % (lineno, name, seen[name]))
            continue
        seen[name] = lineno
        rows.append((name, recorded))
    return rows, problems


def check_rows(base, rows):
    """One problem string per listed file that is absent or has changed."""
    problems = []
    for name, recorded in rows:
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            problems.append("%-40s LISTED BUT ABSENT" % name)
            continue
        actual = sha256_of(path)
        if actual == recorded:
            continue
        if sha256_of(path, crlf_to_lf=True) == recorded:
            problems.append("%-40s SHA CHANGED  line endings only: equal to "
                            "the row after CRLF->LF (written outside git with "
                            "CRLF: an editor or a copy tool)"
                            % name)
        else:
            problems.append("%-40s SHA CHANGED  recorded %s  actual %s"
                            % (name, recorded[:16], actual[:16]))
    return problems


def tracked_files(base):
    """Paths git tracks under `base`, relative to it; None if git cannot say."""
    # cwd= rather than `git -C`, which needs git >= 1.8.5
    try:
        proc = subprocess.Popen(["git", "ls-files", "-z", "--", "."], cwd=base,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = proc.communicate()
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def unlisted_tracked(base, rows, manifest_name):
    """Tracked files that no row claims (None if git is unavailable)."""
    tracked = tracked_files(base)
    if tracked is None:
        return None
    listed = set(name for name, _ in rows)
    exempt = set(NOT_MIRRORED) | set([manifest_name])
    return sorted(p for p in tracked if p not in listed and p not in exempt)


def main(argv):
    manifest = os.path.abspath(argv[1]) if len(argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ORIGIN_MANIFEST.tsv")
    if not os.path.isfile(manifest):
        print("[manifest] ABSENT: %s" % manifest)
        return 1
    base = os.path.dirname(manifest)
    rows, problems = read_rows(manifest)
    problems += check_rows(base, rows)
    unlisted = unlisted_tracked(base, rows, os.path.basename(manifest))

    for p in problems:
        print("[manifest] %s" % p)
    if unlisted is None:
        print("[manifest] NOTE: git cannot list this directory (not a checkout, "
              "or no git); the unlisted-file check was skipped")
    elif unlisted:
        print("[manifest] NOTE: %d tracked file(s) here are in no row, so "
              "nothing claims anything about them:" % len(unlisted))
        for f in unlisted:
            print("[manifest]   %s" % f)
    if problems or not rows:
        print("[manifest] FAIL %d row(s) checked, %d problem(s)%s"
              % (len(rows), len(problems),
                 "" if rows else " -- and no rows at all"))
        return 1
    print("[manifest] PASS %d row(s), every file present and unchanged"
          % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
