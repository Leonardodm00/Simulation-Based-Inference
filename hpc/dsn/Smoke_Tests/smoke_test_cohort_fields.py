"""
smoke_test_cohort_fields.py
===========================

Torch-free checks of the cohort fields added on 2026-10-01 for the Giulia
recordings (cohort.py only; make_mea_specs.py's use of them is check G of
smoke_test_mea_specs.py, which needs torch):

    A. DEFAULTS. A CohortConfig with no ptrain_* field reads exactly the
       pre-2026-10-01 format: raster / "ptrain" / ^ptrain_(\\d+)\\.mat$,
       and no exclusion.
    B. VALIDATION. validate_ptrain_fields and CohortConfig refuse a format
       outside PTRAIN_FORMATS, a non-identifier variable name, a pattern with
       no or two capture groups, one that does not compile, one carrying a
       character that breaks the EXTRA_FLAGS shell path (whitespace, a double
       quote, * ? [), and exclude_wells given as a string, as paths, or with
       a duplicate.
    C. A GIULIA-LIKE BLOCK round-trips: sparse_peaks / peak_train / the MCS
       pattern / grid_width 10 / index_base 0 / one excluded well.
    D. find_wells(exclude=...) drops exactly the named well under the root
       that holds it, keeps the sorted order, and leaves the other roots and
       the no-exclusion call untouched; a missing root still returns None.

Run
---
    cd Main
    PYTHONPATH="$(pwd)" python3 Smoke_Tests/smoke_test_cohort_fields.py

Exit status 0 only if every check passes. Pure ASCII, LF only.
"""

import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.dirname(_HERE)
if _MAIN not in sys.path:
    sys.path.insert(0, _MAIN)

import cohort as C                                               # noqa: E402

GIULIA_PATTERN = r"^ptrain_\d+_DIV\d+_\w+_nbasal_\d{4}_(\d{3})\.mat$"


def _fail(msg):
    raise AssertionError(msg)


def _base():
    return dict(class_roots={"0": ["/a"], "1": ["/b"]}, class_names=["x", "y"],
                extract_root="/e", n_subsets=9, electrodes_per_subset=1,
                fs_raw=10000.0, grid_width=10, index_base=0, mfr_threshold=0.1,
                w_size=0.01, gaussian_window=0.02)


def check_A_defaults():
    c = C.CohortConfig(**_base())
    if (c.ptrain_format, c.ptrain_varname, c.ptrain_name_pattern, list(c.exclude_wells)) != (
            "raster", "ptrain", r"^ptrain_(\d+)\.mat$", []):
        _fail("A: defaults changed: %r" % ((c.ptrain_format, c.ptrain_varname,
                                             c.ptrain_name_pattern, c.exclude_wells),))
    if (C.DEFAULT_PTRAIN_FORMAT, C.DEFAULT_PTRAIN_VARNAME, C.DEFAULT_PTRAIN_NAME_PATTERN) != (
            "raster", "ptrain", r"^ptrain_(\d+)\.mat$"):
        _fail("A: module defaults changed")
    if C.PTRAIN_FORMATS != ("raster", "sparse_peaks"):
        _fail("A: PTRAIN_FORMATS is %r" % (C.PTRAIN_FORMATS,))
    print("  [A] PASS  defaults are the pre-2026-10-01 format, no exclusion")


def check_B_validation():
    bad = [
        ("format", dict(ptrain_format="dense")),
        ("varname not an identifier", dict(ptrain_varname="1peak")),
        ("varname with a dash", dict(ptrain_varname="peak-train")),
        ("pattern without a group", dict(ptrain_name_pattern=r"^ptrain_\d+\.mat$")),
        ("pattern with two groups", dict(ptrain_name_pattern=r"^ptrain_(\d+)_(\d+)\.mat$")),
        ("pattern that does not compile", dict(ptrain_name_pattern=r"^ptrain_(\d+\.mat$")),
        ("pattern with whitespace", dict(ptrain_name_pattern=r"^ptrain_ (\d+)\.mat$")),
        ("pattern with a double quote", dict(ptrain_name_pattern=r'^ptrain_"(\d+)\.mat$')),
        ("pattern with a glob *", dict(ptrain_name_pattern=r"^ptrain_.*(\d+)\.mat$")),
        ("pattern with a glob ?", dict(ptrain_name_pattern=r"^ptrain_x?(\d+)\.mat$")),
        ("pattern with a glob [", dict(ptrain_name_pattern=r"^ptrain_[0-9]+_(\d+)\.mat$")),
        ("empty pattern", dict(ptrain_name_pattern="")),
        ("exclude_wells a string", dict(exclude_wells="ptrain_x")),
        ("exclude_wells a path", dict(exclude_wells=["a/ptrain_x"])),
        ("exclude_wells an empty name", dict(exclude_wells=[" "])),
        ("exclude_wells a duplicate", dict(exclude_wells=["ptrain_x", "ptrain_x"])),
    ]
    for label, kw in bad:
        try:
            C.CohortConfig(**dict(_base(), **kw))
        except ValueError:
            continue
        _fail("B: CohortConfig accepted %s" % label)
    try:
        C.validate_ptrain_fields("sparse_peaks", "peak_train", GIULIA_PATTERN)
    except ValueError as exc:
        _fail("B: a valid triple was refused: %s" % exc)
    print("  [B] PASS  %d bad values refused, the valid triple accepted" % len(bad))


def check_C_giulia_block():
    c = C.CohortConfig(**dict(_base(), ptrain_format="sparse_peaks",
                              ptrain_varname="peak_train",
                              ptrain_name_pattern=GIULIA_PATTERN,
                              exclude_wells=["ptrain_39485_DIV35_50N50A_nbasal_0001"]))
    if (c.ptrain_format, c.ptrain_varname, c.ptrain_name_pattern) != (
            "sparse_peaks", "peak_train", GIULIA_PATTERN):
        _fail("C: the Giulia block did not round-trip")
    if c.exclude_wells != ["ptrain_39485_DIV35_50N50A_nbasal_0001"]:
        _fail("C: exclude_wells did not round-trip")
    if abs(c.fs_ifr() - 100.0) > 1e-9 or abs(c.sigma_bins() - 2.0) > 1e-9:
        _fail("C: fs_ifr / sigma_bins changed")
    print("  [C] PASS  the Giulia-like block round-trips (fs_ifr 100 Hz, sigma 2 bins)")


def check_D_find_wells():
    base = tempfile.mkdtemp(prefix="cohort_fields_D_")
    try:
        r1, r2 = os.path.join(base, "AraC", "50N50A"), os.path.join(base, "AraC", "70N30A")
        for r, ids in ((r1, (38931, 39477, 39485)), (r2, (38927, 42627))):
            for i in ids:
                os.makedirs(os.path.join(r, "ptrain_%d_DIV35_x_nbasal_0001" % i))
            os.makedirs(os.path.join(r, "not_a_well"))
            open(os.path.join(r, "ptrain_file_not_dir"), "w").write("")
        all1 = C.find_wells(r1, "ptrain_*")
        if all1 != ["ptrain_38931_DIV35_x_nbasal_0001", "ptrain_39477_DIV35_x_nbasal_0001",
                    "ptrain_39485_DIV35_x_nbasal_0001"]:
            _fail("D: find_wells without exclusion: %r" % (all1,))
        ex = C.find_wells(r1, "ptrain_*", exclude=["ptrain_39485_DIV35_x_nbasal_0001"])
        if ex != all1[:2]:
            _fail("D: exclusion did not drop exactly the named well: %r" % (ex,))
        ex2 = C.find_wells(r2, "ptrain_*", exclude=["ptrain_39485_DIV35_x_nbasal_0001"])
        if ex2 != C.find_wells(r2, "ptrain_*"):
            _fail("D: a name absent under a root changed that root's list")
        if C.find_wells(os.path.join(base, "missing"), "ptrain_*", exclude=["x"]) is not None:
            _fail("D: a missing root must still return None")
        if C.find_wells(r1, "ptrain_*", exclude=()) != all1:
            _fail("D: an empty exclusion must change nothing")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("  [D] PASS  find_wells(exclude=...) drops the named well only; None for a missing root")


def main():
    print("cohort fields smoke test (2026-10-01: ptrain_* fields, exclude_wells)")
    checks = [check_A_defaults, check_B_validation, check_C_giulia_block, check_D_find_wells]
    for fn in checks:
        fn()
    print("ALL CHECKS PASSED (%d)" % len(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
