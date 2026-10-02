"""
cohort.py -- the real-MEA cohort declaration and the extraction-output layout,
with NO torch dependency.

Migration Stage D, 2026-09-21. CohortConfig used to live in config.py, whose
first imports are backbone.py and augmentation.py, so reading six extraction
numbers out of a JSON config pulled in torch. list_extraction_jobs.py in
Sbi-extractor runs in the extractor's numpy environment and must not need
that. Everything here was MOVED VERBATIM:

    CohortConfig                   from config.py lines 1251-1444
    SUBREGION_PREFIX, MULTICHANNEL_NAME,
    root_name_for, find_wells,
    classify_output_dir, expand    from make_mea_specs.py lines 76-146

config.py re-exports CohortConfig so ExperimentConfig.cohort is unchanged in
type and behaviour; make_mea_specs.py re-imports the helpers so MMS.expand
etc. keep working. There is still exactly ONE definition of each.

Imports: dataclasses, typing, os, fnmatch. Nothing else. Keep it that way.
Pure ASCII, LF only (hpc-python-compat).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

__all__ = [
    "CohortConfig",
    "SUBREGION_PREFIX",
    "MULTICHANNEL_NAME",
    "PTRAIN_FORMATS",
    "DEFAULT_PTRAIN_FORMAT",
    "DEFAULT_PTRAIN_VARNAME",
    "DEFAULT_PTRAIN_NAME_PATTERN",
    "validate_ptrain_fields",
    "root_name_for",
    "find_wells",
    "classify_output_dir",
    "expand",
]

# --------------------------------------------------------------------------- #
# how a cohort's per-electrode files are read (2026-10-01, Giulia cohort)
# --------------------------------------------------------------------------- #
# ONE definition, imported by Sbi-extractor (cohort_config.py, the extractor
# library) so the config, the flags and the loader cannot disagree on the
# vocabulary. The defaults are the pre-2026-10-01 behaviour exactly.
PTRAIN_FORMATS = ("raster", "sparse_peaks")
DEFAULT_PTRAIN_FORMAT = "raster"
DEFAULT_PTRAIN_VARNAME = "ptrain"
DEFAULT_PTRAIN_NAME_PATTERN = r"^ptrain_(\d+)\.mat$"

# Characters a flag value must not contain. The three values are written
# into extraction_flags.sh as EXTRA_FLAGS="..." and word-split unquoted by
# the array job: whitespace would split a value, a double quote would end
# the string, and * ? [ would be subject to pathname expansion.
_FLAG_UNSAFE = " \t\r\n\"*?["


def validate_ptrain_fields(ptrain_format, ptrain_varname, ptrain_name_pattern):
    """Raise ValueError unless the three source-format fields are usable.

    - ptrain_format in PTRAIN_FORMATS;
    - ptrain_varname a MATLAB identifier ([A-Za-z][A-Za-z0-9_]*);
    - ptrain_name_pattern a regex that compiles and has exactly ONE capture
      group (the electrode index), free of the characters that would break
      it as an unquoted shell word (see _FLAG_UNSAFE).
    Returns None. Shared by CohortConfig.__post_init__ and the extractor's
    CLI, so a bad value is refused at config time and again at load time.
    """
    import re
    if ptrain_format not in PTRAIN_FORMATS:
        raise ValueError("ptrain_format must be one of %r, got %r"
                         % (PTRAIN_FORMATS, ptrain_format))
    if (not isinstance(ptrain_varname, str)
            or re.match(r"^[A-Za-z][A-Za-z0-9_]*$", ptrain_varname) is None):
        raise ValueError("ptrain_varname must be a MATLAB identifier "
                         "([A-Za-z][A-Za-z0-9_]*), got %r" % (ptrain_varname,))
    if not isinstance(ptrain_name_pattern, str) or not ptrain_name_pattern:
        raise ValueError("ptrain_name_pattern must be a non-empty regex, got %r"
                         % (ptrain_name_pattern,))
    bad = sorted(set(c for c in ptrain_name_pattern if c in _FLAG_UNSAFE))
    if bad:
        raise ValueError(
            "ptrain_name_pattern %r contains %r, which cannot travel as an "
            "unquoted shell word in EXTRA_FLAGS (no whitespace, no double "
            "quote, no * ? [); write the class another way, e.g. \\d or \\w"
            % (ptrain_name_pattern, bad))
    try:
        rx = re.compile(ptrain_name_pattern)
    except re.error as exc:
        raise ValueError("ptrain_name_pattern %r does not compile: %s"
                         % (ptrain_name_pattern, exc))
    if rx.groups != 1:
        raise ValueError(
            "ptrain_name_pattern %r has %d capture group(s); exactly one is "
            "required, and it must capture the electrode's integer index"
            % (ptrain_name_pattern, rx.groups))


@dataclass
class CohortConfig:
    """Declares WHERE the real MEA cohort lives on disk, per class.

    This block is DECLARATIVE. `run_optimization.py` never reads it: the search
    consumes `data.npz_specs`, a flat list of .npz records. What this block does
    is let ONE file describe both the cohort and the search, so the well list
    used to generate the specs cannot drift from the experiment that consumes
    them. `make_mea_specs.py` reads it and writes the specs file.

    Directory model (three levels, all real on disk):

        <root>/                      one ENTRY of class_roots[c]; a plate/batch
            ptrain_A1/               a WELL == a recording == a CULTURE
                ptrain_10.mat        one electrode's spike raster
                ...
            ptrain_A2/  ...  ptrain_B3/

    and, produced by run_channel_subset_extraction.py, the extraction outputs:

        <extract_root>/<extract_layout>/
            trace_subregion_00.npz   mode per_region_single: N_SUB traces,
            ...                      one CULTURE, K3 groups them
            trace_subregion_08.npz
        or
            traces.npz               mode multichannel: ONE (N_SUB, K) trace

    Both output shapes are recognised. Which one is present decides how many
    trace records a well contributes, so it is reported rather than assumed.

    Fields
    ------
    class_roots : {class index as a STRING : [root paths]}. JSON object keys are
        always strings, hence "0" / "1" rather than 0 / 1. Several roots per
        class is the expected case (one per plate/batch). Class indices must be
        contiguous from 0.
    class_names : optional human labels, parallel to the sorted class indices;
        used for directory naming and log lines, never for the label itself.
    well_glob   : pattern matched against the immediate children of each root.
    extract_root: where run_channel_subset_extraction.py wrote its outputs.
    extract_layout : path template BELOW extract_root, expanded per well with
        {class_index} {class_name} {root_name} {well}. Change this rather than
        editing code if the extraction outputs were laid out differently.
    culture_template : how a culture id is built. It MUST be unique per well
        across the whole cohort -- {root_name}__{well} is unique whenever two
        roots do not share a basename, which is why root_name is included.

    The extraction parameters are recorded so the specs file, the extraction
    array job, and this config cannot disagree about how the traces were made.
    They are metadata here; the extractor takes them as CLI flags.
    """

    class_roots: Dict[str, List[str]] = field(default_factory=dict)
    class_names: List[str] = field(default_factory=list)
    well_glob: str = "ptrain_*"

    extract_root: str = ""
    extract_layout: str = "{class_name}/{root_name}/{well}"
    culture_template: str = "{root_name}__{well}"

    # extraction parameters, mirroring run_channel_subset_extraction.py defaults
    n_subsets: int = 9
    electrodes_per_subset: int = 9
    fs_raw: float = 10110.09
    grid_width: int = 48
    index_base: int = 0
    mfr_threshold: float = 0.1
    w_size: float = 0.02
    gaussian_window: float = 0.04

    # [2026-10-01, Giulia cohort] HOW the per-electrode files are named and
    # stored. The defaults reproduce the one format the extractor read until
    # now (the DUP15HD 3Brain export): ptrain_<k>.mat holding a dense binary
    # raster under the variable "ptrain". The Giulia recordings are
    # ptrain_<well>_DIV35_<cond>_nbasal_0001_<rc>.mat and (per their sizes)
    # not dense rasters, so a cohort DECLARES its format here; the extractor
    # never guesses it from the file. The three values travel to the array
    # job as flags (cohort_config.build_extra_flags) and are recorded in every
    # fragment, so a manifest can say how its spikes were read.
    #   ptrain_format        "raster"       dense (n_samples, 1) raster in
    #                                       {0, 1}; a 1 is a spike
    #                        "sparse_peaks" scipy.sparse (n_samples, 1) (or
    #                                       (1, n_samples)); every stored
    #                                       nonzero is a spike, its value
    #                                       (an amplitude, say) is ignored
    #   ptrain_varname       the MATLAB variable holding the train
    #   ptrain_name_pattern  a regex with exactly ONE capture group, matched
    #                        against the file basename; the group is the
    #                        electrode's integer index k
    #   exclude_wells        well folder names (immediate children of a root
    #                        matching well_glob) left out of the cohort, e.g.
    #                        wells with fewer than n_subsets x
    #                        electrodes_per_subset electrodes at or above
    #                        mfr_threshold, which the extractor would refuse
    #                        and the cohort manifest would then refuse with
    #                        them; a name that matches no well is an error at
    #                        listing time, not a silent no-op
    ptrain_format: str = "raster"
    ptrain_varname: str = "ptrain"
    ptrain_name_pattern: str = r"^ptrain_(\d+)\.mat$"
    exclude_wells: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.class_roots:
            return                      # empty cohort: synthetic/latent configs

        keys = []
        for k in self.class_roots.keys():
            try:
                keys.append(int(k))
            except (TypeError, ValueError):
                raise ValueError(
                    "CohortConfig.class_roots: key %r is not an integer class "
                    "index. JSON object keys are strings, so write \"0\" / "
                    "\"1\"." % (k,))
        keys = sorted(keys)
        if keys != list(range(len(keys))):
            raise ValueError(
                "CohortConfig.class_roots: class indices must be contiguous "
                "from 0; got %r. The pipeline builds C = number of distinct "
                "conditions and indexes classes 0..C-1." % (keys,))

        for k, roots in self.class_roots.items():
            if isinstance(roots, str) or not isinstance(roots, (list, tuple)):
                raise ValueError(
                    "CohortConfig.class_roots[%r] must be a LIST of root paths, "
                    "got %r. A single root still goes in a list." % (k, roots))
            if len(roots) == 0:
                raise ValueError(
                    "CohortConfig.class_roots[%r] is empty; every class needs "
                    "at least one root folder." % (k,))
            seen = set()
            for r in roots:
                rs = str(r)
                if rs in seen:
                    raise ValueError(
                        "CohortConfig.class_roots[%r]: duplicate root %r. The "
                        "same folder listed twice would double-count its wells."
                        % (k, rs))
                seen.add(rs)

        # a root may not appear under two classes: that is a phenotype conflict
        owner = {}
        for k, roots in self.class_roots.items():
            for r in roots:
                rs = str(r)
                if rs in owner and owner[rs] != str(k):
                    raise ValueError(
                        "CohortConfig: root %r is listed under BOTH class %s "
                        "and class %s. A plate has one phenotype."
                        % (rs, owner[rs], k))
                owner[rs] = str(k)

        if self.class_names and len(self.class_names) != len(keys):
            raise ValueError(
                "CohortConfig.class_names has %d entry/entries but there are "
                "%d class(es); they must be parallel to the sorted class "
                "indices %r." % (len(self.class_names), len(keys), keys))

        if int(self.n_subsets) < 1:
            raise ValueError("CohortConfig.n_subsets must be >= 1")
        if int(self.electrodes_per_subset) < 1:
            raise ValueError("CohortConfig.electrodes_per_subset must be >= 1")
        if float(self.fs_raw) <= 0.0:
            raise ValueError("CohortConfig.fs_raw must be > 0")
        if int(self.grid_width) < 1:
            raise ValueError("CohortConfig.grid_width must be >= 1")
        if int(self.index_base) not in (0, 1):
            raise ValueError(
                "CohortConfig.index_base must be 0 or 1 (see REAL_DATA_FINDINGS "
                "O1: both fit the 48x48 grid, so this is an assumption)")
        if float(self.mfr_threshold) < 0.0:
            raise ValueError("CohortConfig.mfr_threshold must be >= 0")
        if float(self.w_size) <= 0.0:
            raise ValueError("CohortConfig.w_size must be > 0")
        if float(self.gaussian_window) < 0.0:
            raise ValueError("CohortConfig.gaussian_window must be >= 0 "
                             "(0 disables smoothing)")
        # sigma is in SECONDS but the filter runs on BINS, so the effective
        # smoothing is sigma/w_size bins. Below ~1 bin the Gaussian is
        # narrower than the sampling grid and does essentially nothing --
        # easy to cause accidentally by shrinking w_size and leaving sigma
        # alone, or vice versa, since the two are set independently.
        if float(self.gaussian_window) > 0.0:
            sigma_bins = float(self.gaussian_window) / float(self.w_size)
            if sigma_bins < 1.0:
                warnings.warn(
                    "CohortConfig: gaussian_window=%.4g s is only %.2f bin(s) "
                    "at w_size=%.4g s, so smoothing is close to a no-op. "
                    "sigma is specified in seconds and applied over bins; if "
                    "you changed one, check the other."
                    % (float(self.gaussian_window), sigma_bins,
                       float(self.w_size)), RuntimeWarning)

        validate_ptrain_fields(self.ptrain_format, self.ptrain_varname,
                               self.ptrain_name_pattern)
        if isinstance(self.exclude_wells, str) or not isinstance(
                self.exclude_wells, (list, tuple)):
            raise ValueError(
                "CohortConfig.exclude_wells must be a LIST of well folder "
                "names, got %r" % (self.exclude_wells,))
        seen = set()
        for w in self.exclude_wells:
            if not isinstance(w, str) or not w.strip():
                raise ValueError(
                    "CohortConfig.exclude_wells: every entry must be a "
                    "non-empty well folder name, got %r" % (w,))
            if "/" in w or os.sep in w:
                raise ValueError(
                    "CohortConfig.exclude_wells: %r is a path; give the well "
                    "folder NAME (an immediate child of a root)" % (w,))
            if w in seen:
                raise ValueError(
                    "CohortConfig.exclude_wells: duplicate entry %r" % (w,))
            seen.add(w)

    # ----- convenience, used by make_mea_specs.py -----
    def n_classes(self):
        return len(self.class_roots)

    def name_of_class(self, c):
        """Human label for class index c; falls back to 'class<c>'."""
        if self.class_names and 0 <= int(c) < len(self.class_names):
            return str(self.class_names[int(c)])
        return "class%d" % int(c)

    def fs_ifr(self):
        """IFR rate implied by the smoothing window: f_s^IFR = 1 / w_size."""
        return 1.0 / float(self.w_size)

    def sigma_bins(self):
        """Effective Gaussian smoothing width in BINS: sigma / w_size.

        The two IFR knobs are easy to confuse, so stating them apart:

          w_size          the DOWNSAMPLING BIN, Delta_t [s]. Sets the output
                          rate, f_s^IFR = 1 / w_size, and hence K and the
                          window length in samples (T = window_s * f_s^IFR).
          gaussian_window sigma of the gaussian_filter1d applied AFTER
                          binning [s]. Does not change the rate, only how
                          much the binned counts are smoothed.

        They interact because sigma is given in seconds but the filter runs
        over bins: halving w_size doubles the smoothing measured in bins
        while leaving sigma nominally unchanged. This is the number to look
        at when deciding whether a change to either knob did what you meant.
        """
        return float(self.gaussian_window) / float(self.w_size)


# --------------------------------------------------------------------------- #
# extraction-output layout and discovery (moved from make_mea_specs.py)
# --------------------------------------------------------------------------- #
SUBREGION_PREFIX = "trace_subregion_"
MULTICHANNEL_NAME = "traces.npz"


def root_name_for(root):
    """The {root_name} value substituted into extract_layout/culture_template.

    THE single place this is computed -- both build_records() below and
    list_extraction_jobs.py's manifest builder call this, so the two can
    never disagree about a well's culture id or output path.

    Two path COMPONENTS (parent + leaf), not one. A bare basename collides
    the moment two batches reuse a leaf name -- e.g. a "SubBatch1" folder
    that exists under both a control batch and a pathological batch is a
    real, observed layout, not a hypothetical: DATA_C/Batch4/SubBatch1 and
    DATA_P/Batch3/SubBatch1 are two different cultures that a bare basename
    would fold into one, silently pointing both wells' extraction at the
    same output directory.

    This is a best-effort disambiguator, not a uniqueness proof: two roots
    that ALSO share their parent's name would still collide. That is exactly
    why the collision check at the call site (build_records here;
    list_extraction_jobs.py's own check) is a hard abort rather than a
    warning -- silent data loss is not an acceptable failure mode for either
    caller, so an unresolved collision must stop the run, not print past it.
    """
    root = os.path.normpath(str(root))
    leaf = os.path.basename(root)
    parent = os.path.basename(os.path.dirname(root))
    return "%s_%s" % (parent, leaf) if parent else leaf


def find_wells(root, well_glob, exclude=()):
    """Immediate child directories of `root` matching `well_glob`, sorted.

    Deliberately NOT recursive: the extractor reads one leaf well folder at a
    time, and recursing would sweep up intermediate Batch/ folders as if they
    were wells (see REAL_DATA_FINDINGS, "Command").

    `exclude` (2026-10-01): well folder NAMES to leave out, normally
    CohortConfig.exclude_wells. A name in `exclude` that is present under
    this root is skipped here; whether every excluded name was found
    somewhere is the caller's check (list_extraction_jobs.py aborts on an
    unmatched name, make_mea_specs.py reports it), so a typo cannot exclude
    nothing in silence.
    """
    import fnmatch
    if not os.path.isdir(root):
        return None                                  # signals "root missing"
    excl = set(str(e) for e in (exclude or ()))
    out = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and fnmatch.fnmatch(entry, well_glob):
            if entry in excl:
                continue
            out.append(entry)
    return out


def classify_output_dir(out_dir):
    """('per_region_single', [paths]) | ('multichannel', [path]) | (None, [])."""
    if not os.path.isdir(out_dir):
        return None, []
    names = sorted(os.listdir(out_dir))
    subs = [n for n in names
            if n.startswith(SUBREGION_PREFIX) and n.endswith(".npz")]
    if subs:
        return "per_region_single", [os.path.join(out_dir, n) for n in subs]
    if MULTICHANNEL_NAME in names:
        return "multichannel", [os.path.join(out_dir, MULTICHANNEL_NAME)]
    return None, []


def expand(template, class_index, class_name, root_name, well):
    return template.format(class_index=class_index, class_name=class_name,
                           root_name=root_name, well=well)
