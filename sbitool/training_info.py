"""Block 48: the model's training, collected for a bundle (specs/SPEC.md
block 48; D-184 (2)).

CONTRACT (wave A of S6): the name and the signature below are fixed by
specs/SPEC.md; the body is block 48's.

Pure ASCII, LF only.
"""

MEMBER_KEYS = ("seed", "epoch", "npe", "val_npe", "dsn", "rep")
MEMBER_LISTS = ("epoch", "npe", "val_npe", "dsn", "rep")
SEARCH_KEYS = ("campaign", "n_trials", "objective", "trials", "chosen")
TRIAL_KEYS = ("trial_id", "status", "nll", "config")
MEMBER_FILE = "training/member_%d.json"
SEARCH_FILE = "training/search.json"


def collect(P, view, e, records, log=print):
    """(members, search) for ensemble `e` of the project's view.

    P        the pipeline's modules (sbitool.export._pipeline())
    records  the members' record dicts, in member order
    members  a list of M dicts with exactly MEMBER_KEYS: `seed` the member's
             and the five lists of MEMBER_LISTS, of one length >= 1, from the
             record's `history` (a NaN becomes None)
    search   a dict with exactly SEARCH_KEYS (each trial exactly TRIAL_KEYS,
             sorted by trial_id; `config` restricted to JSON scalars whose
             key does not start with "_"), or None when the search's folder
             or its trials are absent (one log line; never a refusal)"""
    raise NotImplementedError("block 48: training_info.collect")
