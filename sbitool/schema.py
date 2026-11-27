"""Block 40: the schemas of a model bundle's manifest, of its extraction
block and of its card's header, and their validators (specs/SPEC.md sec.
2.7 and block 40; D-173..D-176).

CONTRACT (wave A): the constants below are fixed by specs/SPEC.md; the
validators' bodies are the implementer's. Standard library only.

Pure ASCII, LF only.
"""

BUNDLE_SCHEMA = "sbitool_bundle/1"

# manifest.json: exactly these top-level keys (sec. 2.7, "The model bundle")
MANIFEST_KEYS = (
    "schema", "name", "created", "card_complete", "arm", "M", "p", "d_z", "W", "members",
    "architecture", "parameters", "extraction", "gate", "sample", "provenance", "software",
    "files")
MEMBER_KEYS = ("file", "sha256", "seed", "state_sha256", "split_seed", "split_hash")
ARCHITECTURE_KEYS = (
    "backbone", "embedding_size", "depth_exponent", "width_multiplier", "block_family",
    "head_fusion", "dropout", "head_pool_ops", "hidden_features", "num_transforms", "num_bins",
    "hidden_layers", "z_score_theta", "z_score_x", "backbone_repr")
BACKBONES = ("dsn", "fixed-stats", "fallback")
PARAMETER_KEYS = ("names", "units", "bounds")
# the parameters that turn a recording into windows ("From a recording to windows")
EXTRACTION_KEYS = (
    "electrodes_per_subset", "n_subsets", "mfr_threshold_hz", "w_size_s", "gaussian_window_s",
    "grid_width", "index_base", "pitch_um", "fs_raw_hz", "detection")
# extraction["detection"]: None, or exactly these (D-176; SpikeInterface's detect_peaks)
DETECTION_KEYS = ("method", "peak_sign", "detect_threshold", "exclude_sweep_ms")
DETECTION_DEFAULT = {"method": "by_channel", "peak_sign": "neg", "detect_threshold": 5.0,
                     "exclude_sweep_ms": 1.0}
GATE_KEYS = ("factors", "alpha", "n_ref", "n_rank", "n_null", "k_gate", "size_mean", "size_lo",
             "size_hi", "level")
SAMPLE_KEYS = ("n_post", "seed")
PROVENANCE_KEYS = ("project", "ensemble", "search", "campaign", "best_fit_complete_sha256",
                   "sample_json_sha256", "commit", "sources")
SOFTWARE_KEYS = ("python", "torch", "sbi", "zuko", "numpy", "safetensors")

# the members of the zip besides members/member_<k>.safetensors
MEMBER_FILE = "members/member_%d.safetensors"
FIXED_MEMBERS = ("manifest.json", "CARD.md", "evaluation/sim.json", "evaluation/gate.safetensors")
GATE_ARRAYS = ("z_ref", "z_rank", "z_null")

# CARD.md's header: exactly these top-level keys, in this order ("The card")
CARD_KEYS = (
    "title", "scientific_question", "simulator", "training", "extraction", "references",
    "authors", "licence", "cite", "data", "software", "limitations")
CARD_SIMULATOR_KEYS = ("model_type", "reference", "fixed_parameters", "searched_parameters")
CARD_TRAINING_KEYS = ("arm", "members", "n_networks", "n_windows", "split_fracs",
                      "hyperparameters", "hyperparameter_search")
CARD_AUTHOR_KEYS = ("name", "contact")
CARD_DATA_KEYS = ("simulated_bank", "recordings")
CARD_BANK_KEYS = ("location", "public", "size_bytes", "sha256")
CARD_RECORDINGS_KEYS = ("location", "public")
TODO = "TODO"                      # the literal an unfilled card field holds
NOT_APPLICABLE = "not applicable"  # card["extraction"] of a bundle whose extraction is null


def validate_extraction(data):
    """Every problem of an extraction block (a dict with exactly
    EXTRACTION_KEYS; detection None or exactly DETECTION_KEYS), as a list of
    strings naming the key; [] when valid. Never raises on a JSON value."""
    raise NotImplementedError("block 40")


def validate_manifest(data):
    """Every problem of a manifest (keys, types, ranges, the cross-conditions
    of sec. 2.7), as a list of strings naming the key; [] when valid. Never
    raises on a JSON value."""
    raise NotImplementedError("block 40")


def validate_card_header(data, manifest=None, complete=False):
    """Every problem of a card's header (a dict: exactly CARD_KEYS and the
    nested key sets). With `complete` true an empty value or a TODO anywhere
    is a problem too; with `manifest` given, the agreement sec. 2.7 states
    (arm, members, searched parameters, extraction, software) is checked.
    Returns a list of strings; [] when valid. Never raises on a YAML value."""
    raise NotImplementedError("block 40")


def card_todo_fields(data):
    """The dotted names of the header's fields that are empty or hold TODO
    (training.hyperparameter_search, authors[0].contact, ...), sorted."""
    raise NotImplementedError("block 40")
