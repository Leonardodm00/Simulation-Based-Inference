"""Block 41: `python3 -m sbitool export` -- a trained ensemble of a recorded
project, with its evaluation (blocks 35-36), written as a model bundle
(specs/SPEC.md block 41).

CONTRACT (wave B): the signatures below are fixed.

Pure ASCII, LF only.
"""

import sys

EXIT_ARGS, EXIT_REFUSE = 2, 3


class ExportRefused(RuntimeError):
    """The export cannot be made; nothing is written (exit 3)."""


def run(cfg_path, ecfg_path, name, out, extraction=None, card=None, bundle_name=None,
        log=print):
    """Exports ensemble `name` of the project at `cfg_path`, evaluated under
    `ecfg_path`, to the bundle `out`. `extraction`: None or the path of a
    JSON file with exactly schema.EXTRACTION_KEYS; `card`: None (block 42's
    template, a draft) or the path of a CARD.md. Returns the manifest.
    ExportRefused (or ev_plan.RefuseError) before anything is written."""
    raise NotImplementedError("block 41")


def main(argv=None):
    """The command line: 0 with BUNDLE EXPORT OK: ..., EXIT_ARGS,
    EXIT_REFUSE."""
    raise NotImplementedError("block 41")


if __name__ == "__main__":
    sys.exit(main())
