"""Block 42: the card of a model bundle (specs/SPEC.md sec. 2.7, "The
card", and block 42; D-158 (4), D-173, D-175). PyYAML's safe_load and
safe_dump only.

CONTRACT (wave B): the signatures below are fixed.

Pure ASCII, LF only.
"""

import sys

EXIT_ARGS, EXIT_REFUSE = 2, 3


class CardError(ValueError):
    """The text is not a card: no header between two --- lines, YAML that
    does not parse or would build Python objects, a header that is not a
    mapping."""


def parse(text):
    """(header, body): the YAML header as a dict and the Markdown after it.
    CardError on a malformed card."""
    raise NotImplementedError("block 42")


def dump(header, body):
    """The card's text: the header by safe_dump in schema.CARD_KEYS' order
    between two --- lines, then the body. parse(dump(h, b)) == (h, b)."""
    raise NotImplementedError("block 42")


def template(manifest, records, project=None):
    """A CARD.md for the bundle of `manifest`: every field the manifest and
    the members' records hold pre-filled, schema.TODO elsewhere, and a body
    of headed sections that say what to write. `records`: the members'
    record dicts (the runner's <arm>_seed<k>.json), in member order."""
    raise NotImplementedError("block 42")


def check(text, manifest=None, complete=False):
    """Every problem of the card's text (schema.validate_card_header on its
    header); CardError on a malformed card."""
    raise NotImplementedError("block 42")


def main_template(argv=None):
    """`python3 -m sbitool card-template --bundle FILE --out CARD.md`: the
    template rebuilt from a bundle's manifest (the pre-filled fields the
    manifest holds; the training fields from the bundle's own card when it
    has them)."""
    raise NotImplementedError("block 42")


def main_check(argv=None):
    """`python3 -m sbitool card-check CARD.md [--bundle FILE]`: 0 with CARD
    OK: complete, or CARD OK: draft, <k> field(s) to fill: <names>;
    EXIT_REFUSE on a malformed card or a key problem."""
    raise NotImplementedError("block 42")


if __name__ == "__main__":
    sys.exit(main_check())
