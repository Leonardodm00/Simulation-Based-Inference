"""Block 40: `python3 -m sbitool <command> ...` (specs/SPEC.md block 40).

CONTRACT (wave A): COMMANDS is fixed (command -> module:function of the
block that owns it, each a main(argv) -> exit code); the body is the
implementer's. Exit 0 with the command's pass line, 2 on the arguments (no
or an unknown command: the commands are listed), 3 on a refusal.

Pure ASCII, LF only.
"""

import sys

EXIT_ARGS, EXIT_REFUSE = 2, 3
COMMANDS = {
    "export": "sbitool.export:main",                 # block 41
    "bundle-check": "sbitool.bundle:main",           # block 41
    "card-template": "sbitool.card:main_template",   # block 42
    "card-check": "sbitool.card:main_check",         # block 42
    "windows": "sbitool.windows:main",               # block 44
    "infer": "sbitool.infer:main",                   # block 45
    "device-figure": "sbitool.plotting.device:main",  # block 46
}


def main(argv=None):
    """Dispatches to the command's main; a command whose module is not
    written yet exits 3 saying so."""
    raise NotImplementedError("block 40")


if __name__ == "__main__":
    sys.exit(main())
