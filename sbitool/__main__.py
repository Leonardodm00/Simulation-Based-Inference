"""Block 40: `python3 -m sbitool <command> ...` (specs/SPEC.md block 40).

CONTRACT (wave A): COMMANDS is fixed (command -> module:function of the
block that owns it, each a main(argv) -> exit code); the body is the
implementer's. Exit 0 with the command's pass line, 2 on the arguments (no
or an unknown command: the commands are listed), 3 on a refusal.

Pure ASCII, LF only.
"""

import importlib
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
    "verify": "sbitool.verify:main",                 # block 47
}


def _usage(out):
    out.write("usage: python3 -m sbitool <command> [arguments]\n\nthe commands:\n")
    for name, target in COMMANDS.items():
        out.write("  %-14s %s\n" % (name, target))


def main(argv=None):
    """Dispatches to the command's main; a command whose module is not
    written yet exits 3 saying so."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        _usage(sys.stdout)
        return 0
    if not argv or argv[0] not in COMMANDS:
        if argv:
            sys.stderr.write("sbitool: unknown command %r\n" % argv[0])
        else:
            sys.stderr.write("sbitool: no command given\n")
        _usage(sys.stderr)
        return EXIT_ARGS
    command, rest = argv[0], argv[1:]
    modname, func = COMMANDS[command].split(":")
    try:
        module = importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        # only the command's own module (or its package) being absent is "not
        # written yet"; a missing dependency of a written module is an error
        if exc.name is None or not (modname == exc.name or modname.startswith(exc.name + ".")):
            raise
        sys.stderr.write("sbitool: command %r is not available yet: its module %s is not "
                         "written\n" % (command, modname))
        return EXIT_REFUSE
    fn = getattr(module, func, None)
    if fn is None:
        sys.stderr.write("sbitool: command %r is not available yet: %s has no %s\n"
                         % (command, modname, func))
        return EXIT_REFUSE
    return fn(rest)


if __name__ == "__main__":
    sys.exit(main())
