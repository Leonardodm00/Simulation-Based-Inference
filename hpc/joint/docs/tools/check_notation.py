#!/usr/bin/env python3
"""Notation coverage: every math symbol used in a document is declared in E0.

The master table is the "## 1. Notation and symbols" table of
E0_READERS_GUIDE_NOTATION.md: its rows' first cells hold one or more `$...$`
spans, each a bare symbol (`$m_g$`, `$\\hat m_g$`, `$B_{\\rm sim}, B_{\\rm met}$`
declares two). A document passes when every token of every `$...$` or
`$$...$$` span in its prose -- fenced code blocks and inline code excluded --
is one of:

  * a declared symbol, compared in a canonical form (whitespace removed,
    subscript written before superscript, accent braces removed);
  * an instance of a declared symbol whose index-like scripts are
    substituted: `\\chi^2_{26}` for a declared `\\chi^2_d`, `x_g` for `x`
    with `g` declared, `T_{gg'}` for `T` with `g`, `g'` declared, `\\Sigma_0^{-1}`
    for `\\Sigma_0` with the decoration `^{-1}`;
  * an operator, relation, delimiter, number or type token (`\\le`, `\\mid`,
    `\\mathbb{R}`, `2.5`, `10^{-3}`), listed in OPERATORS / TYPES below; the
    scripts of an operator are still checked (`\\sum_{j=1}^{d_\\theta}`).

Anything else is a residual: a symbol the document uses and E0 does not
declare. The checker prints each residual with the line it first appears
on and exits 1 if any document has one.

Usage:
    python3 check_notation.py --master ../E0_READERS_GUIDE_NOTATION.md \\
        --docs ../P0_PARAMETERS_OVERVIEW.md ../E1_THE_PROBLEM.md
    python3 check_notation.py --master ../E0_READERS_GUIDE_NOTATION.md --self --list

Pure ASCII, standard library only, deterministic output.
"""
from __future__ import print_function

import argparse
import re
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

ACCENTS = ("\\hat", "\\bar", "\\tilde", "\\widehat", "\\widetilde",
           "\\overline", "\\vec", "\\dot", "\\ddot", "\\check", "\\breve")

FONTS = ("\\mathrm", "\\mathcal", "\\mathbb", "\\mathbf", "\\mathsf",
         "\\mathit", "\\boldsymbol", "\\operatorname", "\\text", "\\textrm")

# a prefix atom that merges with the atom that follows it: \delta\theta_m,
# \Delta t, \Delta y_g are single symbols
COMPOUND_PREFIX = ("\\delta", "\\Delta")

OPERATORS = set("""
\\le \\ge \\leq \\geq \\ne \\neq \\to \\in \\notin \\subset \\subseteq \\supset
\\supseteq \\times \\mid \\sum \\prod \\int \\log \\ln \\exp \\min \\max \\argmin
\\argmax \\sup \\inf \\lim \\partial \\infty \\cdot \\cdots \\ldots \\dots \\vdots
\\ddots \\top \\pm \\mp \\sqrt \\frac \\left \\right \\big \\Big \\bigl \\bigr
\\Bigl \\Bigr \\| \\oplus \\otimes \\sim \\approx \\equiv \\propto \\nabla \\circ
\\langle \\rangle \\lfloor \\rfloor \\lceil \\rceil \\lVert \\rVert \\quad \\qquad \\, \\; \\: \\!
\\ \\lvert \\rvert \\lbrace \\rbrace \\{ \\} \\setminus \\cup \\cap \\forall \\exists
\\implies \\iff \\Rightarrow \\Leftarrow \\leftarrow \\rightarrow \\mapsto \\wedge
\\vee \\neg \\land \\lor \\ast \\star \\prime \\ll \\gg \\simeq \\cong \\perp
\\parallel \\tag \\colon \\Pr \\det \\dim \\deg \\cos \\sin \\tan \\arccos \\arcsin
\\tanh \\mathrm{diag} \\mathrm{tr} \\mathrm{Var} \\mathrm{Cov} \\mathrm{round}
\\mathrm{clamp} \\mathrm{softmax} \\mathrm{sign} \\mathrm{rank} \\mathrm{det}
\\mathrm{sd} \\mathrm{se} \\mathrm{E} \\mathbb{E} \\mathbb{P} \\mathbb{1} \\mathrm{d}
\\mathrm{i.i.d.} \\mathrm{a.s.} \\mathrm{s.t.} \\mathrm{const} \\mathrm{supp}
\\mathrm{vec} \\mathrm{cos} \\mathrm{logit} \\mathrm{softplus} \\mathrm{mean}
\\mathrm{median} \\mathrm{floor} \\mathrm{argmin} \\mathrm{argmax} \\mathrm{KL}
\\mathrm{DEFF} \\mathrm{MMD} \\mathrm{EI} \\mathrm{NLL} \\mathrm{ETF} \\mathrm{SBC}
\\operatorname{diag} \\operatorname{tr} \\operatorname{Var} \\operatorname{Cov}
\\mathrm{LOG\\_PARAMS} \\mathrm{ns} \\mathrm{topo} \\mathrm{var} \\mathrm{std} \\mathrm{max} \\mathrm{min} \\mathrm{len}
""".split())

TYPES = set("""
\\mathbb{R} \\mathbb{N} \\mathbb{Z} \\mathbb{Q} \\mathbb{C}
""".split())

# superscripts allowed on any declared base
DECORATIONS = {"\\top", "-1", "2", "3", "*", "\\star", "\\prime", "'", "+",
               "-", "\\ast", "T", "1/2", "-1/2", "-2", "\\dagger", "{-1}", "{2}"}

# script arguments that need no declaration
INDEX_FREE = {"*", "\\star", "0", "1", "2", "3", "\\min", "\\max", "\\pm",
              "\\ge0", ">0", "\\geq0", "\\le0", "<0", "\\ne0", "\\neq0",
              "\\gt0", "\\lt0", "+", "-"}

_MACRO = re.compile(r"\\[A-Za-z]+|\\[,;:!|{}]|\\ ")
# Relations that may appear inside a script: _{c \ne c'}, _{c : n_c \ge 2}.
# Only the two-letter macros: canonicalisation strips spaces, so `\ne q'
# and `\neq` are the same string, and splitting on the two-letter form reads
# it as a relation followed by an index (`q`), which is the reading that
# keeps an undeclared index visible. Write \ne, \le, \ge inside scripts.
_RELATION = re.compile(r"\\ne|\\le|\\ge|<|>|:")
_BOUND = re.compile(r"(\\ge|\\le|\\geq|\\leq|\\gt|\\lt|\\ne|\\neq|>|<|=)-?\d+(?:\.\d+)?")
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


# ----------------------------------------------------------------------------
# low-level readers
# ----------------------------------------------------------------------------

def _balanced(s: str, i: int) -> int:
    """Index just past the balanced {...} group that starts at s[i] == '{'."""
    depth = 0
    j = i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(s)


def _read_script_arg(s: str, i: int) -> Tuple[str, int]:
    """The argument of a _ or ^ that starts at s[i]: (text, index after it)."""
    if i >= len(s):
        return "", i
    if s[i] == "{":
        j = _balanced(s, i)
        return s[i:j], j
    m = _MACRO.match(s, i)
    if m:
        j = m.end()
        if m.group(0) in FONTS and j < len(s) and s[j] == "{":
            j = _balanced(s, j)
        return s[i:j], j
    if s[i] == "(":
        j = s.find(")", i)
        j = len(s) if j < 0 else j + 1
        return s[i:j], j
    return s[i], i + 1


def _read_scripts(s: str, k: int) -> Tuple[List[Tuple[str, str]], int]:
    """Read the _x, ^x and ' tails that follow an atom; return (parts, next)."""
    parts: List[Tuple[str, str]] = []
    seen_sub = seen_sup = False
    while k < len(s):
        ch = s[k]
        if ch == "'":
            parts.append(("'", "'"))
            k += 1
        elif ch == "_" and not seen_sub:
            arg, k = _read_script_arg(s, k + 1)
            parts.append(("_", arg))
            seen_sub = True
        elif ch == "^" and not seen_sup:
            arg, k = _read_script_arg(s, k + 1)
            parts.append(("^", arg))
            seen_sup = True
        else:
            break
    return parts, k


def _read_atom(s: str, i: int) -> Tuple[str, int, str]:
    """One atom at s[i]: (kind, next index, text).

    kind: 'atom' (a candidate symbol base), 'op' (operator / type),
    'num', 'accent', or 'skip' (whitespace, punctuation, brackets).
    """
    ch = s[i]
    if ch.isspace():
        return "skip", i + 1, ch
    m = _NUMBER.match(s, i)
    if m:
        return "num", m.end(), m.group(0)
    m = _MACRO.match(s, i)
    if m:
        mac = m.group(0)
        j = m.end()
        if mac in ACCENTS:
            return "accent", j, mac
        if mac in ("\\text", "\\textrm", "\\begin", "\\end", "\\label", "\\tag"):
            # prose inside math, environment delimiters, labels: not symbols
            if j < len(s) and s[j] == "{":
                j = _balanced(s, j)
            return "skip", j, mac
        if mac in FONTS:
            if j < len(s) and s[j] == "{":
                k = _balanced(s, j)
                text = s[i:k]
                kind = "op" if (text in OPERATORS or text in TYPES) else "atom"
                return kind, k, text
            return "skip", j, mac
        if mac in ("\\rm", "\\bf", "\\it", "\\displaystyle", "\\textstyle"):
            return "skip", j, mac
        if mac in OPERATORS:
            return "op", j, mac
        return "atom", j, mac
    if ch.isalpha():
        return "atom", i + 1, ch
    return "skip", i + 1, ch


def _canon_parts(parts: List[Tuple[str, str]]) -> str:
    """Scripts in canonical order: primes, then subscript, then superscript."""
    order = {"'": 0, "_": 1, "^": 2}
    return "".join(("'" if k == "'" else k + a)
                   for k, a in sorted(parts, key=lambda p: order[p[0]]))


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def tokenize(span: str) -> List[str]:
    """The candidate symbol tokens of one math span, in canonical form.

    A token is accents + atom + scripts. Operators and numbers are dropped,
    except an operator that carries scripts (its scripts are checked).
    """
    s = span
    out: List[str] = []
    i = 0
    accent = ""
    while i < len(s):
        kind, j, text = _read_atom(s, i)
        if kind == "accent":
            accent += text
            # \hat{m}_g: the braces around the accented atom are transparent
            if j < len(s) and s[j] == "{":
                k = _balanced(s, j)
                inner = tokenize(s[j + 1:k - 1])
                base = inner[0] if inner else ""
                parts, k = _read_scripts(s, k)
                out.append(_norm(accent + base + _canon_parts(parts)))
                accent = ""
                i = k
            else:
                i = j
            continue
        if kind == "skip":
            if not text.isspace():
                accent = ""
            i = j
            continue
        if kind == "num":
            accent = ""
            if j < len(s) and s[j] == "^":          # 10^{-3}
                _, j = _read_script_arg(s, j + 1)
            i = j
            continue
        parts, k = _read_scripts(s, j)
        tok = accent + text + _canon_parts(parts)
        accent = ""
        if kind == "atom" and text in COMPOUND_PREFIX and not parts:
            # \delta\theta_m, \Delta t, \Delta y_g: merge with the next atom
            k2 = k
            while k2 < len(s) and s[k2].isspace():
                k2 += 1
            if k2 < len(s):
                kind2, j2, text2 = _read_atom(s, k2)
                if kind2 == "atom":
                    parts2, k3 = _read_scripts(s, j2)
                    tok = text + text2 + _canon_parts(parts2)
                    k = k3
        tok = _norm(tok)
        if kind == "op":
            if parts:
                out.append(tok)
        else:
            out.append(tok)
        i = k
    return out


# ----------------------------------------------------------------------------
# documents and the master table
# ----------------------------------------------------------------------------

_FENCE = re.compile(r"^```.*?^```[ \t]*$", re.M | re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
_INLINE = re.compile(r"(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)")


def _blank(m) -> str:
    return re.sub(r"[^\n]", " ", m.group(0))


def strip_code(text: str) -> str:
    """Blank fenced blocks and inline code, keeping every line number."""
    return _INLINE_CODE.sub(_blank, _FENCE.sub(_blank, text))


def math_spans(text: str) -> List[Tuple[int, str]]:
    """(line, span) for every $$...$$ and $...$ of the prose, in order."""
    clean = strip_code(text)
    out: List[Tuple[int, str]] = []
    for m in _DISPLAY.finditer(clean):
        out.append((clean.count("\n", 0, m.start()) + 1, m.group(1)))
    clean = _DISPLAY.sub(_blank, clean)
    for m in _INLINE.finditer(clean):
        out.append((clean.count("\n", 0, m.start()) + 1, m.group(1)))
    out.sort(key=lambda t: t[0])
    return out


def _split_arith(s: str) -> List[str]:
    """Split on + and - outside braces and parentheses."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        if ch in "+-" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts]


def _split_top_level(s: str) -> List[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def load_master(text: str, heading: str = "## 1. Notation and symbols") -> List[str]:
    """Declared symbols, canonical, from the first cell of every master row."""
    i = text.find(heading)
    if i < 0:
        raise ValueError("master heading not found: %r" % heading)
    j = text.find("\n## ", i + len(heading))
    section = text[i:] if j < 0 else text[i:j]
    declared: List[str] = []
    for line in section.split("\n"):
        if not line.startswith("| $"):
            continue
        cell = line.split("|")[1]
        for m in _INLINE.finditer(cell):
            for part in _split_top_level(m.group(1)):
                toks = tokenize(part)
                form = toks[0] if len(toks) == 1 else _norm(part)
                if form and form not in declared:
                    declared.append(form)
    return declared


# ----------------------------------------------------------------------------
# matching
# ----------------------------------------------------------------------------

def _split(tok: str) -> Tuple[str, List[Tuple[str, str]]]:
    """(base, scripts) of a canonical token; the base keeps its accents."""
    depth = 0
    for i, ch in enumerate(tok):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch in "_^'" and depth == 0 and i > 0:
            parts, _ = _read_scripts(tok, i)
            return tok[:i], parts
    return tok, []


def _unbrace(arg: str) -> str:
    return arg[1:-1] if arg.startswith("{") and arg.endswith("}") else arg


def _index_ok(inner: str, declared: Set[str]) -> bool:
    """A subscript (or a substituted index) that needs no declaration."""
    if not inner or inner in INDEX_FREE or _NUMBER.fullmatch(inner):
        return True
    if _BOUND.fullmatch(inner):            # _{>0}, _{\ge 1}, ^{< 0}
        return True
    if inner in declared:
        return True
    if inner.endswith("'") and inner.rstrip("'") in declared:   # z_{i'}, v_{c'}
        return True
    if "'" in inner:                        # n_{c'} inside a script: the
        toks = tokenize(inner.replace("'", ""))   # primed index of a declared
        if len(toks) == 1 and (toks[0] in declared or   # indexed symbol
                               toks[0].replace("{", "").replace("}", "") in declared):
            return True
    if inner.startswith("(") and inner.endswith(")"):
        return _index_ok(inner[1:-1], declared)
    if "," in inner:
        pieces = _split_top_level(inner)        # a comma inside (...) or {...}
        if len(pieces) > 1:                     # does not split: fall through
            return all(_index_ok(p, declared) for p in pieces)
    rel = _RELATION.split(inner)                # _{c \ne c'}, _{c : n_c \ge 2}
    if len(rel) > 1:
        return all(_index_ok(p, declared) for p in rel if p)
    if "=" in inner:
        left, right = inner.split("=", 1)
        return _index_ok(left, declared) and _index_ok(right, declared)
    if "\\times" in inner:                 # ^{d_\theta \times d_\theta}
        return all(_index_ok(p, declared) for p in inner.split("\\times"))
    if "\\in" in inner:                     # _{w \in \Omega_{\rm st}}
        return all(_index_ok(p, declared) for p in inner.split("\\in") if p)
    arith = _split_arith(inner)             # _{B_{\rm blk} - 1}, ^{E-1}
    if len(arith) > 1:
        return all(_index_ok(p, declared) for p in arith if p)
    pieces = re.findall(r"\\[A-Za-z]+|[A-Za-z]'?|\d+", inner)
    if pieces and "".join(pieces) == inner and len(pieces) > 1:
        return all(_index_ok(p, declared) for p in pieces)
    return False


def _script_ok(kind: str, arg: str, declared: Set[str]) -> bool:
    inner = _unbrace(arg)
    if kind == "'":
        return True
    if kind == "^":
        return inner in DECORATIONS or _index_ok(inner, declared)
    return _index_ok(inner, declared)


def token_ok(tok: str, declared: Set[str], by_base: Dict[str, List[List[Tuple[str, str]]]]) -> bool:
    if tok in declared or tok in OPERATORS or tok in TYPES:
        return True
    base, parts = _split(tok)
    if not parts:
        return False
    if base in declared or base in OPERATORS or base in TYPES:
        if all(_script_ok(k, a, declared) for k, a in parts):
            return True
    # a declared symbol with the same base whose index-like scripts are
    # substituted: \chi^2_{26} for \chi^2_d; T_{gg'}^2 for T_{gg'}
    for dparts in by_base.get(base, []):
        if len(dparts) > len(parts):
            continue
        ok = True
        for (dk, da), (tk, ta) in zip(dparts, parts):
            if dk != tk:
                ok = False
                break
            if da == ta:
                continue
            if _unbrace(da) in declared and _script_ok(tk, ta, declared):
                continue
            ok = False
            break
        if ok and all(_script_ok(k, a, declared) for k, a in parts[len(dparts):]):
            return True
    return False


def check_document(text: str, declared: Sequence[str]) -> List[Tuple[str, int, str]]:
    """Residual tokens of a document: (token, first line, its span)."""
    dset = set(declared)
    by_base: Dict[str, List[List[Tuple[str, str]]]] = {}
    for d in declared:
        b, p = _split(d)
        if p:
            by_base.setdefault(b, []).append(p)
    seen: Dict[str, Tuple[int, str]] = {}
    for line, span in math_spans(text):
        for tok in tokenize(span):
            if token_ok(tok, dset, by_base):
                continue
            if tok not in seen:
                seen[tok] = (line, span.strip())
    return [(t, seen[t][0], seen[t][1]) for t in sorted(seen)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True, help="E0_READERS_GUIDE_NOTATION.md")
    ap.add_argument("--docs", nargs="*", default=[], help="documents to check")
    ap.add_argument("--list", action="store_true", help="print the declared symbols")
    ap.add_argument("--self", action="store_true", help="also check the master's own prose")
    args = ap.parse_args(argv)
    with open(args.master, "r", encoding="ascii") as fh:
        master = fh.read()
    declared = load_master(master)
    if args.list:
        for d in declared:
            print(d)
        print("%d declared symbols" % len(declared))
    rc = 0
    docs = ([args.master] if args.self else []) + list(args.docs)
    for path in docs:
        with open(path, "r", encoding="ascii") as fh:
            text = fh.read()
        spans = math_spans(text)
        residual = check_document(text, declared)
        if residual:
            rc = 1
            print("FAIL notation: %s  (%d spans; %d undeclared symbols)"
                  % (path, len(spans), len(residual)))
            for tok, line, span in residual:
                print("    %-30s line %5d   in  $%s$" % (tok, line, span[:72]))
        else:
            print("OK   notation: %s  (%d spans, every symbol declared)" % (path, len(spans)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
