"""Static audit of src/nbr and scripts/ with the ast module (no project code is executed).

    python scripts/audit_code.py [--repo DIR] [--out docs/CODE_AUDIT.md] [--extra BRANCH:PATH ...]

For every Python file under src/nbr and scripts/ of the repository, plus files of other
branches read with ``git show`` (default: src/nbr/features_cycle.py and src/nbr/r3_eval.py
of feat/w9-r3-cycle), the script reports

  1. module, class and public-function docstrings, and whether each docstring mentions
     every parameter of its function;
  2. calls into third-party libraries (numpy, pandas, scipy, sklearn, joblib, yaml,
     matplotlib, pyarrow): the resolved call name, file and line, keyword arguments;
  3. method calls on objects whose library could not be resolved statically;
  4. standard-library calls (counts only);
  5. source-attribution comments ("adapted from" and similar);
  6. observations recorded by hand in OBSERVATIONS (checked against the current line text).

The output is Markdown (docs/CODE_AUDIT.md) and is deterministic for a given set of inputs
(no timestamps). Resolution levels of a call:

  A  the call starts at an imported third-party name (``np.power``, ``LogisticRegression(...)``);
  B  the receiver is a variable whose type is known from a parameter annotation or from an
     assignment of a level-A expression (``scores: pd.DataFrame`` -> ``pandas.DataFrame.sort_values``,
     ``m = sparse.csr_matrix(...)`` -> ``scipy.sparse.csr_matrix().tocsc``);
  C  the receiver was derived from a level-A/B/C expression by indexing or a method call
     (``events = history[...]`` -> ``pandas.*.merge``; the exact class is not known);
  S  standard-library call (counted separately); F first-party call (nbr, ml_group9; not listed);
  D  unresolved: listed by method name with a heuristic "probable library".
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

THIRD_PARTY = ("numpy", "pandas", "scipy", "sklearn", "joblib", "yaml", "matplotlib", "pyarrow")
FIRST_PARTY = ("nbr", "ml_group9")
STDLIB = set(getattr(sys, "stdlib_module_names", ()))
DEFAULT_EXTRA = ("feat/w9-r3-cycle:src/nbr/features_cycle.py", "feat/w9-r3-cycle:src/nbr/r3_eval.py")
SELF_LABEL = "scripts/audit_code.py"
ATTRIBUTION_PATTERN = re.compile(r"\badapted\b|\bported from\b|\bcopied from\b|\btaken from\b|\bborrowed from\b",
                                 re.IGNORECASE)

# Heuristic receiver library for level-D method calls (method name -> probable library). Used
# only for the "probable library" column; every entry is a guess from the method name alone.
PROBABLE_LIBRARY = {
    **{m: "pandas (DataFrame/Series/Index)" for m in (
        "groupby", "merge", "drop_duplicates", "sort_values", "cumcount", "transform", "agg", "map", "fillna",
        "reindex", "isin", "to_numpy", "rename", "rename_axis", "reset_index", "set_index", "drop", "head",
        "assign", "diff", "shift", "size", "nunique", "unique", "median", "quantile", "value_counts", "to_dict",
        "to_csv", "to_parquet", "get_indexer", "get_level_values", "duplicated", "equals", "notna", "isna",
        "le", "eq", "first", "intersection", "sort_index", "to_frame", "rank",
    )},
    **{m: "pandas or numpy (ambiguous)" for m in ("sum", "mean", "max", "min", "astype", "copy", "tolist", "any", "all")},
    **{m: "dict or pandas (ambiguous)" for m in ("items", "get", "keys", "values", "index")},
    **{m: "numpy.ndarray" for m in ("ravel", "cumsum", "argsort", "nonzero")},
    "item": "numpy scalar (.item() -> Python scalar)",
    **{m: "scipy.sparse matrix" for m in ("tocsc", "tocsr", "toarray", "multiply", "eliminate_zeros", "tocoo", "sort_indices")},
    **{m: "sklearn estimator" for m in ("fit", "transform", "decision_function", "predict", "predict_proba")},
    **{m: "numpy.random.Generator" for m in ("integers", "permutation")},
    **{m: "matplotlib Axes/Figure" for m in (
        "bar", "axhline", "text", "set_xticks", "set_xticklabels", "set_xlabel", "set_ylabel", "set_title", "grid",
        "set_axisbelow", "set_visible", "set_color", "tick_params", "tight_layout", "savefig",
    )},
    **{m: "builtin str/list/float/bytes" for m in (
        "strip", "split", "startswith", "replace", "isalnum", "join", "encode", "lower", "append", "setdefault",
        "extend", "pop", "add", "insert", "format", "endswith", "count", "lstrip", "partition", "is_integer",
        "rsplit", "update", "splitlines",
    )},
    **{m: "pathlib.Path / file object" for m in (
        "mkdir", "exists", "is_file", "read_text", "write_text", "resolve", "read", "write", "with_suffix",
        "rglob", "glob", "iterdir", "relative_to", "as_posix", "parts",
    )},
    **{m: "hashlib / ctypes / argparse object" for m in (
        "hexdigest", "parse_args", "add_argument", "add_parser", "add_subparsers", "sizeof", "byref",
    )},
}

# Observations from reading the code (not detectable by the automatic checks). Each entry names the
# file label as printed in section 1, the line, and a snippet that must occur on that line; a stale
# entry is printed with a warning instead of silently disappearing.
OBSERVATIONS = [
    {"file": "scripts/make_holdout.py", "line": 17, "snippet": "import resource",
     "kind": "portability",
     "text": "`resource` is a POSIX-only module; the script cannot start on Windows. data/SOURCE.md records the "
             "holdout build under Python 3.10.12 (a non-Windows interpreter), consistent with this."},
    {"file": "scripts/check_data.py", "line": 20, "snippet": 'Path("data/processed")',
     "kind": "consistency",
     "text": "Reads the fixed relative path data/processed and ignores NBR_DATA_DIR / configs/common.yaml "
             "(nbr.config honours both); run from the repository root with the data copied in place."},
    {"file": "feat/w9-r3-cycle:src/nbr/r3_eval.py", "line": 57, "snippet": r"C:\dev_ML-wt-w4",
     "kind": "portability",
     "text": "Hard-coded absolute Windows path as the default gamma* source (overridable with --gamma-dir). "
             "configs/usercf.yaml (selection_source.tuning_dir) carries the same machine-specific path."},
    {"file": "src/nbr/usercf_eval.py", "line": 27, "snippet": "from nbr.usercf import _list_from_vectors",
     "kind": "style",
     "text": "Imports a private helper of another module (also src/nbr/usercf.py line 15 imports "
             "nbr.itemcf._topn_with_ties). The helpers are shared by design but are not part of a public interface."},
    {"file": "src/nbr/lr_repeat.py", "line": 101, "snippet": "np.expm1",
     "kind": "numerical",
     "text": "The tie-break popularity is recovered from the log_pop feature with expm1(log1p(pop)); the float "
             "round trip can differ from the integer pop by about 1e-12, which only matters for exact score ties."},
    {"file": "src/nbr/usercf.py", "line": 47, "snippet": "matrix.user_index, matrix.item_index",
     "kind": "fragility",
     "text": "Row/column labels are attached as ad-hoc attributes of a scipy csr_matrix; this works for the "
             "legacy csr_matrix class but would not for the newer sparse-array API (csr_array)."},
    {"file": "src/nbr/__init__.py", "line": 1, "snippet": "core package (config, data, metrics, io)",
     "kind": "documentation",
     "text": "The package docstring lists only the W1 modules; the package now also holds repeat, baselines, "
             "explore, itemcf, combine, pipeline, features, lr_repeat, usercf and usercf_eval."},
    {"file": "src/nbr/pipeline.py", "line": 490, "snippet": "'inf' if math.isinf(hi) else hi",
     "kind": "style",
     "text": "Redundant inner conditional: the branch is only taken when hi is finite, so the 'inf' alternative "
             "cannot occur. Harmless."},
    {"file": "src/nbr/usercf_eval.py", "line": 79, "snippet": "config.update(yaml.safe_load(handle))",
     "kind": "consistency",
     "text": "Merges every top-level key of configs/usercf.yaml into the profile config, including a second "
             "`bootstrap` block; its values equal configs/common.yaml today, so no behaviour differs."},
    {"file": "src/nbr/metrics.py", "line": 40, "snippet": "for user, items in targets.items():",
     "kind": "performance",
     "text": "per_user_recall / per_user_ndcg / per_user_category_hit loop over users in Python (unlike the "
             "vectorised list builders); correct, but the slowest step for large eval splits."},
    {"file": "src/nbr/usercf_eval.py", "line": 127, "snippet": "print(",
     "kind": "style",
     "text": "Library function prints progress to stdout (also line 248); nbr.pipeline reuses tune_history, so "
             "`python -m nbr tune` output interleaves these lines."},
]


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
@dataclass
class Source:
    """One audited file: ``label`` as printed, its ``text`` and where it came from (``origin``)."""

    label: str          # path as printed, e.g. "src/nbr/explore.py" or "feat/w9-r3-cycle:src/nbr/r3_eval.py"
    text: str
    origin: str         # "worktree" or "git show <branch> (<commit>)"

    @property
    def sha256(self) -> str:
        """Hex SHA-256 of the UTF-8 encoded file text."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def lines(self) -> list[str]:
        """The file text split into lines (without line endings)."""
        return self.text.splitlines()


def git(repo: Path, *args: str) -> str:
    """Run ``git <args>`` in ``repo`` and return its stripped stdout."""
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def collect_sources(repo: Path, extras: list[str]) -> list[Source]:
    """All ``*.py`` files under src/nbr and scripts of ``repo`` plus the ``extras`` (BRANCH:PATH via git show)."""
    sources = []
    for folder in ("src/nbr", "scripts"):
        for path in sorted((repo / folder).glob("*.py")):
            sources.append(Source(path.relative_to(repo).as_posix(), path.read_text(encoding="utf-8"), "worktree"))
    for spec in extras:
        branch, _, path = spec.partition(":")
        commit = git(repo, "rev-parse", "--short", branch)
        text = subprocess.run(["git", "show", f"{branch}:{path}"], cwd=repo, capture_output=True,
                              text=True, check=True, encoding="utf-8").stdout
        sources.append(Source(f"{branch}:{path}", text, f"git show {branch} ({commit})"))
    return sources


# ---------------------------------------------------------------------------
# docstrings
# ---------------------------------------------------------------------------
@dataclass
class FunctionInfo:
    """Docstring facts of one function, method or class: ``qualname``, ``params`` and the unmentioned ones."""

    file: str
    qualname: str
    line: int
    public: bool
    is_class: bool
    params: list[str]
    docstring: str | None
    undocumented: list[str] = field(default_factory=list)


def _params(node: ast.FunctionDef | ast.AsyncFunctionDef, method: bool) -> list[str]:
    args = node.args
    names = [a.arg for a in args.posonlyargs + args.args]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    if method and names and names[0] in ("self", "cls"):
        names = names[1:]
    return names


def _mentioned(name: str, doc: str) -> bool:
    bare = name.lstrip("*")
    return re.search(rf"(?<![\w.]){re.escape(bare)}(?![\w])", doc) is not None


def function_infos(source: Source, tree: ast.Module) -> list[FunctionInfo]:
    """FunctionInfo for every top-level function/class of ``tree`` (from ``source``) and every method."""
    infos = []

    def visit(node, prefix: str, method: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = _params(node, method)
            doc = ast.get_docstring(node)
            info = FunctionInfo(source.label, prefix + node.name, node.lineno, not node.name.startswith("_"),
                                False, params, doc)
            if doc is not None:
                info.undocumented = [p for p in params if not _mentioned(p, doc)]
            infos.append(info)
        elif isinstance(node, ast.ClassDef):
            infos.append(FunctionInfo(source.label, prefix + node.name, node.lineno, not node.name.startswith("_"),
                                      True, [], ast.get_docstring(node)))
            for child in node.body:
                visit(child, prefix + node.name + ".", True)

    for child in tree.body:
        visit(child, "", False)
    return infos


# ---------------------------------------------------------------------------
# call resolution
# ---------------------------------------------------------------------------
@dataclass
class CallSite:
    """One call expression: resolved ``name``, resolution ``level`` (A/B/C/S/D), ``library`` and keyword names."""

    file: str
    line: int
    name: str                 # resolved dotted name (levels A-C, S) or method name (level D)
    level: str
    library: str              # e.g. "pandas"; "stdlib" for level S; "" for level D
    keywords: list[str]
    positional: int
    receiver: str = ""        # receiver expression for level D


def _root_library(dotted: str) -> str:
    return dotted.split(".")[0]


def _classify(dotted: str) -> str:
    root = _root_library(dotted)
    if root in THIRD_PARTY:
        return "A"
    if root in FIRST_PARTY:
        return "F"
    if root == "typing" or dotted.startswith("collections.abc"):
        return "T"  # annotation-only names (typing.Any, collections.abc.Sequence): never bound or counted
    if root in STDLIB or root == "__future__":
        return "S"
    return "A"  # unknown package: treated as third-party


class Resolver(ast.NodeVisitor):
    """Collect import aliases, variable type labels and call sites of one module (``source``)."""

    def __init__(self, source: Source):
        self.source = source
        self.aliases: dict[str, str] = {}         # local name -> dotted import name
        self.scopes: list[dict[str, tuple[str, str]]] = [{}]   # var -> (label, level)
        self.calls: list[CallSite] = []

    # -- imports (collected everywhere, including inside functions)
    def visit_Import(self, node: ast.Import) -> None:
        """Register ``import a.b as c`` aliases (``node``)."""
        for alias in node.names:
            if alias.asname:
                self.aliases[alias.asname] = alias.name
            else:
                self.aliases[alias.name.split(".")[0]] = alias.name.split(".")[0]
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Register ``from a import b as c`` aliases (``node``)."""
        module = node.module or ""
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = f"{module}.{alias.name}" if module else alias.name
        self.generic_visit(node)

    # -- scopes
    def _bind(self, name: str, label: str, level: str) -> None:
        self.scopes[-1][name] = (label, level)

    def _lookup(self, name: str) -> tuple[str, str] | None:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Open a scope for ``node``, bind annotated parameters, visit the body."""
        self.scopes.append({})
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if arg.annotation is not None:
                resolved = self._resolve_expr(arg.annotation)
                if resolved is not None and resolved[1] in ("A", "S", "F"):
                    self._bind(arg.arg, resolved[0], _bound_level(resolved[1]))
        self.generic_visit(node)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        """Bind the targets of ``node`` to the resolved type label of its value."""
        self.visit(node.value)
        resolved = self._resolve_expr(node.value)
        for target in node.targets:
            self._bind_target(target, resolved)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Bind an annotated assignment ``node`` through its annotation, else through its value."""
        if node.value is not None:
            self.visit(node.value)
        resolved = self._resolve_expr(node.annotation)
        if isinstance(node.target, ast.Name):
            if resolved is not None and resolved[1] in ("A", "S", "F"):
                self._bind(node.target.id, resolved[0], _bound_level(resolved[1]))
            elif node.value is not None:
                self._bind_target(node.target, self._resolve_expr(node.value))

    def _bind_target(self, target: ast.expr, resolved: tuple[str, str] | None) -> None:
        if isinstance(target, ast.Name):
            if resolved is None:
                self.scopes[-1].pop(target.id, None)
            else:
                self._bind(target.id, resolved[0], _bound_level(resolved[1]))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, sub_target in enumerate(target.elts):
                if isinstance(sub_target, ast.Name):
                    if resolved is None:
                        self.scopes[-1].pop(sub_target.id, None)
                    else:
                        self._bind(sub_target.id, f"{resolved[0]}[{i}]", _bound_level(resolved[1]))

    # -- expression resolution: (label, level) or None
    def _resolve_expr(self, node: ast.expr) -> tuple[str, str] | None:
        if isinstance(node, ast.Name):
            if node.id in self.aliases:
                dotted = self.aliases[node.id]
                return dotted, _classify(dotted)
            return self._lookup(node.id)
        if isinstance(node, ast.Attribute):
            base = self._resolve_expr(node.value)
            if base is None:
                return None
            return f"{base[0]}.{node.attr}", base[1]
        if isinstance(node, ast.Call):
            base = self._resolve_expr(node.func)
            if base is None:
                return None
            return f"{base[0]}()", _derived_level(base[1])
        if isinstance(node, ast.Subscript):
            base = self._resolve_expr(node.value)
            if base is None:
                return None
            return f"{base[0]}[...]", _derived_level(base[1])
        if isinstance(node, ast.Tuple) and node.elts:
            # (a, b) = (expr1, expr2): resolved element-wise through the "[i]" suffix of _bind_target
            return None
        return None

    def visit_Call(self, node: ast.Call) -> None:
        """Record the call ``node`` with its resolution level and keyword-argument names."""
        keywords = [kw.arg if kw.arg is not None else "**" for kw in node.keywords]
        positional = len(node.args)
        resolved = self._resolve_expr(node.func)
        if resolved is not None:
            label, level = resolved
            if level in ("F", "T"):
                pass  # first-party call or typing name: not listed
            elif level == "S":
                self.calls.append(CallSite(self.source.label, node.lineno, label, "S", "stdlib", keywords, positional))
            elif level == "C":
                self.calls.append(CallSite(self.source.label, node.lineno, _collapse_derived(label), "C",
                                           _root_library(label), keywords, positional))
            else:
                self.calls.append(CallSite(self.source.label, node.lineno, label, level, _root_library(label),
                                           keywords, positional))
        elif isinstance(node.func, ast.Attribute):
            receiver = ast.unparse(node.func.value)
            if receiver not in ("self", "cls"):
                self.calls.append(CallSite(self.source.label, node.lineno, node.func.attr, "D", "", keywords,
                                           positional, receiver=receiver))
        # bare names (builtins, local functions, first-party imports) are not recorded
        self.generic_visit(node)


def _bound_level(level: str) -> str:
    """Level of a variable bound to an expression of ``level``: A -> B, S/F stay, everything else C."""
    return {"A": "B", "S": "S", "F": "F"}.get(level, "C")


def _derived_level(level: str) -> str:
    """Level of a call result or subscript of an expression of ``level``: A/S/F stay, B/C become C."""
    return level if level in ("A", "S", "F") else "C"


def _collapse_derived(label: str) -> str:
    """"pandas.DataFrame[...].merge().groupby" -> "pandas.*.groupby" (exact class unknown after derivation)."""
    library = _root_library(label)
    method = label.rsplit(".", 1)[-1]
    return f"{library}.*.{method}"


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Markdown table lines for ``headers`` and ``rows`` (pipes and newlines in cells are escaped)."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in row) + " |")
    return out


def _sites(sites: list[CallSite]) -> str:
    return ", ".join(f"{s.file}:{s.line}" for s in sorted(sites, key=lambda s: (s.file, s.line)))


def _kw_union(sites: list[CallSite]) -> str:
    counter: Counter[str] = Counter()
    for s in sites:
        counter.update(s.keywords)
    return ", ".join(f"{k}" for k, _ in sorted(counter.items())) or "-"


def build_report(repo: Path, sources: list[Source], extras: list[str]) -> str:
    """Render the Markdown audit of ``sources`` (``extras`` and the HEAD of ``repo`` are printed in the header)."""
    head = git(repo, "rev-parse", "--short", "HEAD")
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    lines: list[str] = []
    lines.append("<!-- AUTO-GENERATED by scripts/audit_code.py (static ast audit; no project code is executed). "
                 "Do not edit by hand: re-run the script. -->")
    lines.append("# Code audit (static)")
    lines.append("")
    lines.append(f"Generated by `python scripts/audit_code.py` from worktree HEAD `{head}` (branch `{branch}`); "
                 f"extra files read with `git show`: " + ", ".join(f"`{e}`" for e in extras) + ".")
    lines.append("The audit is static (ast module); nothing is imported or executed. "
                 "Rules are listed at the end of this document.")
    lines.append("")

    trees: dict[str, ast.Module] = {}
    infos: list[FunctionInfo] = []
    calls: list[CallSite] = []
    for source in sources:
        tree = ast.parse(source.text, filename=source.label)
        trees[source.label] = tree
        infos += function_infos(source, tree)
        resolver = Resolver(source)
        resolver.visit(tree)
        calls += resolver.calls

    # 1. scope
    lines.append("## 1. Scope")
    lines.append("")
    rows = [[s.label, str(len(s.lines)), s.sha256, s.origin] for s in sources]
    lines += md_table(["file", "lines", "sha256 (file content)", "source"], rows)
    lines.append("")

    # 2. summary
    lines.append("## 2. Docstring summary per file")
    lines.append("")
    rows = []
    totals: Counter[str] = Counter()
    for source in sources:
        mine = [i for i in infos if i.file == source.label]
        classes = [i for i in mine if i.is_class]
        funcs = [i for i in mine if not i.is_class]
        public = [i for i in funcs if i.public]
        private = [i for i in funcs if not i.public]
        no_doc = [i for i in public if i.docstring is None]
        undocumented = [i for i in public if i.docstring is not None and i.undocumented]
        module_doc = ast.get_docstring(trees[source.label]) is not None
        rows.append([source.label, "yes" if module_doc else "no", str(len(classes)), str(len(public)),
                     str(len(no_doc)), str(len(undocumented)), str(len(private)),
                     str(sum(1 for i in private if i.docstring is None))])
        totals.update({"public": len(public), "no_doc": len(no_doc), "undocumented": len(undocumented),
                       "private": len(private), "private_no_doc": sum(1 for i in private if i.docstring is None),
                       "module_doc": int(module_doc), "classes": len(classes),
                       "class_no_doc": sum(1 for i in classes if i.docstring is None)})
    lines += md_table(["file", "module docstring", "classes", "public functions", "public without docstring",
                       "public with unmentioned parameters", "private functions", "private without docstring"], rows)
    lines.append("")
    lines.append(f"Totals: {totals['public']} public functions/methods, {totals['no_doc']} without a docstring, "
                 f"{totals['undocumented']} with a docstring that does not mention every parameter; "
                 f"{totals['private']} private functions ({totals['private_no_doc']} without a docstring; no requirement); "
                 f"{totals['module_doc']}/{len(sources)} modules have a module docstring; "
                 f"{totals['classes']} classes ({totals['class_no_doc']} without a docstring).")
    lines.append("")

    # 3. public functions without docstring
    lines.append("## 3. Public functions and classes without a docstring")
    lines.append("")
    missing = [i for i in infos if i.public and i.docstring is None]
    if missing:
        rows = [[i.file, str(i.line), i.qualname, ", ".join(i.params) or "-"] for i in missing]
        lines += md_table(["file", "line", "name", "parameters"], rows)
    else:
        lines.append("None.")
    lines.append("")

    # 4. parameters not mentioned
    lines.append("## 4. Public functions whose docstring does not mention every parameter")
    lines.append("")
    lines.append("A parameter counts as described when its name occurs as a whole word in the function's own docstring "
                 "(plain or in double backticks). Parameters that are only described in the module docstring or by their "
                 "type annotation are listed here.")
    lines.append("")
    partial = [i for i in infos if i.public and i.docstring is not None and i.undocumented]
    if partial:
        rows = [[i.file, str(i.line), i.qualname, ", ".join(i.undocumented),
                 f"{len(i.params) - len(i.undocumented)}/{len(i.params)}"] for i in partial]
        lines += md_table(["file", "line", "function", "parameters not mentioned", "mentioned/total"], rows)
    else:
        lines.append("None.")
    lines.append("")

    # 5. third-party calls
    lines.append("## 5. Third-party library calls (resolved statically)")
    lines.append("")
    lines.append("Levels: A = call starts at an imported name; B = receiver type known from an annotation or an "
                 "assignment of a level-A expression; C = receiver derived from a library object by indexing or a "
                 "method call (exact class unknown, shown as `library.*.method`). Keyword arguments are the union "
                 "over all sites.")
    lines.append("")
    resolved = [c for c in calls if c.level in ("A", "B", "C")]
    by_library: dict[str, dict[str, list[CallSite]]] = defaultdict(lambda: defaultdict(list))
    for c in resolved:
        by_library[c.library][c.name].append(c)
    for number, library in enumerate(sorted(by_library), start=1):
        lines.append(f"### 5.{number} {library}")
        lines.append("")
        rows = []
        for name in sorted(by_library[library], key=lambda n: (n.count("*"), n)):
            sites = by_library[library][name]
            level = "".join(sorted(set(s.level for s in sites)))
            rows.append([f"`{name}`", level, str(len(sites)), _kw_union(sites), _sites(sites)])
        lines += md_table(["call", "level", "sites", "keyword arguments used", "file:line"], rows)
        lines.append("")
    lines.append("### 5.x Per-site keyword arguments of level-A and level-B calls")
    lines.append("")
    rows = []
    for c in sorted(resolved, key=lambda c: (c.library, c.name, c.file, c.line)):
        if c.level == "C":
            continue
        rows.append([f"`{c.name}`", f"{c.file}:{c.line}", str(c.positional), ", ".join(c.keywords) or "-"])
    lines += md_table(["call", "file:line", "positional args", "keyword args"], rows)
    lines.append("")

    # 6. unresolved method calls
    lines.append("## 6. Method calls on objects of unresolved type (level D)")
    lines.append("")
    lines.append("The receiver's library could not be determined statically (e.g. a value returned by a first-party "
                 "function). The probable library is a heuristic from the method name only. Calls on `self` are omitted.")
    lines.append("")
    unresolved = [c for c in calls if c.level == "D"]
    by_method: dict[str, list[CallSite]] = defaultdict(list)
    for c in unresolved:
        by_method[c.name].append(c)
    rows = []
    for method in sorted(by_method):
        sites = by_method[method]
        receivers = sorted(set(s.receiver for s in sites))
        files = sorted(set(s.file for s in sites))
        rows.append([f"`.{method}`", PROBABLE_LIBRARY.get(method, "unknown (no entry in the heuristic table)"),
                     str(len(sites)), _kw_union(sites),
                     ", ".join(receivers[:5]) + (" ..." if len(receivers) > 5 else ""),
                     ", ".join(files)])
    lines += md_table(["method", "probable library (heuristic)", "sites", "keyword arguments used", "receivers (first 5)",
                       "files"], rows)
    lines.append("")

    # 7. stdlib
    lines.append("## 7. Standard-library calls (counts)")
    lines.append("")
    stdlib_calls = [c for c in calls if c.level == "S"]
    by_name: dict[str, list[CallSite]] = defaultdict(list)
    for c in stdlib_calls:
        by_name[c.name].append(c)
    rows = []
    for name in sorted(by_name):
        sites = by_name[name]
        rows.append([f"`{name}`", str(len(sites)), _kw_union(sites), ", ".join(sorted(set(s.file for s in sites)))])
    lines += md_table(["call", "sites", "keyword arguments used", "files"], rows)
    lines.append("")

    # 8. attribution comments
    lines.append("## 8. Source-attribution comments")
    lines.append("")
    lines.append("Lines matching `adapted`, `ported from`, `copied from`, `taken from` or `borrowed from` "
                 "(case-insensitive) in comments or docstrings; scripts/audit_code.py itself is skipped because it quotes the patterns.")
    lines.append("")
    rows = []
    for source in sources:
        if source.label == SELF_LABEL:
            continue
        for number, text in enumerate(source.lines, start=1):
            if ATTRIBUTION_PATTERN.search(text):
                rows.append([f"{source.label}:{number}", text.strip()])
    lines += md_table(["file:line", "text"], rows) if rows else ["None."]
    lines.append("")

    # 9. observations
    lines.append("## 9. Observations from manual reading")
    lines.append("")
    lines.append("Maintained in `OBSERVATIONS` inside scripts/audit_code.py; each entry is checked against the current "
                 "text of the named line. src/ is not modified by W12; these are notes for the team.")
    lines.append("")
    by_label = {s.label: s for s in sources}
    rows = []
    for i, obs in enumerate(OBSERVATIONS, start=1):
        source = by_label.get(obs["file"])
        status = "ok"
        if source is None:
            status = "STALE: file not audited"
        elif not (1 <= obs["line"] <= len(source.lines)):
            status = "STALE: line out of range"
        elif obs["snippet"] not in source.lines[obs["line"] - 1]:
            status = "STALE: snippet not on this line"
        rows.append([str(i), f"{obs['file']}:{obs['line']}", obs["kind"], obs["text"], status])
    lines += md_table(["#", "file:line", "kind", "observation", "check"], rows)
    lines.append("")

    # 10. rules
    lines.append("## 10. Rules used by this audit")
    lines.append("")
    lines += [
        "- Files: every `*.py` under `src/nbr` and `scripts` of the worktree, plus the `--extra BRANCH:PATH` files read with `git show`.",
        "- Public = name does not start with an underscore (dunder names count as private). Nested functions are not listed.",
        "- Module docstring: `ast.get_docstring(module)` is not None. Function/class docstring: same on the node.",
        "- Parameter mention: whole-word occurrence of the parameter name in the function's own docstring (`self`/`cls` excluded; `*args`/`**kwargs` matched by their bare name).",
        "- Third-party libraries: " + ", ".join(THIRD_PARTY) + ". Imports inside functions count. First-party calls (" + ", ".join(FIRST_PARTY) + ") and bare-name calls (builtins, local functions) are not listed; standard-library modules (sys.stdlib_module_names) are counted in section 7.",
        "- Resolution: level A through import aliases; level B through annotations (`x: pd.DataFrame`) and assignments from level-A expressions (`m = sparse.csr_matrix(...)`, tuple unpacking as `call()[i]`); level C when the receiver was derived by indexing or a method call; everything else level D.",
        "- Control flow is ignored: the last assignment to a name in source order wins inside its scope; a reassignment from an unresolvable value clears the binding.",
        "- Keyword arguments are listed by name only; positional values are not inspected.",
    ]
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: parse ``argv`` (``--repo``, ``--out``, ``--extra``) and write the audit."""
    parser = argparse.ArgumentParser(description="static audit of src/nbr and scripts/")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out", default="docs/CODE_AUDIT.md", help="output path relative to --repo")
    parser.add_argument("--extra", action="append", default=None, help="BRANCH:PATH read with git show (repeatable)")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    extras = args.extra if args.extra is not None else list(DEFAULT_EXTRA)
    sources = collect_sources(repo, extras)
    report = build_report(repo, sources, extras)
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(sources)} files audited)")


if __name__ == "__main__":
    main()
