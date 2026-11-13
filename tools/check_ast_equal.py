# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-13
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Check that Python files differ from a reference git ref only in comments and docstrings.
Inputs  : A git ref (tag, branch, or commit) and one or more directories inside the repository.
Outputs : One line per file (SAME, CHANGED, or NEW) and exit code 1 if any file is CHANGED.
Usage   :
    python tools/check_ast_equal.py results-frozen src scripts tests
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import ast
import pathlib
import subprocess
import sys


# ---------------------------------------------------------------------------
# syntax tree normalization
# ---------------------------------------------------------------------------
def normalized_dump(source):
    '''
    Return the syntax tree of the source as text, with all docstrings removed.
    Comments never appear in the syntax tree, so two sources that differ only
    in comments and docstrings give the same text.
    '''
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            # A docstring is a string expression in the first position of a body.
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                # Keep the body non-empty so that the tree stays valid.
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


# ---------------------------------------------------------------------------
# comparison against the reference
# ---------------------------------------------------------------------------
def main():
    '''
    Compare every *.py file under the given directories with its version at the git ref.
    Print SAME, CHANGED, or NEW per file and exit with 1 when any file is CHANGED.
    '''
    # Stop with a usage line when the reference or the directories are missing.
    if len(sys.argv) < 3:
        print("usage: python tools/check_ast_equal.py <git-ref> <dir> [<dir> ...]")
        sys.exit(2)
    # Read the reference name and the directories to scan from the command line.
    ref, roots = sys.argv[1], sys.argv[2:]
    failed = 0
    for root in roots:
        for path in sorted(pathlib.Path(root).rglob("*.py")):
            rel = path.as_posix()
            # Read the version of the file stored at the reference.
            old = subprocess.run(["git", "show", f"{ref}:{rel}"], capture_output=True)
            if old.returncode != 0:
                # The file does not exist at the reference, so there is nothing to compare.
                print(f"NEW      {rel}")
                continue
            old_dump = normalized_dump(old.stdout.decode("utf-8-sig"))
            new_dump = normalized_dump(path.read_text(encoding="utf-8-sig"))
            same = old_dump == new_dump
            print(f"{'SAME' if same else 'CHANGED'}  {rel}")
            failed += not same
    # Report failure to the caller when any file changed beyond comments and docstrings.
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------------------
# command-line entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
