from __future__ import annotations

import pytest

from shrink.edits import Edit, EditError, apply_edits, split_edits

ORIGINAL = """import os

# Settings live here.
def load(path):
    # read it
    with open(path) as f:
        return f.read()


def save(path, text):
    pass
"""


def test_split_edits_takes_edit_sections_out() -> None:
    answer = (
        "## File: a.py\n```python\nx = 1\n```\n"
        "## Edit: b.py\n```\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n```\n"
        "## File: c.py\n```python\ny = 2\n```\n"
    )
    rest, edits = split_edits(answer)
    assert "## Edit" not in rest
    assert "## File: a.py" in rest
    assert "## File: c.py" in rest
    assert [(e.path, e.edits) for e in edits] == [("b.py", [Edit("old\n", "new\n")])]


def test_exact_edit() -> None:
    out = apply_edits(ORIGINAL, [Edit("import os\n", "import os\nimport sys\n")])
    assert out.startswith("import os\nimport sys\n\n# Settings")


def test_loose_edit_ignores_missing_comments_and_indentation() -> None:
    # what the model saw after comments were stripped, and without indentation
    seen = "def load(path):\nwith open(path) as f:\n    return f.read()\n"
    new = "def load(path):\n    return open(path).read()\n"
    out = apply_edits(ORIGINAL, [Edit(seen, new)])
    assert "def load(path):\n    return open(path).read()\n\n\ndef save" in out
    assert "# Settings live here." in out


def test_loose_edit_shifts_indentation_to_the_real_one() -> None:
    src = "def f():\n    a = 1\n    b = 2\n"
    out = apply_edits(src, [Edit("a = 1\nb = 2\n", "a = 1\nc = 3\nb = 2\n")])
    assert out == "def f():\n    a = 1\n    c = 3\n    b = 2\n"


def test_empty_search_appends() -> None:
    assert apply_edits("a\n", [Edit("", "b\n")]) == "a\nb\n"


@pytest.mark.parametrize("search", ["missing line\n", "pass\n" * 0 + "    \n# only\n"])
def test_edit_that_does_not_match_fails(search: str) -> None:
    with pytest.raises(EditError):
        apply_edits(ORIGINAL, [Edit(search, "x\n")])


def test_ambiguous_edit_fails() -> None:
    with pytest.raises(EditError, match="2 times"):
        apply_edits("x = 1\ny = 2\nx = 1\n", [Edit("x = 1\n", "z\n")])
