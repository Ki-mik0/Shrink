from __future__ import annotations

import ast
import textwrap

import pytest
from click.testing import CliRunner

from shrink import validation
from shrink.cli import cli
from shrink.compressors.auto import AutoCompressor, is_python
from shrink.compressors.python_code import (
    STEP_NAMES,
    CodeCompressor,
    shorten_docstrings,
    skeletonize,
    strip_comments,
)
from shrink.document import Document


def src(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


# ------------------------------------------------------------------ level 1: comments


def test_comments_and_comment_only_lines_are_removed() -> None:
    code = src(
        """
        # header comment
        x = 1  # trailing
        # between
        y = 2
        """
    )
    assert strip_comments(code) == "x = 1\ny = 2\n"


def test_shebang_encoding_and_pragmas_are_kept() -> None:
    code = src(
        """
        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        import os  # noqa: F401
        x: int = "a"  # type: ignore[assignment]
        y = 1  # pragma: no cover
        z = 2  # just a note
        """
    )
    assert strip_comments(code) == src(
        """
        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        import os  # noqa: F401
        x: int = "a"  # type: ignore[assignment]
        y = 1  # pragma: no cover
        z = 2
        """
    )


def test_strings_are_never_touched() -> None:
    code = 's = """keep   \n# not a comment\n\n\n\nend"""\nt = "# no"  # yes\n'
    assert strip_comments(code) == 's = """keep   \n# not a comment\n\n\n\nend"""\nt = "# no"\n'


def test_multiline_fstring_is_protected() -> None:
    code = 'name = 1\ns = f"""a {name}   \n\n\nb"""\n'
    assert strip_comments(code) == code


def test_blank_line_runs_are_squeezed_and_edges_trimmed() -> None:
    code = "\n\nx = 1\n\n\n\ndef f():\n    pass\n\n\n"
    assert strip_comments(code) == "x = 1\n\ndef f():\n    pass\n"


def test_cyrillic_before_a_comment() -> None:
    code = 'привіт = "світ"  # коментар\n'
    assert strip_comments(code) == 'привіт = "світ"\n'


def test_crlf_and_bom_are_preserved() -> None:
    code = "\ufeffx = 1  # c\r\n\r\n\r\ny = 2\r\n"
    assert strip_comments(code) == "\ufeffx = 1\r\n\r\ny = 2\r\n"


def test_lone_cr_is_rejected() -> None:
    with pytest.raises(ValueError, match="lone CR"):
        strip_comments("x = 1\ry = 2\r")


def test_empty_and_blank_input_pass_through() -> None:
    assert strip_comments("") == ""
    assert strip_comments("\n\n") == "\n\n"


# ------------------------------------------------------------------ level 2: docstrings


def test_multiline_docstrings_keep_their_first_line() -> None:
    code = src(
        '''
        """Module summary.

        Details.
        """


        class A:
            """Class summary.

            More.
            """

            def m(self):
                """Method summary.
                More.
                """
                return 1
        '''
    )
    out = shorten_docstrings(code)
    assert '"""Module summary."""' in out
    assert '    """Class summary."""' in out
    assert '        """Method summary."""\n        return 1' in out
    assert "Details" not in out
    assert "More" not in out


def test_one_line_docstrings_keep_their_quoting() -> None:
    code = "def f():\n    'Short.'\n    return 1\n"
    assert shorten_docstrings(code) == code


def test_summary_that_cannot_be_triple_quoted_uses_repr() -> None:
    code = 'def f():\n    r"""Uses \\d+ here.\n\n    More.\n    """\n'
    out = shorten_docstrings(code)
    assert ast.get_docstring(ast.parse(out).body[0]) == "Uses \\d+ here."  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("content", "summary"),
    [
        ("Short.\n\nMore.", "Short."),
        ("Object for wrapping text.  The public\ninterface is here.", "Object for wrapping text."),
        (
            "A sentence that wraps\nonto the next line.\n\nDetails.",
            "A sentence that wraps onto the next line.",
        ),
        ("No full stop here\nand more", "No full stop here"),
        ("x" * 250 + ".\nrest", "x" * 250 + "."),
    ],
)
def test_docstring_summary(content: str, summary: str) -> None:
    from shrink.compressors.python_code import docstring_summary

    expected = summary if len(summary) <= 200 else content.split("\n", 1)[0]
    assert docstring_summary(content) == expected


def test_cyrillic_docstring() -> None:
    code = 'def f():\n    """Привітати.\n\n    Довгий опис.\n    """\n'
    assert shorten_docstrings(code) == 'def f():\n    """Привітати."""\n'


# ------------------------------------------------------------------ level 3: skeleton


def test_function_bodies_become_ellipsis_and_signatures_stay() -> None:
    code = src(
        '''
        import os


        @decorator(1)
        def greet(name: str = "світ", *, loud: bool = False) -> str:
            """Say hello."""
            msg = f"hi {name}"
            return msg.upper() if loud else msg


        class A:
            x: int = 1

            async def run(self) -> None:
                await other()

            def stub(self) -> int: ...


        def one(): return 1
        '''
    )
    assert skeletonize(code) == src(
        '''
        import os


        @decorator(1)
        def greet(name: str = "світ", *, loud: bool = False) -> str:
            """Say hello."""
            ...


        class A:
            x: int = 1

            async def run(self) -> None:
                ...

            def stub(self) -> int: ...


        def one(): ...
        '''
    )


def test_nested_and_decorated_functions_vanish_with_the_outer_body() -> None:
    code = src(
        """
        def outer():
            @wraps(f)
            def inner():
                pass
            return inner
        """
    )
    assert skeletonize(code) == "def outer():\n    ...\n"


def test_functions_inside_module_level_blocks_are_reduced() -> None:
    code = src(
        """
        if TYPE_CHECKING:
            def f() -> int:
                return 1
        try:
            import x
        except ImportError:
            def g():
                return 2
        """
    )
    out = skeletonize(code)
    assert "return" not in out
    assert "import x" in out


def test_docstring_only_function_is_left_alone() -> None:
    code = 'def f():\n    """Only a docstring."""\n'
    assert skeletonize(code) == code


# ------------------------------------------------------------------ compressor, trace, failures


@pytest.mark.parametrize("level", [1, 2, 3])
def test_trace_steps_end_with_the_compress_result(level: int) -> None:
    code = 'def f():\n    """A.\n\n    B.\n    """\n    return 1  # c\n'
    doc = Document(code, path="a.py")
    steps = CodeCompressor().trace(doc, level)
    assert [s.name for s in steps] == [STEP_NAMES[k] for k in range(1, level + 1)]
    assert steps[-1].after == CodeCompressor().compress(doc, level)
    assert all(len(s.after.text) <= len(code) for s in steps)


def test_invalid_python_raises_so_the_pipeline_can_fail_open() -> None:
    with pytest.raises(SyntaxError):
        CodeCompressor().compress(Document("def broken(:\n"), 1)


def test_validation_rejects_changed_code() -> None:
    with pytest.raises(validation.ValidationError, match="changed the code"):
        validation.check_same_code(ast.parse("x = 1\n"), "x = 2\n")


def test_validation_rejects_changed_signatures() -> None:
    with pytest.raises(validation.ValidationError, match="outside function bodies"):
        validation.check_same_signatures(ast.parse("def f(a): pass\n"), "def f(b): ...\n")


def test_validation_rejects_invalid_output() -> None:
    with pytest.raises(validation.ValidationError, match="not valid Python"):
        validation.check_same_code(ast.parse("x = 1\n"), "x = (\n")


# ------------------------------------------------------------------ auto mode


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        (Document("anything", path="pkg/mod.py"), True),
        (Document("anything", path="stubs/mod.PYI"), True),
        (Document("import os\n", path="notes.md"), False),
        (Document("import os\nx = 1\n"), True),
        (Document("hello world\n"), False),
        (Document("x = 1\n"), False),
        (Document("def broken(:\n"), False),
    ],
)
def test_is_python(doc: Document, expected: bool) -> None:
    assert is_python(doc) is expected


def test_auto_routes_by_document() -> None:
    auto = AutoCompressor()
    py = Document("def f():\n    return 1  # c\n", path="a.py")
    md = Document("a   b\n", path="a.md")
    assert auto.compress(py, 3).text == "def f():\n    ...\n"
    assert auto.compress(md, 2).text == "a b\n"
    assert [s.name for s in auto.trace(py, 2)] == ["comments", "docstrings"]
    assert [s.name for s in auto.trace(md, 1)] == ["whitespace"]


def test_auto_trace_of_an_untraceable_mode_is_one_step() -> None:
    class Plain:
        name = "plain"
        summary = ""
        help = ""

        def compress(self, doc: Document, level: int) -> Document:
            return doc.with_text("x")

    auto = AutoCompressor(code=Plain(), text=Plain())
    assert [s.name for s in auto.trace(Document("hi"), 1)] == ["plain"]


# ------------------------------------------------------------------ CLI


def test_default_mode_handles_python_and_text_in_one_packed_stream(runner: CliRunner) -> None:
    packed = "a.py\n---\ndef f(x: int) -> int:\n    return x * 2  # double\n\n---\n"
    packed += "b.md\n---\nsome    text\n\n---\n"
    result = runner.invoke(cli, ["--level", "3"], input=packed)
    assert result.exit_code == 0
    assert result.stdout == (
        "a.py\n---\ndef f(x: int) -> int:\n    ...\n\n---\nb.md\n---\nsome text\n\n---\n"
    )


def test_code_mode_on_non_python_passes_through_with_a_warning(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--mode", "code"], input="this is: not python (\n")
    assert result.exit_code == 0
    assert result.stdout == "this is: not python (\n"
    assert "code mode failed" in result.stderr
    assert "passed through unchanged" in result.stderr


def test_mode_help_code(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--mode-help", "code"])
    assert result.exit_code == 0
    assert "level 3  skeleton" in result.stdout
