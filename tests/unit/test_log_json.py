from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from hypothesis import given, settings
from hypothesis import strategies as st

from shrink.cli import cli
from shrink.compressors.auto import AutoCompressor, choose_mode
from shrink.compressors.json_mode import JsonCompressor, compress_json, looks_like_json
from shrink.compressors.log import LogCompressor, compress_log, looks_like_log
from shrink.document import Document

ESC = chr(27)
BOM = chr(0xFEFF)

# ------------------------------------------------------------------ log, level 1


def test_ansi_codes_are_removed() -> None:
    text = f"{ESC}[32mok{ESC}[0m {ESC}]0;title\a done\n"
    assert compress_log(text, 1) == "ok  done\n"


def test_carriage_return_progress_keeps_the_last_state() -> None:
    assert compress_log("10%\r50%\r100%\nnext\n", 1) == "100%\nnext\n"


def test_identical_neighbours_are_merged_with_a_count() -> None:
    text = "same line\nsame line\nsame line\nb\nsame line\n"
    assert compress_log(text, 1) == "same line [x3]\nb\nsame line\n"


def test_merging_that_would_be_longer_is_skipped() -> None:
    assert compress_log("a\na\nb\n", 1) == "a\na\nb\n"


def test_blank_runs_are_squeezed() -> None:
    assert compress_log("a\n\n\n\nb\n", 1) == "a\n\nb\n"


def test_crlf_is_kept() -> None:
    text = "same line\r\nsame line\r\nb\r\n"
    assert compress_log(text, 1) == "same line [x2]\r\nb\r\n"


# ------------------------------------------------------------------ log, level 2


@pytest.mark.parametrize(
    "line",
    [
        "2026-11-21T10:00:00.123Z hello",
        "2026-11-21 10:00:00,123 - hello",
        "[2026-11-21 10:00:00] hello",
        "10:00:00 hello",
        "Nov 21 10:00:00 hello",
        "(10:00:00.5) hello",
    ],
)
def test_leading_timestamps_are_removed(line: str) -> None:
    assert compress_log(line + "\n", 2) == "hello\n"


def test_lines_equal_after_timestamps_merge() -> None:
    text = "10:00:01 tick\n10:00:02 tick\n10:00:03 tick\n"
    assert compress_log(text, 2) == "tick [x3]\n"


def test_repeating_blocks_are_kept_once() -> None:
    text = "retry\nwait\nretry\nwait\nretry\nwait\nend\n"
    assert compress_log(text, 2) == "retry\nwait\n[previous 2 lines x3]\nend\n"


def test_numbers_inside_a_line_are_not_timestamps() -> None:
    assert compress_log("took 10:00:00 hours\n", 2) == "took 10:00:00 hours\n"


# ------------------------------------------------------------------ log, level 3


def test_low_level_lines_are_dropped_with_a_marker() -> None:
    text = "DEBUG loading a very long configuration file\nDEBUG parsing all of it\nERROR boom\n"
    assert compress_log(text, 3) == "[2 lines dropped]\nERROR boom\n"


def test_a_run_shorter_than_its_marker_is_kept() -> None:
    assert compress_log("INFO a\nERROR b\n", 3) == "INFO a\nERROR b\n"


@pytest.mark.parametrize("level", [1, 2, 3])
def test_log_trace_matches_compress(level: int) -> None:
    doc = Document("10:00:00 DEBUG x\n10:00:01 DEBUG x\nERROR y\n")
    steps = LogCompressor().trace(doc, level)
    assert len(steps) == level
    assert steps[-1].after == LogCompressor().compress(doc, level)


@given(st.lists(st.sampled_from(["a", "b", "", "10:00:00 a", "DEBUG long line here", "x [x2]"])))
@settings(max_examples=200, deadline=None)
def test_log_levels_are_idempotent_and_never_longer(lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    for level in (1, 2, 3):
        once = compress_log(text, level)
        assert len(once) <= len(text)
        assert compress_log(once, level) == once


def test_looks_like_log() -> None:
    log = "".join(f"2026-11-21T10:00:0{i}Z event {i}\n" for i in range(5))
    assert looks_like_log(log + "  traceback line\n")
    assert not looks_like_log("plain\nprose\nhere\n")


# ------------------------------------------------------------------ json


def test_minify_is_exact() -> None:
    text = (
        '{\n  "n": 1.50,\n  "big": 12345678901234567890,\n  "e": 1E3,\n  "d": {"a": 1, "a": 2}\n}\n'
    )
    assert (
        compress_json(text, 1)
        == '{"n":1.50,"big":12345678901234567890,"e":1E3,"d":{"a":1,"a":2}}\n'
    )


def test_unicode_escapes_become_characters() -> None:
    text = '{"name": "' + "\\u0406\\u0440\\u0438\\u043d\\u0430" + '"}'
    assert compress_json(text, 1) == '{"name":"Ірина"}'


def test_control_characters_stay_escaped() -> None:
    text = '["a\\nb", "tab\\t", "q\\"", "back\\\\"]'
    out = compress_json(text, 1)
    assert json.loads(out) == json.loads(text)


def test_long_arrays_and_strings_are_cut_at_level_2() -> None:
    value = {"xs": [f"item number {i}" for i in range(20)], "s": "x" * 400}
    out = json.loads(compress_json(json.dumps(value), 2))
    assert out["xs"][:5] == value["xs"][:5]
    assert out["xs"][5] == "... +15 more items"
    assert out["s"] == "x" * 200 + "... (+200 chars)"


def test_level_3_keeps_the_shape() -> None:
    value = {"users": [{"id": i, "name": f"user {i}"} for i in range(10)]}
    out = json.loads(compress_json(json.dumps(value), 3))
    assert out == {"users": [{"id": 0, "name": "user 0"}, "... +9 more items"]}


def test_cutting_that_would_be_longer_is_skipped() -> None:
    assert compress_json("[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]", 2) == "[1,2,3,4,5,6,7,8,9,10]"


def test_json_lines_are_handled_line_by_line() -> None:
    assert compress_json('{"a": 1}\n\n{"b": [1, 2]}\n', 1) == '{"a":1}\n{"b":[1,2]}\n'


def test_bom_is_kept() -> None:
    assert compress_json(BOM + '{ "a" : 1 }', 1) == BOM + '{"a":1}'


def test_not_json_raises() -> None:
    with pytest.raises(ValueError):
        compress_json("{not json", 1)


@given(
    st.recursive(
        st.none() | st.booleans() | st.integers() | st.text(max_size=80),
        lambda children: (
            st.lists(children, max_size=12)
            | st.dictionaries(st.text(max_size=8), children, max_size=6)
        ),
        max_leaves=40,
    )
)
@settings(max_examples=200, deadline=None)
def test_json_levels_are_valid_idempotent_and_never_longer(value: object) -> None:
    text = json.dumps(value, indent=2)
    assert json.loads(compress_json(text, 1)) == value
    for level in (1, 2, 3):
        once = compress_json(text, level)
        json.loads(once)
        assert len(once) <= len(text)
        assert compress_json(once, level) == once


@pytest.mark.parametrize("level", [1, 2, 3])
def test_json_trace_matches_compress(level: int) -> None:
    doc = Document(json.dumps({"xs": list(range(30))}, indent=1))
    steps = JsonCompressor().trace(doc, level)
    assert steps[-1].after == JsonCompressor().compress(doc, level)


def test_looks_like_json() -> None:
    assert looks_like_json('  {"a": 1}')
    assert looks_like_json("[1]\n[2]\n")
    assert not looks_like_json("{not json")
    assert not looks_like_json("plain text")


# ------------------------------------------------------------------ auto routing


@pytest.mark.parametrize(
    ("doc", "mode"),
    [
        (Document("x", path="a.py"), "code"),
        (Document("x", path="data/a.JSON"), "json"),
        (Document("x", path="rows.jsonl"), "json"),
        (Document("x", path="build.log"), "log"),
        (Document("x", path="notes.md"), "text"),
        (Document('{"a": 1}'), "json"),
        (Document("import os\n"), "code"),
        (Document("".join(f"10:00:0{i} step {i}\n" for i in range(5))), "log"),
        (Document("hello world\n"), "text"),
    ],
)
def test_choose_mode(doc: Document, mode: str) -> None:
    assert choose_mode(doc) == mode


def test_auto_creates_each_mode_once_and_on_demand() -> None:
    auto = AutoCompressor()
    auto.compress(Document("a   b\n", path="a.md"), 2)
    auto.compress(Document("c   d\n", path="b.md"), 2)
    assert list(auto._made) == ["text"]


def test_cli_log_and_json_modes(runner: CliRunner) -> None:
    result = runner.invoke(
        cli, ["--mode", "log", "-l", "2"], input="10:00:01 tick tock\n10:00:02 tick tock\n"
    )
    assert result.stdout == "tick tock [x2]\n"
    result = runner.invoke(cli, ["--mode", "json"], input='{ "a" : [ 1 , 2 ] }\n')
    assert result.stdout == '{"a":[1,2]}\n'


def test_cli_json_mode_on_text_fails_open(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--mode", "json"], input="not json\n")
    assert result.exit_code == 0
    assert result.stdout == "not json\n"
    assert "json mode failed" in result.stderr
