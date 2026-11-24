from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from shrink import models
from shrink.cli import cli
from shrink.compressors import translate as translate_module
from shrink.compressors.translate import (
    MAX_PIECE_CHARS,
    TranslateCompressor,
    TranslatorUnavailableError,
    translate_text,
)
from shrink.document import Document

CYRILLIC = re.compile(r"[А-Яа-яІіЇїЄєҐґ'’]+")


class FakeBackend:
    """Cyrillic words become "w"; detection by script, with a few overrides."""

    def __init__(self, *, drop_marks: bool = False, change_digits: bool = False) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []
        self.drop_marks = drop_marks
        self.change_digits = change_digits

    def detect(self, text: str) -> tuple[str, float]:
        if "Guten" in text:
            return "de", 0.95
        if "Halbe" in text:
            return "de", 0.4
        if "Привет" in text:
            return "ru", 0.97
        if CYRILLIC.search(text):
            return "uk", 0.98
        return "en", 0.9

    def translate(self, texts: list[str], sources: list[str]) -> list[str]:
        self.calls.append((texts, sources))
        out = []
        for text in texts:
            english = CYRILLIC.sub("w", text).replace("Guten Tag", "Hi")
            if self.drop_marks:
                english = re.sub(r"__\d+__", "", english)
            if self.change_digits:
                english = english.replace("15", "16")
            out.append(english)
        return out


def run(text: str, backend: FakeBackend | None = None, **kwargs: object) -> str:
    return translate_text(text, backend or FakeBackend(), len, **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------ what is translated


def test_prose_is_translated_and_protected_spans_come_back() -> None:
    text = "Запустіть `shrink --stats` і відкрийте src/app.py або https://example.com/docs.\n"
    assert run(text) == "w `shrink --stats` w w src/app.py w https://example.com/docs.\n"


def test_english_is_not_sent_to_the_model() -> None:
    backend = FakeBackend()
    text = "Already English text here.\n\nУкраїнський текст тут.\n"
    out = run(text, backend)
    assert out == "Already English text here.\n\nw w w.\n"
    assert backend.calls == [(["Український текст тут."], ["ukr_Cyrl"])]


def test_fenced_code_tables_indented_code_and_html_are_left_alone() -> None:
    text = (
        "```python\n# коментар\nprint('Привіт')\n```\n\n"
        "| Колонка | Значення |\n|---|---|\n\n"
        "    відступ коду\n\n"
        "<div>Розмітка</div>\n"
    )
    assert run(text) == text


def test_headings_list_items_and_quotes_keep_their_prefix() -> None:
    text = "## Заголовок\n\n- пункт перший\n- [ ] завдання\n1. крок\n> цитата\n"
    assert run(text) == "## w\n\n- w w\n- [ ] w\n1. w\n> w\n"


def test_wrapped_paragraph_becomes_one_line() -> None:
    assert run("Перший рядок абзацу\nпродовження абзацу.\n") == "w w w w w.\n"


def test_crlf_is_kept() -> None:
    assert run("Текст тут.\r\n\r\nЩе текст.\r\n") == "w w.\r\n\r\nw w.\r\n"


def test_identifiers_and_flags_are_protected() -> None:
    backend = FakeBackend()
    run("Змінна user_id і функція getName() та прапорець --debug.\n", backend)
    sent = backend.calls[0][0][0]
    assert "user_id" not in sent
    assert "getName()" not in sent
    assert "--debug" not in sent
    assert sent.count("__") >= 6


# ------------------------------------------------------------------ safety rules


def test_lost_marker_keeps_the_original() -> None:
    text = "Запустіть `make` зараз.\n"
    assert run(text, FakeBackend(drop_marks=True)) == text


def test_changed_numbers_keep_the_original() -> None:
    text = "Програма завершується за 15 мілісекунд.\n"
    assert run(text, FakeBackend(change_digits=True)) == text


def test_translation_that_costs_more_tokens_is_rejected() -> None:
    text = "Текст.\n"
    longer = translate_text(text, FakeBackend(), lambda s: -len(s))
    assert longer == text


def test_only_listed_languages_are_translated() -> None:
    text = "Привет мир.\n\nПривіт світ.\n"
    assert run(text, only=frozenset({"uk"})) == "Привет мир.\n\nw w.\n"


def test_latin_script_needs_a_confident_detection() -> None:
    assert run("Guten Tag, Freunde.\n") == "Hi, Freunde.\n"
    assert run("Halbe Sachen hier.\n") == "Halbe Sachen hier.\n"


def test_too_few_letters_are_skipped() -> None:
    backend = FakeBackend()
    assert run("Ок\n", backend) == "Ок\n"
    assert backend.calls == [([], [])]


def test_long_paragraphs_are_split_into_pieces() -> None:
    backend = FakeBackend()
    sentence = "Це досить довге речення для перевірки розбиття. "
    run(sentence * 20 + "\n", backend)
    pieces = backend.calls[0][0]
    assert len(pieces) > 1
    assert all(len(piece) <= MAX_PIECE_CHARS for piece in pieces)


def test_blank_text_passes_through() -> None:
    assert run("  \n") == "  \n"


# ------------------------------------------------------------------ the compressor


def test_code_and_json_documents_are_never_translated() -> None:
    backend = FakeBackend()
    compressor = TranslateCompressor(backend, count=len)
    for path in ("a.py", "data.json"):
        doc = Document("# Коментар\nx = 1\n", path=path)
        assert compressor.compress(doc, 2) == doc
    assert backend.calls == []


def test_trace_is_one_step() -> None:
    compressor = TranslateCompressor(FakeBackend(), count=len)
    steps = compressor.trace(Document("Текст тут.\n", path="a.md"), 1)
    assert [s.name for s in steps] == ["translate"]
    assert steps[0].after.text == "w w.\n"


def test_missing_model_warns_once_and_leaves_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> None:
        raise TranslatorUnavailableError("not downloaded")

    monkeypatch.setattr(translate_module, "load_backend", unavailable)
    warnings: list[str] = []
    compressor = TranslateCompressor(count=len, warn=warnings.append)
    doc = Document("Текст.\n")
    assert compressor.compress(doc, 1) == doc
    assert compressor.compress(doc, 1) == doc
    assert warnings == ["translate: not downloaded; text left as it is"]


def test_backend_without_packages_explains(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    with pytest.raises(TranslatorUnavailableError, match=r"shrink\[translate\]"):
        translate_module.NllbBackend(Path("."))


def test_backend_without_model_explains(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("ctranslate2")
    with pytest.raises(TranslatorUnavailableError, match="--download translate"):
        translate_module.load_backend()


def test_default_counter_is_made_lazily() -> None:
    compressor = TranslateCompressor(FakeBackend())
    out = compressor.compress(Document("Український текст тут.\n"), 1)
    assert out.text == "w w w.\n"


# ------------------------------------------------------------------ downloads


def remote_for(path: Path, *, sha: bool = True) -> models.RemoteFile:
    data = path.read_bytes()
    return models.RemoteFile(
        path.as_uri(),
        f"sub/{path.name}",
        len(data),
        hashlib.sha256(data).hexdigest() if sha else None,
    )


def test_fetch_verifies_and_skips_a_verified_copy(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"x" * 1000)
    messages: list[str] = []
    target = models.fetch(remote_for(source), tmp_path / "out", messages.append)
    assert target.read_bytes() == b"x" * 1000
    models.fetch(remote_for(source), tmp_path / "out", messages.append)
    assert messages[-1] == "sub/model.bin: already downloaded"


def test_fetch_rejects_a_wrong_checksum(tmp_path: Path) -> None:
    source = tmp_path / "model.bin"
    source.write_bytes(b"abc")
    bad = models.RemoteFile(source.as_uri(), "model.bin", 3, "0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        models.fetch(bad, tmp_path / "out", lambda _: None)
    assert not (tmp_path / "out" / "model.bin").exists()
    assert not (tmp_path / "out" / "model.bin.part").exists()


def test_download_translate_marks_the_folder_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "lid.ftz"
    source.write_bytes(b"lid")
    monkeypatch.setattr(models, "TRANSLATE_FILES", (remote_for(source, sha=False),))
    assert not models.translate_ready()
    folder = models.download_translate(lambda _: None)
    assert models.translate_ready()
    assert (folder / "sub" / "lid.ftz").read_bytes() == b"lid"


def test_pinned_files_are_complete() -> None:
    names = {remote.name for remote in models.TRANSLATE_FILES}
    assert f"{models.NLLB_DIR}/model.bin" in names
    assert "lid.176.ftz" in names
    assert all(
        models.NLLB_REVISION in r.url for r in models.TRANSLATE_FILES if "huggingface" in r.url
    )


# ------------------------------------------------------------------ CLI


def test_cli_translate_flag(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(translate_module, "load_backend", FakeBackend)
    result = runner.invoke(cli, ["--translate", "-l", "1"], input="Привіт світ.\n")
    assert result.exit_code == 0
    assert result.stdout == "w w.\n"


def test_cli_translate_from(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(translate_module, "load_backend", FakeBackend)
    result = runner.invoke(
        cli, ["--translate", "--translate-from", "ru", "-l", "1"], input="Привіт світ.\n"
    )
    assert result.stdout == "Привіт світ.\n"


def test_cli_translate_step_in_stats(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(translate_module, "load_backend", FakeBackend)
    result = runner.invoke(cli, ["--translate", "--stats", "-l", "1"], input="Привіт світ.\n")
    assert "shrink:   translate" in result.stderr


def test_cli_download_translate(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "download_translate", lambda: Path("/models"))
    result = runner.invoke(cli, ["--download", "translate"])
    assert result.exit_code == 0
    assert "translate ready in" in result.stderr


# ------------------------------------------------------------------ the real model (opt-in)


def test_real_model_translates_ukrainian(
    monkeypatch: pytest.MonkeyPatch, real_cache_dir: Path
) -> None:
    pytest.importorskip("ctranslate2")
    if not (real_cache_dir / "translate" / models.READY).is_file():
        pytest.skip("translation model not downloaded (shrink --download translate)")
    monkeypatch.setenv("SHRINK_CACHE_DIR", str(real_cache_dir))
    text = "Запустіть `shrink --stats`, щоб побачити 15 зекономлених токенів.\n"
    out = translate_text(text, translate_module.load_backend(), len)
    assert "`shrink --stats`" in out
    assert "15" in out
    assert not CYRILLIC.search(out)


def test_a_piece_the_model_gave_up_on_keeps_the_original() -> None:
    class GivesUp(FakeBackend):
        def translate(self, texts: list[str], sources: list[str]) -> list[str]:
            return ["" for _ in texts]

    text = "Текст тут.\n"
    assert run(text, GivesUp()) == text


def test_real_model_handles_dashes_and_guillemets(
    monkeypatch: pytest.MonkeyPatch, real_cache_dir: Path
) -> None:
    pytest.importorskip("ctranslate2")
    if not (real_cache_dir / "translate" / models.READY).is_file():
        pytest.skip("translation model not downloaded (shrink --download translate)")
    monkeypatch.setenv("SHRINK_CACHE_DIR", str(real_cache_dir))
    backend = translate_module.load_backend()
    out = backend.translate(["Етапи 2–4 — це «ядро» проєкту."], ["ukr_Cyrl"])
    assert out[0]
    assert chr(0x2047) not in out[0]
