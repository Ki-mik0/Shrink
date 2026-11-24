"""Translate mode: prose in any language becomes English, with a local model.

Most tokenizers are trained mostly on English, so the same sentence costs more
tokens in other languages (Ukrainian about 1.5-2.4x in ``o200k_base``). This
mode translates prose with NLLB-200 (CTranslate2, CPU, int8) and keeps a
translation only when it is really cheaper and nothing was lost:

* code blocks, inline code, URLs, e-mail addresses, paths and code-like
  identifiers are swapped for markers before translation and put back after;
* a unit (heading, list item, paragraph) is translated only if its language is
  detected (fastText ``lid.176``) as something other than English;
* the translation is used only if every marker came back exactly once, the
  digits are the same, and it has fewer tokens than the original.

Otherwise the original text stays. Code and JSON files are never touched.
The model is fetched once with ``shrink --download translate``; nothing goes
over the network while translating.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from shrink import messages
from shrink.compressors.base import Step
from shrink.compressors.text import fenced_segments
from shrink.document import Document

HELP = """\
Prose in any language becomes English (NLLB-200, local, CPU). Applies to text
and logs; code and JSON files are left alone. The level does not matter.

Each heading, list item and paragraph is translated separately, and only if its
language is not English. Code blocks, inline code, URLs, e-mails, paths and
identifiers are protected. A translation is kept only when all of them came
back, the numbers match and it costs fewer tokens; otherwise the original stays.

Needs `pip install 'shrink[translate]'` and, once, `shrink --download translate`
(about 630 MB). The NLLB weights are licensed CC-BY-NC-4.0 (non-commercial)."""

# fastText language code -> NLLB-200 (FLORES-200) code
NLLB_CODES = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab", "az": "azj_Latn",
    "be": "bel_Cyrl", "bg": "bul_Cyrl", "bn": "ben_Beng", "bs": "bos_Latn",
    "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "el": "ell_Grek", "eo": "epo_Latn", "es": "spa_Latn",
    "et": "est_Latn", "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn",
    "fr": "fra_Latn", "ga": "gle_Latn", "gl": "glg_Latn", "gu": "guj_Gujr",
    "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn",
    "hy": "hye_Armn", "id": "ind_Latn", "is": "isl_Latn", "it": "ita_Latn",
    "ja": "jpn_Jpan", "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr",
    "ko": "kor_Hang", "ky": "kir_Cyrl", "lt": "lit_Latn", "lv": "lvs_Latn",
    "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva",
    "ms": "zsm_Latn", "my": "mya_Mymr", "ne": "npi_Deva", "nl": "nld_Latn",
    "nn": "nno_Latn", "no": "nob_Latn", "pa": "pan_Guru", "pl": "pol_Latn",
    "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "si": "sin_Sinh",
    "sk": "slk_Latn", "sl": "slv_Latn", "sq": "als_Latn", "sr": "srp_Cyrl",
    "sv": "swe_Latn", "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu",
    "tg": "tgk_Cyrl", "th": "tha_Thai", "tl": "tgl_Latn", "tr": "tur_Latn",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn", "vi": "vie_Latn",
    "zh": "zho_Hans",
}  # fmt: skip
TARGET = "eng_Latn"
MIN_LETTERS = 3
LATIN_MIN_CONFIDENCE = 0.7  # Latin-script text is often English mixed with names
MAX_PIECE_CHARS = 400  # NLLB was trained on sentences; long units are split

# Markers the model copies verbatim (measured: "⟦0⟧" came back as "⁇ 0 ⁇",
# "__0__", "<0>", "[0]" survived 5/5); "__0__" never occurs in ordinary prose.
_MARK = "__{}__"
_MARK_RE = re.compile(r"__(\d+)__")
# Characters missing from the NLLB vocabulary come back as the unknown token;
# their ASCII forms survive (measured on the model's sentencepiece vocabulary).
_MODEL_SAFE = str.maketrans({
    chr(0x2014): "-", chr(0x2013): "-", chr(0x2012): "-", chr(0x2212): "-",
    chr(0xAB): '"', chr(0xBB): '"', chr(0x201C): '"', chr(0x201D): '"', chr(0x201E): '"',
    chr(0x2018): "'", chr(0x2019): "'", chr(0x02BC): "'", chr(0xA0): " ",
})  # fmt: skip
_UNKNOWN = chr(0x2047)
_PROTECT = re.compile(
    r"`[^`\n]+`"  # inline code
    r"|\]\([^)\s]*\)"  # markdown link target
    r"|<[^<>\s][^<>]*>"  # html tag or <placeholder>
    r"|https?://\S+|www\.\S+"  # urls
    r"|[\w.+-]+@[\w-]+\.[\w.-]+"  # e-mail
    r"|(?:[A-Za-z]:)?[\w.-]*[/\\][\w./\\-]+"  # paths
    r"|--?[A-Za-z][\w-]*"  # command-line flags
    r"|\{[^{}\s]*\}|%[sdif]"  # format placeholders
    r"|\b\w*[_.]\w+\b"  # snake_case, dotted.names, file.ext
    r"|\b[a-z]+[A-Z]\w*\b|\b[A-Z]\w*[A-Z]\w*\b"  # camelCase, CamelCase, ACRONYMS
    r"|\b\w+\(\)"  # calls
)
_PREFIX = re.compile(r"^(\s{0,3}(?:#{1,6}\s+|>\s?)*(?:[-*+]\s+(?:\[[ xX]\]\s+)?|\d+[.)]\s+)?)")
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")
_DIGITS = re.compile(r"\d")
_LETTER = re.compile(r"[^\W\d_]")


class Backend(Protocol):
    def detect(self, text: str) -> tuple[str, float]:
        """Language code (fastText) and confidence."""

    def translate(self, texts: list[str], sources: list[str]) -> list[str]:
        """Translate ``texts`` (NLLB source codes in ``sources``) to English."""


class TranslatorUnavailableError(RuntimeError):
    pass


class NllbBackend:
    """NLLB-200 through CTranslate2, fastText ``lid.176`` for language detection."""

    def __init__(self, folder: Path) -> None:
        try:
            import ctranslate2
            import fasttext
            import sentencepiece
        except ImportError as exc:
            raise TranslatorUnavailableError(
                "translation needs: pip install 'shrink[translate]'"
            ) from exc
        from shrink.models import NLLB_DIR, translate_ready

        if not translate_ready():
            raise TranslatorUnavailableError(
                "the translation model is not downloaded: shrink --download translate"
            )
        model = folder / NLLB_DIR
        self._translator = ctranslate2.Translator(
            str(model),
            device="cpu",
            compute_type="int8",
            intra_threads=min(os.cpu_count() or 1, 8),
        )
        self._sp = sentencepiece.SentencePieceProcessor(
            model_file=str(model / "sentencepiece.bpe.model")
        )
        self._lid = fasttext.load_model(str(folder / "lid.176.ftz"))

    def detect(self, text: str) -> tuple[str, float]:
        labels, scores = self._lid.predict(text.replace("\n", " "), k=1)
        return str(labels[0]).removeprefix("__label__"), float(scores[0])

    def translate(self, texts: list[str], sources: list[str]) -> list[str]:
        if not texts:
            return []
        batch = [
            [source, *self._sp.encode(text.translate(_MODEL_SAFE), out_type=str), "</s>"]
            for text, source in zip(texts, sources, strict=True)
        ]
        results = self._translator.translate_batch(
            batch,
            target_prefix=[[TARGET]] * len(batch),
            beam_size=1,  # greedy: deterministic, fastest
            max_decoding_length=512,
        )
        decoded = [self._sp.decode(result.hypotheses[0][1:]) for result in results]
        # "⁇" is the model's unknown token: something was lost, so give up on that piece.
        return ["" if _UNKNOWN in text else text for text in decoded]


def load_backend() -> Backend:
    from shrink.models import translate_dir

    return NllbBackend(translate_dir())


# ------------------------------------------------------------------ the mode


@dataclass
class _Unit:
    """A heading, list item or paragraph: its lines and what goes to the model."""

    lines: list[str]
    prefix: str
    body: str
    marks: list[str]


class TranslateCompressor:
    name = "translate"
    summary = "prose in any language becomes English (local NLLB model)"
    help = HELP

    def __init__(
        self,
        backend: Backend | None = None,
        *,
        only: frozenset[str] | None = None,
        count: Callable[[str], int] | None = None,
        warn: Callable[[str], None] = messages.warn,
    ) -> None:
        self._backend = backend
        self._only = only
        self._count = count
        self._warn = warn
        self._unavailable = False

    def compress(self, doc: Document, level: int) -> Document:
        if not _applies(doc):
            return doc
        backend = self._get_backend()
        if backend is None:
            return doc
        return doc.with_text(translate_text(doc.text, backend, self._counter(), only=self._only))

    def trace(self, doc: Document, level: int) -> list[Step]:
        return [Step(self.name, self.compress(doc, level))]

    def _get_backend(self) -> Backend | None:
        if self._backend is None and not self._unavailable:
            try:
                self._backend = load_backend()
            except TranslatorUnavailableError as exc:
                self._unavailable = True
                self._warn(f"translate: {exc}; text left as it is")
        return self._backend

    def _counter(self) -> Callable[[str], int]:
        if self._count is None:
            from shrink.observability.stats import make_counter

            self._count = make_counter().count
        return self._count


def _applies(doc: Document) -> bool:
    from shrink.compressors.auto import choose_mode

    return choose_mode(doc) in ("text", "log")


def translate_text(
    text: str,
    backend: Backend,
    count: Callable[[str], int],
    *,
    only: frozenset[str] | None = None,
) -> str:
    if not text.strip():
        return text
    pure_crlf = "\r\n" in text and "\n" not in text.replace("\r\n", "")
    lines = text.replace("\r\n", "\n").split("\n")
    out_parts: list[list[str] | _Unit] = []
    for is_code, segment in fenced_segments(lines):
        if is_code:
            out_parts.append(segment)
        else:
            out_parts.extend(_units(segment))
    units = [part for part in out_parts if isinstance(part, _Unit)]
    _translate_units(units, backend, count, only)
    result: list[str] = []
    for part in out_parts:
        result.extend(part.lines if isinstance(part, _Unit) else part)
    joined = "\n".join(result)
    return joined.replace("\n", "\r\n") if pure_crlf else joined


def _units(lines: list[str]) -> list[list[str] | _Unit]:
    """Group prose lines into headings, list items and paragraphs."""
    out: list[list[str] | _Unit] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            out.append(_make_unit(current.copy()))
            current.clear()

    for line in lines:
        if not line.strip() or _is_literal(line):
            flush()
            out.append([line])
            continue
        prefix = _PREFIX.match(line)
        starts_item = prefix is not None and prefix.group(1).strip() != ""
        if starts_item or line.lstrip().startswith("#"):
            flush()
        current.append(line)
        if line.lstrip().startswith("#"):
            flush()  # a heading is always a unit of its own
    flush()
    return out


def _is_literal(line: str) -> bool:
    """Lines that are not running prose: tables, indented code, html, rules."""
    stripped = line.strip()
    return (
        line.startswith(("    ", "\t"))
        or stripped.startswith(("|", "<"))
        or set(stripped) <= set("-=*_ ")
    )


def _make_unit(lines: list[str]) -> _Unit:
    match = _PREFIX.match(lines[0])
    prefix = match.group(1) if match else ""
    body = " ".join([lines[0][len(prefix) :].strip(), *(line.strip() for line in lines[1:])])
    marks: list[str] = []

    def protect(found: re.Match[str]) -> str:
        marks.append(found.group(0))
        return _MARK.format(len(marks) - 1)

    return _Unit(lines, prefix, _PROTECT.sub(protect, body), marks)


def _translate_units(
    units: list[_Unit],
    backend: Backend,
    count: Callable[[str], int],
    only: frozenset[str] | None,
) -> None:
    pieces: list[str] = []
    sources: list[str] = []
    owners: list[tuple[_Unit, int]] = []  # unit and how many pieces it has
    for unit in units:
        source = _source_language(unit.body, backend, only)
        if source is None:
            continue
        parts = _split_long(unit.body)
        pieces.extend(parts)
        sources.extend([source] * len(parts))
        owners.append((unit, len(parts)))
    translated = backend.translate(pieces, sources)
    position = 0
    for unit, size in owners:
        english = " ".join(t.strip() for t in translated[position : position + size])
        position += size
        restored = _restore(english, unit)
        if restored is None:
            continue
        candidate = unit.prefix + restored
        original = "\n".join(unit.lines)
        if count(candidate) < count(original):
            unit.lines = [candidate]


def _source_language(body: str, backend: Backend, only: frozenset[str] | None) -> str | None:
    plain = _MARK_RE.sub(" ", body)
    letters = _LETTER.findall(plain)
    if len(letters) < MIN_LETTERS:
        return None
    language, confidence = backend.detect(plain)
    if language == "en" or language not in NLLB_CODES:
        return None
    if only is not None and language not in only:
        return None
    latin = sum(1 for ch in letters if ch.isascii())
    if latin * 2 > len(letters) and confidence < LATIN_MIN_CONFIDENCE:
        return None
    return NLLB_CODES[language]


def _split_long(body: str) -> list[str]:
    if len(body) <= MAX_PIECE_CHARS:
        return [body]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE.split(body):
        if current and len(current) + len(sentence) + 1 > MAX_PIECE_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _restore(english: str, unit: _Unit) -> str | None:
    """Put the protected spans back; None if the model lost or duplicated any."""
    found = [int(n) for n in _MARK_RE.findall(english)]
    if sorted(found) != list(range(len(unit.marks))):
        return None
    restored = _MARK_RE.sub(lambda m: unit.marks[int(m.group(1))], english)
    original = _MARK_RE.sub(lambda m: unit.marks[int(m.group(1))], unit.body)
    if Counter(_DIGITS.findall(restored)) != Counter(_DIGITS.findall(original)):
        return None
    return restored.strip() or None
