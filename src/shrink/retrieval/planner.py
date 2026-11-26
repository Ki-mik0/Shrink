"""Decide how hard to compress each document: ``--query`` and ``--budget``.

Both need the whole input before the first document is written (ranking needs
every chunk; a budget needs every size), so with these options the stream is
buffered. Without them shrink keeps streaming one document at a time.

``--query``: every Python function or method and every paragraph of other text
is a chunk, ranked by BM25. Chunks scoring at least half of the best are
relevant. Relevant functions keep their bodies at the skeleton level; other
documents with a relevant paragraph are compressed lightly (level 1); the rest
gets the level asked for.

``--budget N``: while the output is over N tokens, the least relevant document
(the largest one, without a query) is compressed one step harder: its level
goes up, to 3 at most, and only then do its kept functions lose their bodies.
Nothing is cut off: if even that is too much, a warning says how close it got.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from shrink.compressors.base import MAX_LEVEL
from shrink.document import Document
from shrink.retrieval.bm25 import BM25, tokenize

RELEVANT_SHARE = 0.5

Compress = Callable[[Document, int], Document]
Count = Callable[[str], int]


@dataclass
class Plan:
    doc: Document  # the input document, with ``keep`` set for relevant functions
    level: int
    relevance: float  # 0..1, best chunk of this document against the best overall
    output: Document | None = None
    tokens: int = 0


@dataclass(frozen=True)
class _Chunk:
    index: int  # which document
    name: str | None  # function name for Python, None for prose
    text: str


class Planner:
    def __init__(
        self,
        *,
        query: str | None = None,
        budget: int | None = None,
        count: Count | None = None,
        report: Callable[[str], None] | None = None,
        warn: Callable[[str], None] | None = None,
    ) -> None:
        if budget is not None and count is None:
            raise ValueError("a budget needs a token counter")
        self._query = query
        self._budget = budget
        self._count = count
        self._report = report
        self._warn = warn

    def plan(self, docs: list[Document], level: int, compress: Compress) -> list[Plan]:
        plans = [Plan(doc, level, 0.0) for doc in docs]
        if self._query:
            self._apply_query(plans, level)
        if self._budget is not None:
            self._fit_budget(plans, compress)
        else:
            for plan in plans:
                plan.output = compress(plan.doc, plan.level)
        return plans

    # ---------------------------------------------------------------- query

    def _apply_query(self, plans: list[Plan], level: int) -> None:
        assert self._query is not None
        chunks = [chunk for i, plan in enumerate(plans) for chunk in _chunks(i, plan.doc)]
        if not chunks:
            return
        relevance = query_relevance([chunk.text for chunk in chunks], self._query)
        if max(relevance) <= 0:
            self._say(f"query: nothing in {len(plans)} file(s) matches; compressing as usual")
            return
        keep: dict[int, set[str]] = {}
        for chunk, share in zip(chunks, relevance, strict=True):
            plan = plans[chunk.index]
            plan.relevance = max(plan.relevance, share)
            if share >= RELEVANT_SHARE and chunk.name is not None:
                keep.setdefault(chunk.index, set()).add(chunk.name)
        kept: list[str] = []
        for i, plan in enumerate(plans):
            if i in keep:
                plan.doc = replace(plan.doc, keep=frozenset(keep[i]))
                kept.extend(f"{plan.doc.label}:{name}" for name in sorted(keep[i]))
            elif plan.relevance >= RELEVANT_SHARE:
                plan.level = min(level, 1)
                kept.append(plan.doc.label)
        matched = sum(1 for plan in plans if plan.relevance >= RELEVANT_SHARE)
        self._say(f"query: {matched} of {len(plans)} file(s) relevant; kept: {', '.join(kept)}")

    # ---------------------------------------------------------------- budget

    def _fit_budget(self, plans: list[Plan], compress: Compress) -> None:
        assert self._budget is not None
        assert self._count is not None
        for plan in plans:
            self._measure(plan, compress)
        total = sum(plan.tokens for plan in plans)
        steps = 0
        while total > self._budget:
            candidates = [plan for plan in plans if plan.doc.keep or plan.level < MAX_LEVEL]
            if not candidates:
                break
            plan = min(
                candidates,
                key=lambda p: (p.relevance, -p.tokens, plans.index(p)),
            )
            if plan.level < MAX_LEVEL:
                plan.level += 1
            else:  # kept functions only matter at the skeleton level: drop them last
                plan.doc = replace(plan.doc, keep=frozenset())
            before = plan.tokens
            self._measure(plan, compress)
            total += plan.tokens - before
            steps += 1
        if total > self._budget:
            if self._warn is not None:
                self._warn(
                    f"budget of {self._budget} tokens not reached: {total} tokens at the "
                    "strongest compression; nothing was cut off"
                )
        else:
            self._say(f"budget: {total} of {self._budget} tokens after {steps} step(s)")

    def _measure(self, plan: Plan, compress: Compress) -> None:
        assert self._count is not None
        plan.output = compress(plan.doc, plan.level)
        plan.tokens = self._count(plan.output.full_text)

    def _say(self, message: str) -> None:
        if self._report is not None:
            self._report(message)


_QUESTION_END = re.compile(r"(?<=[.?!;])\s+|\n+")  # "a.b" in names has no space after the dot


def query_relevance(chunks: list[str], query: str) -> list[float]:
    """Each chunk's relevance in 0..1: its best score against any question in ``query``.

    A query may hold several questions. Ranked as one bag of words, the
    questions with the most distinctive words take the top scores and push the
    others' chunks under the threshold. So every question is ranked on its own,
    against its own best chunk, and a chunk counts as much as its best match.
    """
    index = BM25(chunks)
    parts = [part for part in _QUESTION_END.split(query) if tokenize(part)] or [query]
    relevance = [0.0] * len(chunks)
    for part in parts:
        scores = index.scores(part)
        best = max(scores, default=0.0)
        if best > 0:
            relevance = [
                max(old, score / best) for old, score in zip(relevance, scores, strict=True)
            ]
    return relevance


def _chunks(index: int, doc: Document) -> list[_Chunk]:
    """Python functions and methods, or paragraphs; the file name helps both."""
    from shrink.compressors.auto import choose_mode

    label = doc.path or ""
    if choose_mode(doc) == "code":
        from shrink.compressors.python_code import functions

        try:
            found = functions(doc.text)
        except (SyntaxError, ValueError):
            found = []
        if found:
            return [_Chunk(index, name, f"{label} {name} {source}") for name, source in found]
    paragraphs = [part for part in doc.text.split("\n\n") if part.strip()]
    return [_Chunk(index, None, f"{label} {part}") for part in paragraphs or [doc.text]]
