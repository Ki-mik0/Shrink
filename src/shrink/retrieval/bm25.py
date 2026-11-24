"""BM25 (Okapi) over small text chunks, with a tokenizer that understands code.

Identifiers are split into words (``getUserName`` -> get, user, name;
``user_id`` -> user, id), so a question in plain words finds the code that
implements it. Any script works (Cyrillic queries match Cyrillic prose). No
stemming and no dependencies: chunks are few and the ranking only has to tell
relevant parts from the rest.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

K1 = 1.5
B = 0.75

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORD = re.compile(r"[^\W_]+")
STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how", "i",
    "in", "is", "it", "of", "on", "or", "the", "this", "to", "what", "when", "where", "which",
    "who", "why", "with", "self", "def", "return", "import", "none", "true", "false",
})  # fmt: skip


def tokenize(text: str) -> list[str]:
    words = _WORD.findall(_CAMEL.sub(" ", text))
    return [w for w in (word.lower() for word in words) if w not in STOP_WORDS]


class BM25:
    def __init__(self, chunks: Sequence[str]) -> None:
        self._docs = [Counter(tokenize(chunk)) for chunk in chunks]
        self._lengths = [sum(doc.values()) for doc in self._docs]
        self._average = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        frequency: Counter[str] = Counter()
        for doc in self._docs:
            frequency.update(doc.keys())
        total = len(self._docs)
        self._idf = {
            term: math.log((total - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = set(tokenize(query))
        result = []
        for doc, length in zip(self._docs, self._lengths, strict=True):
            score = 0.0
            norm = K1 * (1 - B + B * length / self._average) if self._average else K1
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    score += self._idf[term] * tf * (K1 + 1) / (tf + norm)
            result.append(score)
        return result
