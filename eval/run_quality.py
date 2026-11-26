"""Quality eval: does a model answer as well from shrink's output as from the original?

For each corpus the model gets the files (in files-to-prompt format) and all
the corpus's questions at once, and answers in JSON. The context comes in five
variants: original, shrink code level 1, 2, 3, and level 3 with ``--query``
(the questions as the query, so functions that matter keep their bodies).
Answers are scored by keywords from ``questions.json`` (no LLM judge: free and
reproducible). Each variant runs several times because answers vary.

Designed for OpenRouter's free models (20 requests a minute, 50 a day without
credits): one request per corpus, variant and run - 3 x 5 x 3 = 45 by default.
Every answer is saved, so a run that hits the daily limit continues where it
stopped the next day.

    set OPENROUTER_API_KEY=...                 (PowerShell: $env:OPENROUTER_API_KEY="...")
    python eval/run_quality.py --dry-run       # prompts and token counts, no network
    python eval/run_quality.py                 # run (default model: a free one)
    python eval/run_quality.py --report        # only rebuild the table from saved answers
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import ROOT, STDLIB, python_version

from shrink.compressors.auto import AutoCompressor
from shrink.document import Document
from shrink.observability.stats import make_counter
from shrink.retrieval.planner import Planner

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "quality" / "questions.json"
RESULTS = HERE / "results"
# Free models are shared: the Gemma ones (served only by Google AI Studio) are often
# "temporarily rate-limited upstream". Any other model works with --model.
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
RETRY_WAITS = [30, 60, 120, 240]  # seconds, for a busy model (HTTP 429/502/503)
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
VARIANTS = ["original", "L1", "L2", "L3", "L3+query"]
PAUSE = 3.5  # seconds between requests: free models allow 20 a minute

SYSTEM = (
    "You answer questions about the source code you are given, using only that code. "
    "Reply with a single JSON object mapping each question number to a short answer, "
    'e.g. {"1": "70"}. Give values, names or a few words, no explanations. '
    'If the code does not contain the answer, use "unknown".'
)


class DailyLimitError(RuntimeError):
    pass


class TemporaryLimitError(RuntimeError):
    """The model stayed busy through every retry."""


@dataclass(frozen=True)
class Corpus:
    name: str
    files: list[tuple[str, str]]  # (label, text)
    questions: list[dict[str, Any]]


def load_corpora() -> list[Corpus]:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))["corpora"]
    corpora = []
    for name, spec in data.items():
        paths = [(f"stdlib/{p}", STDLIB / p) for p in spec.get("stdlib", [])]
        paths += [(p, ROOT / p) for p in spec.get("project", [])]
        files = [(label, path.read_text(encoding="utf-8")) for label, path in paths]
        corpora.append(Corpus(name, files, spec["questions"]))
    return corpora


def build_context(corpus: Corpus, variant: str) -> str:
    docs = [Document(text, path=label) for label, text in corpus.files]
    auto = AutoCompressor()
    if variant == "original":
        outputs = docs
    elif variant == "L3+query":
        query = " ".join(q["q"] for q in corpus.questions)
        plans = Planner(query=query).plan(docs, 3, auto.compress)
        outputs = [plan.output or plan.doc for plan in plans]
    else:
        level = int(variant[1])
        outputs = [auto.compress(doc, level) for doc in docs]
    return "".join(f"{doc.path}\n---\n{doc.text.rstrip()}\n---\n" for doc in outputs)


def build_prompt(corpus: Corpus, context: str) -> str:
    numbered = "\n".join(f"{i}. {q['q']}" for i, q in enumerate(corpus.questions, start=1))
    return f"Source files:\n\n{context}\nQuestions:\n{numbered}\n"


def ask(
    model: str,
    prompt: str,
    key: str,
    *,
    system: str = SYSTEM,
    max_tokens: int | None = None,
    on_chunk: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    request_body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        # Streamed, so a long answer arrives in pieces instead of one response a
        # gateway may cut off; reasoning models think briefly and keep it to themselves.
        "stream": True,
        "reasoning": {"effort": "low", "exclude": True},
    }
    if max_tokens is not None:
        request_body["max_tokens"] = max_tokens
    body = json.dumps(request_body).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": "shrink eval",
        },
    )
    for attempt, wait in enumerate([*RETRY_WAITS, None]):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return read_stream(response, on_chunk)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            lowered = detail.lower()
            if exc.code == 429 and any(w in lowered for w in ("per-day", "per day", "daily")):
                raise DailyLimitError(detail[:300]) from exc
            if exc.code not in (429, 502, 503):
                raise RuntimeError(f"HTTP {exc.code}: {detail[:300]}") from exc
            problem = f"busy (HTTP {exc.code})"
        except (TemporaryLimitError, http.client.HTTPException, OSError) as exc:
            # dropped connection, timeout, provider error in the middle of the stream
            problem = f"connection problem ({type(exc).__name__}: {str(exc)[:80]})"
        if wait is None:  # still failing: stop cleanly, the next run continues
            raise TemporaryLimitError(problem)
        print(f"  {problem}, waiting {wait} s (attempt {attempt + 1})")
        time.sleep(wait)
    raise AssertionError("unreachable")


def read_stream(
    response: Iterable[bytes], on_chunk: Callable[[int], None] | None = None
) -> dict[str, Any]:
    """Collect a server-sent-events chat completion into one reply.

    ``on_chunk`` gets the number of characters received so far, after every piece.
    """
    received = 0
    content: list[str] = []
    usage: dict[str, Any] = {}
    finished = False
    reason = ""
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line or line.startswith(":"):  # blank line or keep-alive comment
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            finished = True
            break
        event = json.loads(payload)
        if "error" in event:
            raise TemporaryLimitError(str(event["error"])[:300])
        for choice in event.get("choices") or []:
            piece = (choice.get("delta") or {}).get("content") or ""
            content.append(piece)
            received += len(piece)
            if piece and on_chunk is not None:
                on_chunk(received)
            if choice.get("finish_reason"):
                finished = True
                reason = str(choice["finish_reason"])
        usage = event.get("usage") or usage
    if not finished:
        raise TemporaryLimitError("the stream ended before the answer was complete")
    return {
        "choices": [{"message": {"content": "".join(content)}, "finish_reason": reason}],
        "usage": usage,
    }


def archive(model: str, variant: str) -> None:
    """Move a variant's saved answers aside, so the next run asks again."""
    folder = raw_path(model, "x", "x", 0).parent
    target = folder / "previous" / variant.replace("+", "_")
    moved = 0
    for path in folder.glob(f"*--{variant.replace('+', '_')}--*.json"):
        target.mkdir(parents=True, exist_ok=True)
        path.replace(target / path.name)
        moved += 1
    print(f"{moved} saved answer(s) for {variant} moved to {target}")


def parse_answers(text: str) -> dict[str, str]:
    """The JSON object in the reply, even inside a code fence or prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return {}
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def correct(answer: str, expect: list[list[str]]) -> bool:
    text = answer.lower().replace("`", "").replace('"', "'")
    if text.strip() in ("", "unknown"):
        return False
    return all(any(option.lower() in text for option in group) for group in expect)


def raw_path(model: str, corpus: str, variant: str, run: int) -> Path:
    slug = re.sub(r"[^\w.-]+", "_", model)
    return RESULTS / "raw" / slug / f"{corpus}--{variant.replace('+', '_')}--{run}.json"


def run(model: str, runs: int, dry_run: bool) -> None:
    counter = make_counter()
    if counter.approximate:
        print("note: no exact tokenizer here (shrink --download tokenizer); counts are estimates")
    count = counter.count
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not dry_run and not key:
        sys.exit("set OPENROUTER_API_KEY first (or use --dry-run)")
    calls = 0
    for corpus in load_corpora():
        for variant in VARIANTS:
            prompt = build_prompt(corpus, build_context(corpus, variant))
            if dry_run:
                print(f"{corpus.name:<9} {variant:<9} {count(prompt):>7} tokens")
                continue
            for number in range(1, runs + 1):
                path = raw_path(model, corpus.name, variant, number)
                if path.is_file():
                    continue
                if calls:
                    time.sleep(PAUSE)
                try:
                    reply = ask(model, prompt, key)
                except DailyLimitError as exc:
                    print(f"daily limit reached, run again tomorrow to continue: {exc}")
                    report(model)
                    return
                except TemporaryLimitError as exc:
                    print(f"{model} is still busy upstream after retries: {str(exc)[:160]}")
                    print(
                        "Answers so far are saved. Run the same command later to continue, "
                        "or use another free model: --model inclusionai/ling-3.2-flash:free"
                    )
                    report(model)
                    return
                calls += 1
                path.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "model": model,
                    "corpus": corpus.name,
                    "variant": variant,
                    "run": number,
                    "context_tokens": count(prompt),
                    "usage": reply.get("usage", {}),
                    "content": reply["choices"][0]["message"].get("content") or "",
                }
                path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
                )
                print(f"{corpus.name:<9} {variant:<9} run {number}: saved")
    if dry_run:
        total = len(load_corpora()) * len(VARIANTS) * runs
        print(f"{total} requests for {runs} run(s) per variant")
    else:
        report(model)


def wrong_answers(
    misses: dict[tuple[str, str], list[str]], corpora: dict[str, Corpus]
) -> list[str]:
    if not misses:
        return []
    questions = {q["id"]: q for c in corpora.values() for q in c.questions}
    lines = ["", "## Wrong answers", "", "| Context | Question | Needs | Answers |"]
    lines.append("| --- | --- | --- | --- |")
    order = {v: i for i, v in enumerate(VARIANTS)}
    for (variant, qid), answers in sorted(misses.items(), key=lambda m: (order[m[0][0]], m[0][1])):
        question = questions[qid]
        shown = ", ".join(sorted({a.strip() or "(none)" for a in answers}))[:80]
        lines.append(
            f"| {variant} | {qid}: {question['q']} | {question['needs']} | "
            f"{len(answers)}x: {shown} |"
        )
    return lines


def report(model: str) -> None:
    corpora = {c.name: c for c in load_corpora()}
    rows: dict[str, dict[str, list[float]]] = {v: {} for v in VARIANTS}
    tokens: dict[str, dict[str, int]] = {v: {} for v in VARIANTS}
    by_need: dict[str, dict[str, list[int]]] = {v: {} for v in VARIANTS}
    misses: dict[tuple[str, str], list[str]] = {}
    counter = make_counter()
    folder = raw_path(model, "x", "x", 0).parent
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        corpus = corpora[record["corpus"]]
        answers = parse_answers(record["content"])
        hits = []
        for i, question in enumerate(corpus.questions, start=1):
            answer = answers.get(str(i), "")
            ok = correct(answer, question["expect"])
            hits.append(ok)
            need = "body" if question["needs"] == "body" else "structure"
            by_need[record["variant"]].setdefault(need, []).append(int(ok))
            if not ok:
                misses.setdefault((record["variant"], question["id"]), []).append(answer)
        rows[record["variant"]].setdefault(corpus.name, []).append(sum(hits) / len(hits))
        if corpus.name not in tokens[record["variant"]]:
            # Counted now, exactly, rather than trusting the count saved with the answer:
            # a run without the tokenizer falls back to the chars / 4 estimate.
            prompt = build_prompt(corpus, build_context(corpus, record["variant"]))
            tokens[record["variant"]][corpus.name] = counter.count(prompt)
    names = list(corpora)
    lines = [
        "# Quality eval",
        "",
        f"Model `{model}` via OpenRouter, {python_version()}; prompt tokens counted with "
        f"{counter.name}{' (estimate)' if counter.approximate else ''}.",
        "Accuracy = share of questions answered correctly (keyword check), mean over runs.",
        "",
        "| Context | Tokens | "
        + " | ".join(names)
        + " | All | Structure questions | Body questions |",
        "| --- | ---: |" + " ---: |" * (len(names) + 3),
    ]
    incomplete: list[str] = []
    for variant in VARIANTS:
        if not rows[variant]:
            continue
        cells = [
            f"{mean(rows[variant][n]) * 100:.0f}%" if n in rows[variant] else "—" for n in names
        ]
        every = [score for n in names for score in rows[variant].get(n, [])]
        need = by_need[variant]
        structure = mean(need["structure"]) * 100 if need.get("structure") else 0
        body = mean(need["body"]) * 100 if need.get("body") else 0
        partial = len(tokens[variant]) < len(names)
        total = f"{sum(tokens[variant].values())}{' (partial)' if partial else ''}"
        lines.append(
            f"| {variant} | {total} | "
            + " | ".join(cells)
            + f" | {mean(every) * 100:.0f}% | {structure:.0f}% | {body:.0f}% |"
        )
        if partial:
            incomplete.append(variant)
    if incomplete:
        lines += [
            "",
            f"Not finished yet: {', '.join(incomplete)} (some corpora have no answers;"
            " run the same command again to continue).",
        ]
    lines += wrong_answers(misses, corpora)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = "\n".join(lines) + "\n"
    (RESULTS / "quality.md").write_text(out, encoding="utf-8", newline="\n")
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="no network: show prompt sizes")
    parser.add_argument("--report", action="store_true", help="only rebuild the table")
    parser.add_argument(
        "--redo",
        choices=VARIANTS,
        help="ask again for one context variant (saved answers move to raw/.../previous/)",
    )
    args = parser.parse_args()
    try:
        if args.redo:
            archive(args.model, args.redo)
        if args.report:
            report(args.model)
        else:
            run(args.model, args.runs, args.dry_run)
    except KeyboardInterrupt:
        sys.exit("stopped; answers saved so far are kept, run again to continue")
    except RuntimeError as exc:  # bad key, unknown model...: one line, no traceback
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
