"""Generation eval: how many tokens shrink saves on a large, realistic task.

The model gets a small but realistic FastAPI + JavaScript project
(``eval/generation/starter``, about 11k tokens) and a task that needs a long
answer: user accounts and personal todo lists from the database migration to
the frontend pages (``eval/generation/task.md``). The project is sent in five
variants, as in the quality eval: original, shrink level 1, 2, 3 and level 3
with ``--query`` (the task text as the query), plus level 3 with ``--brief``
(a note asking for a terse answer, which targets the output tokens), level 3
with ``--no-docs --brief`` (no docstrings or JSDoc left in the context either)
and level 3 with ``--brief`` plus a note after the task asking for SEARCH/REPLACE
edits instead of whole existing files (``+edits``), and the two notes that
``shrink apply`` undoes: ``+lazy`` (existing files come back shortened, with
stub bodies and ``...`` gaps) and ``+dedent`` (the context and the answer in a
compact layout: Python with one space per level, JS/HTML/CSS unindented).

For every answer it records input tokens (exact o200k_base and what the
provider counted), output tokens, whether the answer was cut off, the time it
took, and a simple completeness check that does not run anything: which of
the required files and endpoints are present and whether the Python parses.
With ``+edits`` the model gives changes to existing files as SEARCH/REPLACE
edits; they are applied to the starter project's real files (as a tool would)
and the checks run on the result; edits that do not apply are counted. With
``+lazy`` and ``+dedent`` every file of the answer goes through
:func:`shrink.restore.restore`, exactly as ``shrink apply`` would write it, and
the checks run on the restored files; stubs left with nothing to fill them in
are counted (an unfinished answer, not a saving).
Every answer stores a hash of its prompt: when shrink changes and a variant's
prompt is no longer the same, the next run moves that answer aside and asks again.

    python eval/run_generation.py --dry-run   # prompt sizes only, no network
    python eval/run_generation.py             # one request per variant
    python eval/run_generation.py --runs 3    # more runs to average the output
    python eval/run_generation.py --runs 3 --variants original,L3,L3+brief
    python eval/run_generation.py --runs 3 --variants L3+brief+lazy,L3+dedent+brief+lazy
    python eval/run_generation.py --report    # rebuild the table from saved answers
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_quality as rq
from corpus import python_version

from shrink.brief import BRIEF_NOTE, DEDENT_NOTE, EDITS_NOTE, LAZY_NOTE
from shrink.compressors.auto import AutoCompressor
from shrink.compressors.chain import ChainCompressor
from shrink.compressors.dedent import DedentCompressor
from shrink.compressors.nodocs import NoDocsCompressor
from shrink.document import Document
from shrink.edits import EditError, apply_edits, split_edits
from shrink.observability.stats import make_counter
from shrink.restore import RestoreError, restore
from shrink.splitting.formats import FORMATS
from shrink.splitting.framed import FramedSplitter

HERE = Path(__file__).resolve().parent
STARTER = HERE / "generation" / "starter"
TASK = HERE / "generation" / "task.md"
RESULTS = HERE / "results"
MAX_TOKENS = 16000
ALLOWED_ROOTS = ("backend/", "frontend/")
DEFAULT_REASONING = "low"  # what the earlier answers were asked with
REASONING_EFFORTS = ["none", "minimal", "low", "medium", "high"]
VARIANTS = [
    *rq.VARIANTS,
    "L3+brief",
    "L3+no-docs+brief",
    "L3+brief+edits",
    "L3+brief+lazy",
    "L3+dedent+brief+lazy",
]
_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)

SYSTEM = (
    "You are a senior full-stack engineer working in an existing repository. "
    "Read the repository files you are given, follow their conventions, reuse their helpers "
    "and implement the task completely. Answer only with the files, in the requested format."
)

# What a complete answer contains: (description, check on {path: content}).
CHECKS: list[tuple[str, Any]] = [
    ("auth router file", lambda f: _find(f, "backend/app/routers/auth.py")),
    ("todos router file", lambda f: _find(f, "backend/app/routers/todos.py")),
    ("todos migration", lambda f: re.search(r"create\s+table\s+todos", _all(f), re.I)),
    ("routers registered", lambda f: "auth.router" in _all(f) and "todos.router" in _all(f)),
    (
        "register/login/me",
        lambda f: all(r in _get(f, "auth.py") for r in ("/register", "/login", "/me")),
    ),
    (
        "password helpers used",
        lambda f: "hash_password" in _all(f) and "verify_password" in _all(f),
    ),
    (
        "todo PATCH and DELETE",
        lambda f: "patch" in _get(f, "todos.py") and "delete" in _get(f, "todos.py"),
    ),
    ("todos are per user", lambda f: "get_current_user" in _get(f, "todos.py")),
    ("paginated list", lambda f: "paginate" in _get(f, "todos.py")),
    ("login page", lambda f: _find(f, "frontend/login.html") and _find(f, "frontend/js/login.js")),
    ("todos page", lambda f: _find(f, "frontend/todos.html") and _find(f, "frontend/js/todos.js")),
    ("frontend uses apiFetch", lambda f: "apifetch" in _get(f, "todos.js")),  # _get lowercases
]


def _find(files: dict[str, str], path: str) -> bool:
    return path in files


def _all(files: dict[str, str]) -> str:
    return "\n".join(files.values())


def _get(files: dict[str, str], name: str) -> str:
    return "\n".join(text for path, text in files.items() if path.endswith(name)).lower()


@dataclass(frozen=True)
class Answer:
    files: dict[str, str]
    rejected: list[str]  # paths outside the repository or otherwise unsafe
    edits: int = 0  # SEARCH/REPLACE edits in the answer (+edits)
    edits_failed: int = 0
    restored: int = 0  # files where shrink apply filled something in (+lazy)
    restore_failed: int = 0  # files shrink apply could not merge
    stubs_left: int = 0  # function bodies left as `...` with nothing to fill them in


def starter_files() -> list[tuple[str, str]]:
    return [
        (path.relative_to(STARTER).as_posix(), path.read_text(encoding="utf-8"))
        for path in sorted(STARTER.rglob("*"))
        if path.is_file()
    ]


def task_text() -> str:
    return TASK.read_text(encoding="utf-8")


def build_context(variant: str) -> str:
    """The repository as files-to-prompt text, compressed like the quality eval;
    ``+brief`` adds the note ``shrink --brief`` writes after its output."""
    corpus = rq.Corpus("taskboard", starter_files(), [{"q": task_text()}])
    # The edits and lazy notes go after the task.
    variant = variant.removesuffix("+edits").removesuffix("+lazy")
    note = BRIEF_NOTE if variant.endswith("+brief") else ""
    base = variant.removesuffix("+brief")
    if base.endswith("+no-docs"):
        return context_with(int(base[1]), NoDocsCompressor()) + note
    if base.endswith("+dedent"):
        return context_with(int(base[1]), DedentCompressor()) + note
    return rq.build_context(corpus, base) + note


def context_with(level: int, step: Any) -> str:
    """``shrink --level N`` plus one step (``--no-docs``, ``--dedent``) over the
    starter, framed like the other variants."""
    compressor = ChainCompressor([AutoCompressor(), step])
    docs = [
        compressor.compress(Document(text, path=label), level) for label, text in starter_files()
    ]
    return "".join(f"{doc.path}\n---\n{doc.text.rstrip()}\n---\n" for doc in docs)


def build_prompt(context: str) -> str:
    return f"Repository files:\n\n{context}\nTask:\n\n{task_text()}"


def prompt_for(variant: str) -> str:
    """``+edits``/``+lazy``: their note after the task (which asks for complete
    files), where an agent proxy would add it; ``+dedent`` asks for the compact
    layout there too."""
    prompt = build_prompt(build_context(variant))
    if variant.endswith("+edits"):
        return prompt + EDITS_NOTE
    if variant.endswith("+lazy"):
        return prompt + (DEDENT_NOTE if "+dedent" in variant else "") + LAZY_NOTE
    return prompt


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def is_current(path: Path, prompt: str, reasoning: str = DEFAULT_REASONING) -> bool:
    """Was the saved answer given for exactly this prompt and reasoning setting?"""
    record = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        record.get("prompt_sha256") == prompt_hash(prompt)
        and record.get("reasoning", DEFAULT_REASONING) == reasoning
    )


def safe_path(path: str) -> bool:
    """Only relative paths inside backend/ or frontend/, no '..' and no drive letters."""
    normal = path.strip().replace("\\", "/")
    return (
        bool(normal)
        and not normal.startswith("/")
        and ":" not in normal
        and ".." not in normal.split("/")
        and normal.startswith(ALLOWED_ROOTS)
    )


def parse_answer(text: str, restore_files: bool = False) -> Answer:
    """Split '## File: path' + fenced block answers into files; apply '## Edit: path'
    SEARCH/REPLACE blocks to the starter project's real files, as a tool would.
    ``restore_files``: run every file through what ``shrink apply`` does."""
    files: dict[str, str] = {}
    rejected: list[str] = []
    text, file_edits = split_edits(text)
    splitter = FramedSplitter(FORMATS["repomix-markdown"])
    for doc in splitter.split(text.splitlines(keepends=True)):
        if doc.path is None:
            continue
        path = doc.path.strip().strip("`").replace("\\", "/")
        if safe_path(path):
            files[path] = doc.text
        else:
            rejected.append(path)
    starter = dict(starter_files())
    edits = failed = 0
    for item in file_edits:
        path = item.path.replace("\\", "/")
        edits += len(item.edits)
        base = files.get(path, starter.get(path))
        whole = _FENCE.search(item.body)
        if not item.edits and whole is not None and safe_path(path):
            files[path] = whole.group(1)  # an "Edit" section holding the whole file
            continue
        if not safe_path(path) or base is None or not item.edits:
            failed += max(len(item.edits), 1)
            continue
        try:
            files[path] = apply_edits(base, item.edits)
        except EditError:
            failed += len(item.edits)
    if not restore_files:
        return Answer(files, rejected, edits, failed)
    restored = restore_failed = stubs = 0
    for path, content in list(files.items()):
        try:
            result = restore(starter.get(path), content, path)
        except RestoreError:
            restore_failed += 1
            continue
        files[path] = result.text
        restored += result.report.restored
        stubs += len(result.report.unresolved)
    return Answer(files, rejected, edits, failed, restored, restore_failed, stubs)


def completeness(answer: Answer) -> dict[str, Any]:
    passed = [name for name, check in CHECKS if check(answer.files)]
    python = [text for path, text in answer.files.items() if path.endswith(".py")]
    parsed = 0
    for source in python:
        try:
            ast.parse(source)
            parsed += 1
        except SyntaxError:
            pass
    return {
        "files": len(answer.files),
        "rejected": answer.rejected,
        "checks_passed": passed,
        "checks_total": len(CHECKS),
        "python_files": len(python),
        "python_valid": parsed,
        "edits": answer.edits,
        "edits_failed": answer.edits_failed,
        "restored": answer.restored,
        "restore_failed": answer.restore_failed,
        "stubs_left": answer.stubs_left,
    }


def answer_of(variant: str, content: str) -> Answer:
    """The files of an answer as the variant's tooling would write them."""
    return parse_answer(content, restore_files="+lazy" in variant or "+dedent" in variant)


def report_path(model: str) -> Path:
    """``generation.md`` for the default model, ``generation-<model>.md`` for others."""
    if model == rq.DEFAULT_MODEL:
        return RESULTS / "generation.md"
    slug = re.sub(r"[^\w.-]+", "_", model)
    return RESULTS / f"generation-{slug}.md"


def raw_path(model: str, variant: str, run: int) -> Path:
    slug = re.sub(r"[^\w.-]+", "_", model)
    return RESULTS / "raw-generation" / slug / f"{variant.replace('+', '_')}--{run}.json"


def archive(model: str, variant: str) -> None:
    folder = raw_path(model, "x", 0).parent
    moved = 0
    for path in sorted(folder.glob(f"{variant.replace('+', '_')}--*.json")):
        # one folder per prompt, so answers to different prompts never overwrite each other
        sha = json.loads(path.read_text(encoding="utf-8")).get("prompt_sha256") or "unknown"
        target = folder / "previous" / variant.replace("+", "_") / sha
        target.mkdir(parents=True, exist_ok=True)
        name, n = path.name, 1
        while (target / name).exists():
            n += 1
            name = f"{path.stem}-{n}.json"
        path.replace(target / name)
        moved += 1
    target = folder / "previous" / variant.replace("+", "_")
    print(f"{moved} saved answer(s) for {variant} moved to {target}")


class Progress:
    """Show how much of a long answer has arrived, at most once a second."""

    def __init__(self, started: float) -> None:
        self.started = started
        self.shown = 0.0

    def __call__(self, chars: int) -> None:
        now = time.perf_counter()
        if now - self.shown >= 1.0:
            self.shown = now
            elapsed = now - self.started
            print(f"\r  receiving: {chars} characters, {elapsed:.0f} s", end="", flush=True)


def drop_stale(
    model: str, variant: str, runs: int, prompt: str, reasoning: str = DEFAULT_REASONING
) -> None:
    """Move a variant's answers aside if any of them was given for another prompt."""
    saved = [raw_path(model, variant, n) for n in range(1, runs + 1)]
    if any(path.is_file() and not is_current(path, prompt, reasoning) for path in saved):
        print(f"{variant:<9} the prompt or reasoning changed since the saved answer; asking again")
        archive(model, variant)


def run(
    model: str,
    runs: int,
    dry_run: bool,
    only: list[str] | None = None,
    reasoning: str = DEFAULT_REASONING,
) -> None:
    counter = make_counter()
    if counter.approximate:
        print("note: no exact tokenizer here (shrink --download tokenizer); counts are estimates")
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not dry_run and not key:
        sys.exit("set OPENROUTER_API_KEY first (or use --dry-run)")
    original = counter.count(prompt_for("original"))
    calls = 0
    for variant in [v for v in VARIANTS if not only or v in only]:
        prompt = prompt_for(variant)
        tokens = counter.count(prompt)
        if dry_run:
            print(
                f"{variant:<9} {tokens:>7} tokens  ({(tokens - original) / original * 100:+.1f}%)"
            )
            continue
        drop_stale(model, variant, runs, prompt, reasoning)
        for number in range(1, runs + 1):
            path = raw_path(model, variant, number)
            if path.is_file():
                continue
            if calls:
                time.sleep(rq.PAUSE)
            started = time.perf_counter()
            print(f"{variant:<9} run {number}: asking the model...", flush=True)
            try:
                reply = rq.ask(
                    model,
                    prompt,
                    key,
                    system=SYSTEM,
                    max_tokens=MAX_TOKENS,
                    on_chunk=Progress(started),
                    reasoning=reasoning,
                )
            except rq.DailyLimitError as exc:
                print(f"daily limit reached, run again tomorrow to continue: {exc}")
                report(model)
                return
            except rq.TemporaryLimitError as exc:
                print(f"{model} is still busy after retries: {str(exc)[:160]}; run again later")
                report(model)
                return
            calls += 1
            choice = reply["choices"][0]
            record = {
                "model": model,
                "variant": variant,
                "run": number,
                "prompt_sha256": prompt_hash(prompt),
                "reasoning": reasoning,
                "seconds": round(time.perf_counter() - started, 1),
                "finish_reason": choice.get("finish_reason", ""),
                "usage": reply.get("usage", {}),
                "content": choice["message"].get("content") or "",
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
            )
            seconds, reason = record["seconds"], record["finish_reason"]
            print(f"\r{variant:<9} run {number}: saved ({seconds} s, {reason})" + " " * 20)
    if dry_run:
        print(f"{len(only or VARIANTS) * runs} request(s) for {runs} run(s) per variant")
    else:
        report(model)


def percent(new: float, base: float) -> str:
    return f"{(new - base) / base * 100:+.1f}%" if base else "—"


def report(model: str) -> None:
    counter = make_counter()
    folder = raw_path(model, "x", 0).parent
    rows: dict[str, list[dict[str, Any]]] = {v: [] for v in VARIANTS}
    prompts = {v: prompt_for(v) for v in VARIANTS}
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        variant = record.get("variant")
        if variant in rows:
            record["current"] = record.get("prompt_sha256") == prompt_hash(prompts[variant])
            rows[variant].append(record)
    inputs = {v: counter.count(prompts[v]) for v in VARIANTS}
    efforts = sorted(
        {r.get("reasoning", DEFAULT_REASONING) for records in rows.values() for r in records}
    )
    lines = [
        "# Generation eval",
        "",
        f"Model `{model}` via OpenRouter, {python_version()}. Task: accounts and todo lists, "
        "backend to frontend, in a FastAPI + JS starter project.",
        f"Reasoning effort: {', '.join(efforts) or DEFAULT_REASONING}. "
        f"Input tokens: whole prompt, {counter.name}{' (estimate)' if counter.approximate else ''}."
        " Output tokens: as counted by the provider (mean over runs).",
        "",
        "| Context | Input | Input saved | Output | Total | Total saved | Cut off | Complete"
        " | Python parses | Edits applied | Restored | Time, s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    base_total = 0.0
    for variant in VARIANTS:
        records = rows[variant]
        if not records:
            continue
        label = variant if all(r["current"] for r in records) else f"{variant} (stale)"
        outputs = [
            float(r["usage"].get("completion_tokens") or counter.count(r["content"]))
            for r in records
        ]
        output = mean(outputs)
        total = inputs[variant] + output
        base_total = base_total or (total if variant == "original" else 0.0)
        checks = [completeness(answer_of(variant, r["content"])) for r in records]
        complete = mean(c["checks_passed"].__len__() / c["checks_total"] for c in checks) * 100
        py_total = sum(c["python_files"] for c in checks)
        py_ok = sum(c["python_valid"] for c in checks)
        edits = sum(c["edits"] for c in checks)
        applied = f"{edits - sum(c['edits_failed'] for c in checks)}/{edits}" if edits else "—"
        restored = "—"
        if "+lazy" in variant or "+dedent" in variant:
            restored = f"{sum(c['restored'] for c in checks)} files"
            if sum(c["restore_failed"] for c in checks):
                restored += f", {sum(c['restore_failed'] for c in checks)} failed"
            if sum(c["stubs_left"] for c in checks):
                restored += f", {sum(c['stubs_left'] for c in checks)} stubs left"
        cut = sum(1 for r in records if r.get("finish_reason") == "length")
        lines.append(
            f"| {label} | {inputs[variant]} | {percent(inputs[variant], inputs['original'])} | "
            f"{output:.0f} | {total:.0f} | {percent(total, base_total)} | {cut}/{len(records)} | "
            f"{complete:.0f}% | {py_ok}/{py_total} | {applied} | {restored} | "
            f"{mean(r['seconds'] for r in records):.0f} |"
        )
    missing = [
        f"{v}: {', '.join(sorted({n for n, _ in CHECKS} - set(c['checks_passed'])))}"
        for v in VARIANTS
        for c in (completeness(answer_of(v, r["content"])) for r in rows[v])
        if len(c["checks_passed"]) < c["checks_total"]
    ]
    if any(not r["current"] for records in rows.values() for r in records):
        lines += [
            "",
            "(stale): the answer was given for an older prompt (shrink has changed since); "
            "run the eval again to replace it.",
        ]
    if missing:
        lines += ["", "## Missing parts", ""] + [f"- {item}" for item in missing]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = "\n".join(lines) + "\n"
    report_path(model).write_text(out, encoding="utf-8", newline="\n")
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=rq.DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="no network: show prompt sizes")
    parser.add_argument("--report", action="store_true", help="only rebuild the table")
    parser.add_argument("--redo", choices=VARIANTS, help="ask again for one context variant")
    parser.add_argument(
        "--reasoning",
        choices=REASONING_EFFORTS,
        default=DEFAULT_REASONING,
        help="reasoning effort sent to the model (none = no hidden thinking tokens)",
    )
    parser.add_argument(
        "--variants",
        type=lambda text: [v.strip() for v in text.split(",") if v.strip()],
        help="only these variants, comma-separated, e.g. original,L3,L3+brief",
    )
    args = parser.parse_args()
    try:
        if args.redo:
            archive(args.model, args.redo)
        if args.report:
            report(args.model)
        else:
            unknown = sorted(set(args.variants or []) - set(VARIANTS))
            if unknown:
                parser.error(f"unknown variant(s): {', '.join(unknown)}")
            run(args.model, args.runs, args.dry_run, args.variants, args.reasoning)
    except KeyboardInterrupt:
        sys.exit("\nstopped; answers saved so far are kept, run again to continue")
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
