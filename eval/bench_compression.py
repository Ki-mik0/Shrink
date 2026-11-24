"""Compression benchmark: shrink against Repomix and skim on the same corpus.

For every Python file: tokens (o200k_base, exact) after each tool and level,
and whether the output still parses as Python (``ast.parse``). For text, logs
and JSON: shrink's own modes. Results go to ``eval/results/compression.md``
and ``.json``.

    python eval/bench_compression.py            # shrink only
    python eval/bench_compression.py --others   # also Repomix and skim via npx

Needs the tokenizer (``shrink --download tokenizer``); Repomix and skim need
Node.js (``npx`` downloads them on first use).
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import ROOT, Sample, other_samples, python_samples, python_version

from shrink.compressors.auto import AutoCompressor
from shrink.compressors.python_code import CodeCompressor
from shrink.document import Document
from shrink.observability.stats import TiktokenCounter
from shrink.splitting.formats import FORMATS
from shrink.splitting.framed import FramedSplitter

RESULTS = ROOT / "eval" / "results"
NPX = shutil.which("npx") or "npx"
REPOMIX = "repomix@1.19.0"
SKIM = "rskim@3.3.0"


def parses(text: str) -> bool:
    try:
        ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    return True


def run_shrink(samples: list[Sample], count) -> dict[str, dict[str, dict]]:  # type: ignore[no-untyped-def]
    code = CodeCompressor()
    out: dict[str, dict[str, dict]] = {}
    for sample in samples:
        text = sample.read()
        row = {"original": {"tokens": count(text), "valid": parses(text), "ms": 0.0}}
        for level in (1, 2, 3):
            started = time.perf_counter()
            result = code.compress(Document(text, path=sample.name), level).text
            row[f"shrink L{level}"] = {
                "tokens": count(result),
                "valid": parses(result),
                "ms": (time.perf_counter() - started) * 1000,
            }
        out[sample.name] = row
    return out


def run_repomix(samples: list[Sample], count, table: dict) -> None:  # type: ignore[no-untyped-def]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        names = {}
        for i, sample in enumerate(samples):
            name = f"f{i:02d}_{Path(sample.name).name}"
            (folder / name).write_text(sample.read(), encoding="utf-8")
            names[name] = sample.name
        started = time.perf_counter()
        packed = subprocess.run(
            [
                NPX,
                "-y",
                REPOMIX,
                "--compress",
                "--style",
                "plain",
                "--stdout",
                "--quiet",
                "--no-file-summary",
                "--no-directory-structure",
                "--no-security-check",
                ".",
            ],
            cwd=folder,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
        elapsed = (time.perf_counter() - started) * 1000
    docs = list(FramedSplitter(FORMATS["repomix-plain"]).split(packed.splitlines(keepends=True)))
    for doc in docs:
        original = names.get(Path(doc.path or "").name)
        if original is not None:
            table[original]["repomix --compress"] = {
                "tokens": count(doc.text),
                "valid": parses(doc.text),
                "ms": elapsed / len(samples),
            }


def run_skim(samples: list[Sample], count, table: dict, mode: str) -> None:  # type: ignore[no-untyped-def]
    for sample in samples:
        started = time.perf_counter()
        result = subprocess.run(
            [NPX, "-y", SKIM, "--mode", mode, "--no-header", "--no-cache", str(sample.path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        table[sample.name][f"skim {mode}"] = {
            "tokens": count(result.stdout),
            "valid": parses(result.stdout),
            "ms": (time.perf_counter() - started) * 1000,
            "notice": result.stderr.strip()[:200],
        }


def run_other_modes(samples: list[Sample], count) -> dict[str, dict]:  # type: ignore[no-untyped-def]
    auto = AutoCompressor()
    out = {}
    for sample in samples:
        text = sample.read()
        doc = Document(text, path=sample.path.name)
        row = {"kind": sample.kind, "original": count(text)}
        for level in (1, 2, 3):
            row[f"L{level}"] = count(auto.compress(doc, level).text)
        out[sample.name] = row
    return out


def percent(before: int, after: int) -> str:
    return f"{(after - before) / before * 100:+.1f}%" if before else "—"


def markdown(code: dict, other: dict, columns: list[str], tool_versions: str) -> str:
    lines = [
        "# Compression benchmark",
        "",
        f"{python_version()}, tokenizer o200k_base (exact). {tool_versions}",
        "Generated by `python eval/bench_compression.py --others`.",
        "",
        "## Python: totals over the corpus",
        "",
        "| Tool | Tokens | Change | Files that still parse |",
        "| --- | ---: | ---: | ---: |",
    ]
    total_original = sum(row["original"]["tokens"] for row in code.values())
    for column in columns:
        rows = [row[column] for row in code.values() if column in row]
        if not rows:
            continue
        tokens = sum(r["tokens"] for r in rows)
        valid = sum(1 for r in rows if r["valid"])
        lines.append(
            f"| {column} | {tokens} | {percent(total_original, tokens)} | {valid}/{len(rows)} |"
        )
    lines += ["", "## Python: per file (tokens; ✗ = output is not valid Python)", ""]
    lines.append("| File | " + " | ".join(columns) + " |")
    lines.append("| --- |" + " ---: |" * len(columns))
    for name, row in code.items():
        cells = []
        for column in columns:
            cell = row.get(column)
            cells.append(
                "—" if cell is None else f"{cell['tokens']}{'' if cell['valid'] else ' ✗'}"
            )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Text, logs, JSON (shrink, mode chosen by `auto`)",
        "",
        "| File | Kind | Original | Level 1 | Level 2 | Level 3 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in other.items():
        base = row["original"]
        lines.append(
            f"| {name} | {row['kind']} | {base} | "
            + " | ".join(f"{row[f'L{k}']} ({percent(base, row[f'L{k}'])})" for k in (1, 2, 3))
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--others", action="store_true", help="also run Repomix and skim (npx)")
    args = parser.parse_args()
    count = TiktokenCounter().count  # exact counts are the point: fail if missing
    samples = python_samples()
    code = run_shrink(samples, count)
    columns = ["original", "shrink L1", "shrink L2", "shrink L3"]
    versions = ""
    if args.others:
        run_repomix(samples, count, code)
        run_skim(samples, count, code, "structure")
        run_skim(samples, count, code, "signatures")
        columns += ["repomix --compress", "skim structure", "skim signatures"]
        versions = f"Repomix {REPOMIX.split('@')[1]}, skim {SKIM.split('@')[1]} (via npx)."
    other = run_other_modes(other_samples(), count)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "compression.json").write_text(
        json.dumps({"python": code, "other": other, "env": python_version()}, indent=1) + "\n",
        encoding="utf-8",
    )
    report = markdown(code, other, columns, versions)
    (RESULTS / "compression.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
