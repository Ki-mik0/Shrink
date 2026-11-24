"""The ``shrink gain`` log: what ``shrink wrap`` saved, request by request.

One JSON line per request that changed, with counts only: no text, no
headers, nothing that could leak a prompt or a key. Lives in
:func:`shrink.paths.state_dir`.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shrink.integrations.shrinker import Saving, describe
from shrink.paths import state_dir


def gain_path() -> Path:
    return state_dir() / "gain.jsonl"


class GainLog:
    def __init__(self, path: Path | None = None, agent: str = "") -> None:
        self._path = path or gain_path()
        self._agent = agent
        self._lock = threading.Lock()
        self.requests = 0
        self.tokens_before = 0
        self.tokens_after = 0

    def record(self, saving: Saving) -> None:
        line = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "agent": self._agent,
            **describe(saving),
        }
        with self._lock:
            self.requests += 1
            self.tokens_before += saving.tokens_before
            self.tokens_after += saving.tokens_after
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(line, ensure_ascii=False) + "\n")


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """All records; damaged lines are skipped."""
    source = path or gain_path()
    if not source.is_file():
        return []
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


@dataclass
class Totals:
    requests: int = 0
    before: int = 0
    after: int = 0

    def add(self, record: dict[str, Any]) -> None:
        self.requests += 1
        self.before += int(record.get("tokens_before", 0))
        self.after += int(record.get("tokens_after", 0))

    @property
    def saved(self) -> int:
        return self.before - self.after

    def percent(self) -> str:
        return f"{-self.saved / self.before * 100:+.1f}%" if self.before else "0.0%"


def report(records: list[dict[str, Any]], days: int = 7) -> list[str]:
    if not records:
        return ["shrink gain: nothing recorded yet; run an agent through `shrink wrap`"]
    total = Totals()
    by_day: dict[str, Totals] = defaultdict(Totals)
    by_agent: dict[str, Totals] = defaultdict(Totals)
    for record in records:
        total.add(record)
        by_day[str(record.get("time", ""))[:10]].add(record)
        by_agent[str(record.get("agent") or record.get("api") or "?")].add(record)
    lines = [
        f"shrink gain: {total.requests} requests, changed text {total.before} -> {total.after} "
        f"tokens ({total.percent()}), saved {total.saved} tokens",
    ]
    lines += [
        f"  {agent:<10} {t.requests:>6} requests  saved {t.saved:>9} tokens ({t.percent()})"
        for agent, t in sorted(by_agent.items())
    ]
    lines.append(f"  last {days} days:")
    lines += [
        f"  {day}  {t.requests:>6} requests  saved {t.saved:>9} tokens ({t.percent()})"
        for day, t in sorted(by_day.items())[-days:]
    ]
    lines.append(
        "  (the conversation is re-sent every turn, so a saving counts once per request;"
        " cached input is billed cheaper)"
    )
    return lines


def summary_json(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = Totals()
    for record in records:
        total.add(record)
    return {
        "requests": total.requests,
        "tokens_before": total.before,
        "tokens_after": total.after,
        "tokens_saved": total.saved,
    }
