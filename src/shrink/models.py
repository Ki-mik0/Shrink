"""Files shrink downloads once, on request (``shrink --download translate``).

Every file comes from a pinned URL (a fixed Hugging Face revision, the official
fastText release) and is checked against a known size and, for large files, a
SHA-256 digest before it is used. Downloads go to a ``.part`` file first, so an
interrupted download never leaves a half file behind that looks complete.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from shrink.paths import cache_dir

NLLB_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
NLLB_REVISION = "302d78f00e6fdb50a1064059df7c392b735e9d05"
NLLB_DIR = "nllb-200-distilled-600M-ct2-int8"
_HF = f"https://huggingface.co/{NLLB_REPO}/resolve/{NLLB_REVISION}"


@dataclass(frozen=True)
class RemoteFile:
    url: str
    name: str
    size: int
    sha256: str | None = None  # small text files are checked by size only


TRANSLATE_FILES = (
    RemoteFile(f"{_HF}/config.json", f"{NLLB_DIR}/config.json", 159),
    RemoteFile(f"{_HF}/shared_vocabulary.txt", f"{NLLB_DIR}/shared_vocabulary.txt", 2_568_098),
    RemoteFile(
        f"{_HF}/sentencepiece.bpe.model",
        f"{NLLB_DIR}/sentencepiece.bpe.model",
        4_852_054,
        "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a",
    ),
    RemoteFile(
        f"{_HF}/model.bin",
        f"{NLLB_DIR}/model.bin",
        622_595_991,
        "ed1beaf75134de7505315a5223162f56acff397eff6b50638a500d3936fe707b",
    ),
    RemoteFile(
        "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
        "lid.176.ftz",
        938_013,
    ),
)

READY = "translate.ready"


def translate_dir() -> Path:
    return cache_dir() / "translate"


def translate_ready() -> bool:
    return (translate_dir() / READY).is_file()


Progress = Callable[[str], None]


def _say(message: str) -> None:
    sys.stderr.write(f"shrink: {message}\n")
    sys.stderr.flush()


def download_translate(progress: Progress = _say) -> Path:
    folder = translate_dir()
    for remote in TRANSLATE_FILES:
        fetch(remote, folder, progress)
    (folder / READY).write_text("ok\n", encoding="utf-8")
    return folder


def fetch(remote: RemoteFile, folder: Path, progress: Progress = _say) -> Path:
    """Download one file unless a verified copy is already there."""
    target = folder / remote.name
    if target.is_file() and _verified(target, remote):
        progress(f"{remote.name}: already downloaded")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    request = urllib.request.Request(remote.url, headers={"User-Agent": "shrink"})
    with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
        done, step = 0, max(remote.size // 10, 1)
        shown = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if remote.size > 10_000_000 and done - shown >= step:
                shown = done
                progress(f"{remote.name}: {done * 100 // remote.size}%")
    if not _verified(part, remote):
        part.unlink(missing_ok=True)
        raise ValueError(f"{remote.name}: size or checksum does not match the pinned file")
    part.replace(target)
    progress(f"{remote.name}: ok ({remote.size / 1_000_000:.1f} MB)")
    return target


def _verified(path: Path, remote: RemoteFile) -> bool:
    if path.stat().st_size != remote.size:
        return False
    if remote.sha256 is None:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest() == remote.sha256
