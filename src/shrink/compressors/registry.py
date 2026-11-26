"""Mode registry: name → compressor class, imported only when it is used.

Adding a mode means writing a module and adding one :class:`ModeSpec` line to
:data:`BUILTIN_MODES`; the CLI and the pipeline stay untouched (Open/Closed).
Keeping targets as strings means ``import shrink.cli`` never imports a
compressor it will not run, which keeps start-up time flat as modes grow.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from shrink.compressors.base import Compressor


@dataclass(frozen=True)
class ModeSpec:
    name: str
    target: str  # "package.module:ClassName"
    summary: str


BUILTIN_MODES: tuple[ModeSpec, ...] = (
    ModeSpec(
        name="auto",
        target="shrink.compressors.auto:AutoCompressor",
        summary="choose code, web, json, log or text for each file",
    ),
    ModeSpec(
        name="code",
        target="shrink.compressors.python_code:CodeCompressor",
        summary="Python: drop comments, shorten docstrings, reduce functions to signatures",
    ),
    ModeSpec(
        name="json",
        target="shrink.compressors.json_mode:JsonCompressor",
        summary="JSON / JSON Lines: minify, shorten long arrays and strings",
    ),
    ModeSpec(
        name="log",
        target="shrink.compressors.log:LogCompressor",
        summary="logs and build output: colour codes, timestamps, repeats, debug lines",
    ),
    ModeSpec(
        name="text",
        target="shrink.compressors.text:TextCompressor",
        summary="normalise whitespace, drop duplicate paragraphs and markup noise",
    ),
    ModeSpec(
        name="web",
        target="shrink.compressors.web:WebCompressor",
        summary="JavaScript, CSS, HTML: comments, JSDoc, CSS minification, function bodies",
    ),
)


class UnknownModeError(LookupError):
    pass


class Registry:
    def __init__(self, specs: Iterable[ModeSpec] = ()) -> None:
        self._specs: dict[str, ModeSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ModeSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"mode {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self) -> list[ModeSpec]:
        return [self._specs[name] for name in self.names()]

    def spec(self, name: str) -> ModeSpec:
        try:
            return self._specs[name]
        except KeyError:
            known = ", ".join(self.names()) or "none"
            raise UnknownModeError(f"unknown mode {name!r} (known: {known})") from None

    def create(self, name: str) -> Compressor:
        module_name, _, class_name = self.spec(name).target.partition(":")
        module = importlib.import_module(module_name)
        factory = getattr(module, class_name)
        compressor: Compressor = factory()
        return compressor

    def create_chain(self, names: Sequence[str]) -> Compressor:
        """One mode as itself, several as a chain run left to right."""
        if len(names) == 1:
            return self.create(names[0])
        from shrink.compressors.chain import ChainCompressor

        return ChainCompressor([self.create(name) for name in names])


default_registry = Registry(BUILTIN_MODES)
