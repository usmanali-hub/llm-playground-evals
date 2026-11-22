"""Prompt templates kept as versioned files on disk, instead of strings embedded in code.

A prompt is a plain text file with a small frontmatter block (between two ``---`` lines) naming its variables, and
a body that references them as ``{{var}}``. Several versions of the same prompt live side by side as separate files
(``summarize/v1.prompt``, ``summarize/v2.prompt``, ...), so changing a prompt is a new file, not an edit that loses
the old wording, and a specific call can pin an exact version instead of always getting "whatever is newest".

This only reads and renders files; running a prompt through a provider and recording which version was used (a
natural pairing with `UsageTracker`) is left to the caller, since that depends on how the rest of an application is
wired.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

VERSION_PATTERN = re.compile(r"^v(\d+)$")
PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptError(ValueError):
    """A prompt file was malformed, or a render call's variables didn't match what the template declares."""


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: int
    description: str
    variables: tuple[str, ...]
    template: str
    path: Path

    def placeholders(self) -> set[str]:
        """Variable names actually referenced in the template body as ``{{var}}``."""
        return set(PLACEHOLDER.findall(self.template))

    def unused_variables(self) -> set[str]:
        """Declared variables the template body never references — usually a stale frontmatter entry."""
        return set(self.variables) - self.placeholders()

    def undeclared_placeholders(self) -> set[str]:
        """``{{var}}`` references in the body that frontmatter never declared — usually a typo."""
        return self.placeholders() - set(self.variables)

    def render(self, **values: str) -> str:
        """Substitute every declared variable; raises `PromptError` if any is missing or an extra one was passed.

        A literal single-brace ``{...}`` (common in a prompt that shows example JSON) is left untouched; only the
        double-brace ``{{var}}`` form is treated as a placeholder.
        """
        missing = set(self.variables) - values.keys()
        if missing:
            raise PromptError(f"{self.name} {self._label()}: missing value(s) for {sorted(missing)}")
        extra = values.keys() - set(self.variables)
        if extra:
            raise PromptError(f"{self.name} {self._label()}: got value(s) for undeclared variable(s) {sorted(extra)}")

        def substitute(match: re.Match[str]) -> str:
            return str(values[match.group(1)])

        rendered = PLACEHOLDER.sub(lambda m: substitute(m) if m.group(1) in values else m.group(0), self.template)
        return rendered

    def _label(self) -> str:
        return f"v{self.version}"


def _parse_frontmatter(text: str, path: Path) -> tuple[dict[str, str], str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise PromptError(f"{path}: must start with a '---' frontmatter block")
    fields: dict[str, str] = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        line = lines[i]
        if line.strip():
            if ":" not in line:
                raise PromptError(f"{path}: line {i + 1} in frontmatter is not 'key: value': {line!r}")
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
        i += 1
    if i == len(lines):
        raise PromptError(f"{path}: frontmatter block was never closed with a second '---'")
    body = "".join(lines[i + 1 :]).lstrip("\n")
    return fields, body


def parse_prompt_file(path: Path) -> PromptVersion:
    """Parse one ``.prompt`` file. Its filename stem must be ``v<N>`` (``v1``, ``v2``, ...) and its parent
    directory's name is the prompt's ``name`` (``prompts/summarize/v2.prompt`` -> name "summarize", version 2)."""
    match = VERSION_PATTERN.match(path.stem)
    if not match:
        raise PromptError(f"{path}: filename must be 'v<N>.prompt' (e.g. v1.prompt), got {path.name!r}")
    fields, body = _parse_frontmatter(path.read_text(encoding="utf-8"), path)
    variables = tuple(v.strip() for v in fields.get("variables", "").split(",") if v.strip())
    return PromptVersion(
        name=path.parent.name,
        version=int(match.group(1)),
        description=fields.get("description", ""),
        variables=variables,
        template=body,
        path=path,
    )


@dataclass
class PromptLibrary:
    """All prompt versions found under a root directory, one subdirectory per prompt name."""

    root: Path
    _versions: dict[str, dict[int, PromptVersion]]

    @classmethod
    def load(cls, root: Path) -> PromptLibrary:
        root = Path(root)
        versions: dict[str, dict[int, PromptVersion]] = {}
        for prompt_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
            for file in sorted(prompt_dir.glob("*.prompt")):
                parsed = parse_prompt_file(file)
                by_version = versions.setdefault(parsed.name, {})
                if parsed.version in by_version:
                    raise PromptError(f"{parsed.name!r} has two files claiming version {parsed.version}")
                by_version[parsed.version] = parsed
        return cls(root, versions)

    def names(self) -> list[str]:
        return sorted(self._versions)

    def versions(self, name: str) -> list[int]:
        """Every version number on file for ``name``, ascending. Raises `PromptError` if the name is unknown."""
        if name not in self._versions:
            raise PromptError(f"no prompt named {name!r}; known prompts: {self.names()}")
        return sorted(self._versions[name])

    def get(self, name: str, version: int | None = None) -> PromptVersion:
        """The given version of ``name``, or its highest (latest) version if ``version`` is omitted."""
        available = self.versions(name)
        chosen = available[-1] if version is None else version
        if chosen not in self._versions[name]:
            raise PromptError(f"{name!r} has no version {chosen}; available: {available}")
        return self._versions[name][chosen]

    def render(self, name: str, version: int | None = None, **values: str) -> str:
        return self.get(name, version).render(**values)

    def __iter__(self) -> Iterator[PromptVersion]:
        for name in self.names():
            for version in self.versions(name):
                yield self._versions[name][version]
