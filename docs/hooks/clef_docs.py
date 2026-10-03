"""MkDocs hooks for the clef docs.

* Publish the repo-root ``openapi.json`` at the site root so the interactive API
  reference always shows the current schema (also works with ``mkdocs serve``).
* Wrapper pages (changelog, contributing, security) pull in the root files and
  rewrite their repo-relative links so they resolve on the site.
"""

from __future__ import annotations

import re
from pathlib import Path

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[2]

# Link targets that point outside docs/ (or are relative to the repo root).
_REWRITES = [
    ("](../openapi.json)", "](openapi.json)"),
    ("](../CHANGELOG.md)", "](changelog.md)"),
    ("](../CONTRIBUTING.md)", "](contributing.md)"),
    ("](../SECURITY.md)", "](security.md)"),
]
_ROOT_PAGES = {
    "changelog.md": "CHANGELOG.md",
    "contributing.md": "CONTRIBUTING.md",
    "security.md": "SECURITY.md",
}


def on_files(files, config):
    src = ROOT / "openapi.json"
    if src.exists():
        files.append(File.generated(config, "openapi.json", abs_src_path=str(src)))
    return files


def on_page_markdown(markdown, page, config, files):
    root_file = _ROOT_PAGES.get(page.file.src_uri)
    if root_file:
        # Single source of truth: the root file is the page body (snippets would run
        # after this hook, too late for link rewriting).
        markdown = (ROOT / root_file).read_text(encoding="utf-8")
    for old, new in _REWRITES:
        markdown = markdown.replace(old, new)
    if root_file:
        # `docs/foo.md` -> `foo.md`; other repo paths -> the repository on GitHub
        markdown = markdown.replace("](docs/", "](")
        repo = config["repo_url"] + "/blob/main/"
        markdown = re.sub(
            r"\]\(((?:examples|src|scripts|requirements|deploy|bench|clients|tests|\.github)[^)]*)\)",
            lambda m: f"]({repo}{m.group(1)})",
            markdown,
        )
    return markdown
