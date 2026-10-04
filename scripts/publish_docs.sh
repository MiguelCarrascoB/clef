#!/usr/bin/env bash
# Build the docs site (strict) and publish it to the gh-pages branch, which GitHub Pages serves
# ("Deploy from a branch", gh-pages, /). No GitHub Actions needed: .nojekyll skips the Jekyll build.
# Usage: bash scripts/publish_docs.sh [remote]   (default remote: origin; needs .venv-docs, see CONTRIBUTING.md)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
remote="${1:-origin}"
mkdocs=".venv-docs/bin/mkdocs"
[ -x "$mkdocs" ] || mkdocs=".venv-docs/Scripts/mkdocs"
[ -x "$mkdocs" ] || mkdocs=".venv-docs/Scripts/mkdocs.exe"
if [ ! -x "$mkdocs" ]; then
  echo "no .venv-docs: python -m venv .venv-docs && .venv-docs/bin/pip install -r requirements/docs.txt" >&2
  exit 1
fi
if [ -n "$(git status --porcelain -- docs mkdocs.yml openapi.json README.md CHANGELOG.md SECURITY.md CONTRIBUTING.md)" ]; then
  echo "uncommitted docs changes: commit them first so the site matches a commit" >&2
  exit 1
fi
rev="$(git rev-parse --short HEAD)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

NO_MKDOCS_2_WARNING=true "$mkdocs" build --strict -q -d "$work/site"
touch "$work/site/.nojekyll"

url="$(git remote get-url "$remote")"
if git ls-remote --exit-code --heads "$remote" gh-pages >/dev/null 2>&1; then
  git clone -q --depth 1 --branch gh-pages "$url" "$work/pages"
else
  git init -q -b gh-pages "$work/pages"
  git -C "$work/pages" remote add origin "$url"
fi
find "$work/pages" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$work/site/." "$work/pages/"
git -C "$work/pages" config user.name "$(git config user.name)"
git -C "$work/pages" config user.email "$(git config user.email)"
git -C "$work/pages" add -A
if git -C "$work/pages" diff --cached --quiet; then
  echo "gh-pages already matches $rev"
  exit 0
fi
git -C "$work/pages" commit -q -m "Docs site built from $rev"
git -C "$work/pages" push -q origin gh-pages
echo "published $rev to gh-pages; GitHub Pages updates in about a minute"
