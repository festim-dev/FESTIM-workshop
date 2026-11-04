#!/usr/bin/env bash
# Runs once when the codespace is created (onCreateCommand).
#
# Most workshop notebooks are stored as MyST markdown (.md, paired via jupytext) so they diff
# well and build with jupyter-book. VS Code opens those as plain text, so generate the .ipynb
# twin of each one. The generated files are listed in .git/info/exclude (local to this clone,
# never committed) so they don't flood the Source Control panel.
set -euo pipefail
cd "$(dirname "$0")/.."

exclude=.git/info/exclude
mkdir -p "$(dirname "$exclude")"
touch "$exclude"

count=0
while IFS= read -r -d '' md; do
    grep -q '^jupytext:' "$md" || continue          # plain markdown page, not a notebook
    nb="${md%.md}.ipynb"
    if git ls-files --error-unmatch "$nb" >/dev/null 2>&1; then
        continue                                    # a committed .ipynb already exists
    fi
    jupytext --to ipynb --quiet "$md"
    grep -qxF "/$nb" "$exclude" || echo "/$nb" >> "$exclude"
    count=$((count + 1))
done < <(find book/content -name '*.md' -not -path '*/_build/*' -print0)

echo "on-create: generated $count notebooks from MyST sources"
