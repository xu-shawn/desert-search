#!/bin/bash
# Build docs/PROGRESSION.pdf from docs/PROGRESSION.md (needs pandoc and chromium).
set -e; cd "$(dirname "$0")"
pandoc PROGRESSION.md -f markdown+pipe_tables+implicit_figures -t html5 --standalone --wrap=none \
  --metadata pagetitle="$(head -1 PROGRESSION.md | sed 's/^# //')" --css progression.css -o PROGRESSION.html
chromium --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=10000 \
  --print-to-pdf="$PWD/PROGRESSION.pdf" "file://$PWD/PROGRESSION.html" 2>/dev/null
rm -f PROGRESSION.html
echo "wrote $PWD/PROGRESSION.pdf"
