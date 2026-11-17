#!/usr/bin/env bash
# Copy little-coder (branch harness-rsi) into repos/HGM/lc/ at a pinned commit.
# lc/ is the agent that the harness-rsi arms of HGM evaluate and let improve itself.
#   usage: scripts/vendor_littlecoder.sh <little-coder checkout> [commit]
set -euo pipefail
SRC=${1:?little-coder checkout}; REV=${2:-HEAD}
HERE="$(cd "$(dirname "$0")/.." && pwd)"; DST="$HERE/lc"
rm -rf "$DST"; mkdir -p "$DST"
git -C "$SRC" archive "$REV" | tar -x -C "$DST" \
  --exclude='tests' --exclude='docs' --exclude='demos' --exclude='assets' --exclude='voice' \
  --exclude='video' --exclude='benchmarks' --exclude='.github'
git -C "$SRC" rev-parse "$REV" > "$DST/VENDORED_FROM"
# container requirements: runtime only (the upstream file lists optional voice backends)
printf 'openai>=1.30\nanthropic>=0.40\nhttpx>=0.27\nrich>=13\n' > "$DST/requirements.txt"
echo "vendored little-coder $(cat "$DST/VENDORED_FROM") into $DST ($(find "$DST" -type f | wc -l) files)"
