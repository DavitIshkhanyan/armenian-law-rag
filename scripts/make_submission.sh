#!/usr/bin/env bash
# Build the submission zip from the committed tree (so .env, caches and the source PDFs can never
# be included), optionally adding the screen recording.
#
# Usage: scripts/make_submission.sh [path/to/recording.mp4]
set -euo pipefail
cd "$(dirname "$0")/.."
name="armenian-law-rag"
out="../${name}-submission.zip"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Commit or stash your changes first; the zip is built from HEAD." >&2
  exit 1
fi
rm -f "$out"
git archive --format=zip --prefix="${name}/" -o "$out" HEAD
if [ $# -ge 1 ]; then
  tmp="$(mktemp -d)"
  mkdir -p "$tmp/${name}"
  cp "$1" "$tmp/${name}/screen-recording.${1##*.}"
  (cd "$tmp" && zip -q "$OLDPWD/$out" "${name}/screen-recording.${1##*.}")
  rm -rf "$tmp"
fi
echo "Wrote $(cd "$(dirname "$out")" && pwd)/$(basename "$out")"
unzip -l "$out" | tail -1
