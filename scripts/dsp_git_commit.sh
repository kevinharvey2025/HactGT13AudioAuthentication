#!/bin/zsh
# Commit DSP-track paths onto a branch (default: main) WITHOUT switching branches,
# staging through the shared index, or touching other files. The working tree is
# shared with another Claude session that commits its own paths; `git checkout`,
# `git add -A`, `git add .` or `git commit -a` would disturb it or sweep up its files.
#
# How: build the new tree in a private GIT_INDEX_FILE (branch tip + DSP paths from
# the working tree), commit it with commit-tree, and advance the branch with a
# compare-and-swap update-ref (fails safely if someone committed meanwhile; just
# rerun). If the branch is the checked-out HEAD, the shared index entries for the
# DSP paths only are then synced to HEAD so `git status` stays clean.
#
# Usage: scripts/dsp_git_commit.sh <commit-message-file> [branch]
set -euo pipefail
cd "$(dirname "$0")/.."
MSG="$1"
BR="${2:-main}"
DSP_PATHS=()
for p in hearsay_dsp configs/dsp.yaml tests/dsp README_DSP.md HANDOFF_DSP.md Dockerfile.dsp \
         Dockerfile.dsp.dockerignore requirements-dsp.txt reports/dsp; do
  [[ -e "$p" ]] && DSP_PATHS+=("$p")
done
for p in scripts/dsp_*(N); do DSP_PATHS+=("$p"); done

PARENT=$(git rev-parse "refs/heads/$BR")
IDX="$(mktemp -t dsp_index)"
rm -f "$IDX"
GIT_INDEX_FILE="$IDX" git read-tree "$PARENT"
GIT_INDEX_FILE="$IDX" git add -- "${DSP_PATHS[@]}"
TREE=$(GIT_INDEX_FILE="$IDX" git write-tree)
rm -f "$IDX"
if [[ "$TREE" == "$(git rev-parse "$PARENT^{tree}")" ]]; then
  echo "nothing new to commit on $BR"; exit 0
fi
COMMIT=$(git commit-tree "$TREE" -p "$PARENT" -F "$MSG")
git update-ref -m "dsp_git_commit.sh" "refs/heads/$BR" "$COMMIT" "$PARENT"
if [[ "$(git symbolic-ref -q HEAD)" == "refs/heads/$BR" ]]; then
  git reset -q -- "${DSP_PATHS[@]}"   # sync shared index for DSP paths only
fi
git log --oneline -1 "$BR"
git show --stat --format="" "$BR" | tail -3
