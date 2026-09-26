#!/bin/zsh
# Commit DSP-track paths to branch `dsp-track` WITHOUT touching HEAD, the shared
# index, or the working tree. The working tree is shared with another Claude
# session that commits its own paths to `main`; `git checkout`, `git add -A`,
# `git commit -a` would disturb it or sweep up its files.
#
# Usage: scripts/dsp_git_commit.sh <commit-message-file>
# The first commit bases dsp-track on main; later commits stack on dsp-track.
set -euo pipefail
cd "$(dirname "$0")/.."
BR=dsp-track
MSG="$1"
IDX="$(mktemp -t dsp_index)"
rm -f "$IDX"
export GIT_INDEX_FILE="$IDX"
if git rev-parse --verify -q "refs/heads/$BR" >/dev/null; then
  PARENT=$(git rev-parse "refs/heads/$BR"); OLD=$PARENT
else
  PARENT=$(git rev-parse refs/heads/main); OLD=""
fi
git read-tree "$PARENT"
PATHS=()
for p in hearsay_dsp configs/dsp.yaml tests/dsp README_DSP.md HANDOFF_DSP.md Dockerfile.dsp \
         Dockerfile.dsp.dockerignore requirements-dsp.txt reports/dsp; do
  [[ -e "$p" ]] && PATHS+=("$p")
done
for p in scripts/dsp_*(N); do PATHS+=("$p"); done
git add -- "${PATHS[@]}"
TREE=$(git write-tree)
if [[ "$TREE" == "$(git rev-parse "$PARENT^{tree}")" ]]; then
  echo "nothing new to commit on $BR"; rm -f "$IDX"; exit 0
fi
COMMIT=$(git commit-tree "$TREE" -p "$PARENT" -F "$MSG")
if [[ -n "$OLD" ]]; then git update-ref "refs/heads/$BR" "$COMMIT" "$OLD"; else git update-ref "refs/heads/$BR" "$COMMIT"; fi
rm -f "$IDX"
unset GIT_INDEX_FILE
git log --oneline -1 "$BR"
git show --stat --format="" "$BR" | tail -3
