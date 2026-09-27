#!/usr/bin/env bash
# Extra bona fide speech for the fine-tuning pool: LibriSpeech dev-clean + test-clean (80 speakers, ~10.5 h,
# CC BY 4.0), none of them cloned by DiffSSD. Writes data/external/extra_reals/{librispeech_<part>/..., index.csv}.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/external/extra_reals
mkdir -p "$D"
for part in dev-clean test-clean; do
  if [ ! -d "$D/librispeech_$part" ]; then
    curl -L --fail -s -o "$D/$part.tar.gz" "https://www.openslr.org/resources/12/$part.tar.gz"
    mkdir -p "$D/librispeech_$part"
    tar -xzf "$D/$part.tar.gz" -C "$D/librispeech_$part" --strip-components=2   # LibriSpeech/<part>/<spk>/<chapter>/*.flac
    rm -f "$D/$part.tar.gz"
  fi
done
# index: file (relative to $D), speaker, chapter, source
( echo "file,speaker,chapter,source"
  for part in dev-clean test-clean; do
    find "$D/librispeech_$part" -name "*.flac" | sort | while read -r f; do
      rel=${f#"$D/"}; spk=$(basename "$(dirname "$(dirname "$f")")"); ch=$(basename "$(dirname "$f")")
      echo "$rel,$spk,$ch,librispeech_$part"
    done
  done ) > "$D/index.csv"
echo "extra reals: $(( $(wc -l < "$D/index.csv") - 1 )) utterances"
