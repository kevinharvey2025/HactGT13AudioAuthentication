#!/bin/bash -l
# Unpack the organizers' archives and fetch the bona fide references into $WS/data (idempotent):
#   data/hearsay_test/            HackGTHearsayTesting.zip: 1,671 test WAVs + score template
#   data/DiffSSD/real_speech/     LJRealResampled.zip: the 242 organizer reals (16 kHz)
#   data/DiffSSD/generated_speech DiffSSD.zip (70,000 fakes), only once DiffSSD.zip.verified exists
#   data/DiffSSD/metadata.csv     rebuilt from the tree (scripts/rebuild_diffssd_metadata.py)
#   data/HackGTMinDCF/            organizers' ASVspoof5 scoring package (minDCF, Pspoof 0.5, Cfa 4)
#   data/external/LJSpeech-1.1    keithito.com (2.6 GB)
#   data/external/librispeech_10spk  the 10 DiffSSD clone speakers (HF parquet row groups)
# The zips are expected in $WS/data as uploaded from the laptop ("Copy of <name>.zip").
#SBATCH --job-name=hearsay-data
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
set -eo pipefail
source "$CODE_DIR/mpcdf/env.sh"
use_venv neural
D=$WS/data
cd "$D"

unpack() {  # unpack ZIP DEST [tar args...]: each zip holds a single tar
  local zip=$1 dest=$2; shift 2
  mkdir -p "$dest"
  unzip -p "$zip" | tar -x -C "$dest" "$@"
}

if [ ! -f hearsay_test/HGT_Hearsay_score_template.csv ]; then
  unpack "Copy of HackGTHearsayTesting.zip" "$D" --transform 's,^HackGTHearsayTesting,hearsay_test,'
fi
echo "test wavs: $(ls hearsay_test/*.wav | wc -l)"

if [ ! -d DiffSSD/real_speech ]; then
  unpack "Copy of LJRealResampled.zip" "$D/DiffSSD" --transform 's,^resampled,real_speech,'
fi
echo "organizer reals: $(ls DiffSSD/real_speech/*.wav | wc -l)"

if [ ! -d HackGTMinDCF ]; then
  unpack "Copy of HackGTMinDCF.zip" "$D"
fi

if [ -f DiffSSD.zip.verified ] && [ ! -f DiffSSD/.generated_complete ]; then
  unpack DiffSSD.zip "$D" --exclude='*.DS_Store'
  touch DiffSSD/.generated_complete
fi
if [ -f DiffSSD/.generated_complete ]; then
  echo "DiffSSD fakes: $(find DiffSSD/generated_speech -type f | wc -l)"
  python "$CODE_DIR/scripts/rebuild_diffssd_metadata.py" "$D/DiffSSD"
else
  echo "DiffSSD.zip not verified yet: skipping the fakes"
fi

cd "$CODE_DIR"   # the fetch scripts write to data/external relative to the repo (data -> $WS/data)
if [ "$(ls data/external/LJSpeech-1.1/wavs 2>/dev/null | wc -l)" != 13100 ]; then
  bash scripts/fetch_ljspeech.sh
  rm -f data/external/LJSpeech-1.1.tar.bz2
fi
if [ ! -f data/external/librispeech_10spk/index.csv ]; then
  python scripts/fetch_librispeech_speakers.py
fi
echo "librispeech: $(tail -n +2 data/external/librispeech_10spk/index.csv | wc -l) utterances"
du -sh "$D"/* 2>/dev/null
