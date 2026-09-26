#!/usr/bin/env bash
# Download LJSpeech-1.1 (2.6 GB) into data/external/: bona fide references for the four
# LJ-voice DiffSSD generators (diffgan_tts, grad_tts, pro_diff, wavegrad2).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/external
curl -L --fail -o data/external/LJSpeech-1.1.tar.bz2 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf data/external/LJSpeech-1.1.tar.bz2 -C data/external
ls data/external/LJSpeech-1.1/wavs | wc -l   # expect 13100
