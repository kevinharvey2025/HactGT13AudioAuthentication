"""Paths and constants shared by every stage. Override roots with HEARSAY_DATA / HEARSAY_CACHE."""
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("HEARSAY_DATA", REPO / "data"))
CACHE = Path(os.environ.get("HEARSAY_CACHE", REPO / "cache"))
RUNS = Path(os.environ.get("HEARSAY_RUNS", REPO / "runs" / "diffusion"))

DIFFSSD = DATA / "DiffSSD"                      # HF layout: metadata.csv, generated_speech/, real_speech/
TEST_DIR = DATA / "hearsay_test"                # NSA sample test set (wav + HGT_Hearsay_score_template.csv)
EXTERNAL = DATA / "external"                    # bona fide references (LibriSpeech speakers, LJSpeech-1.1)

SR = 16000                                      # every model sees 16 kHz mono, as the test files are
TEAM = os.environ.get("HEARSAY_TEAM", "SideQuests")

# DiffSSD generator families. LJ-voice TTS models were trained on LJSpeech; the cloning
# systems imitate 10 LibriSpeech train-clean-360 speakers.
LJ_VOICE = ["diffgan_tts", "grad_tts", "pro_diff", "wavegrad2"]
CLONE = ["elevenlabs", "openvoicev2", "playht", "unit_speech", "xtts_v2", "your_tts"]
GENERATORS = LJ_VOICE + CLONE


def ffmpeg_bin(name="ffmpeg"):
    """ffmpeg from PATH, else from the active env (conda envs ship it next to python)."""
    from shutil import which
    return which(name) or str(Path(sys.prefix) / "bin" / name)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
