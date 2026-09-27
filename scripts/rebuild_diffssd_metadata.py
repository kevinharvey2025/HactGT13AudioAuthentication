"""Rebuild data/DiffSSD/metadata.csv from the audio tree.

    python scripts/rebuild_diffssd_metadata.py [data/DiffSSD] [--listing FILE]

--listing takes relative paths (generated_speech/..., real_speech/...) from a file instead of walking
the tree, e.g. to sample manifests before every file has been unpacked.

The organizers' DiffSSD archive has no metadata file (the first machine used one from a private
Hugging Face copy). Both manifest builders only need what the paths encode:

    generated_speech/<lj_voice_gen>/sentence_<sid>.wav                        speaker LJSpeech
    generated_speech/<clone_gen>/speaker_<spk>/sentence_<sid>[_<style>].<ext>  style: openvoicev2 only
    real_speech/<LJxxx-xxxx>.wav                                               the 242 organizer reals

Columns: file_name, label (fake|real), generator, speaker, sentence_id, style, utterance_id.
Rows are sorted by file_name, so the file is identical on every machine with the same audio.
"""
import re
from pathlib import Path

import pandas as pd

LJ_VOICE = {"diffgan_tts", "grad_tts", "pro_diff", "wavegrad2"}
CLONE = {"elevenlabs", "openvoicev2", "playht", "unit_speech", "xtts_v2", "your_tts"}
AUDIO = {".wav", ".mp3", ".flac"}
SENT = re.compile(r"sentence_(\d+)(?:_(.+))?$")


def tree(root):
    """Relative paths of every file under generated_speech/ and real_speech/."""
    for sub in ("generated_speech", "real_speech"):
        for p in (root / sub).rglob("*"):
            if p.is_file():
                yield p.relative_to(root)


def rows(paths):
    paths = sorted(Path(p) for p in paths)
    for rel in paths:
        if rel.parts[0] != "generated_speech" or rel.suffix.lower() not in AUDIO:
            continue  # real_speech below; .DS_Store and friends
        gen = rel.parts[1]
        m = SENT.match(rel.stem)
        if m is None or gen not in LJ_VOICE | CLONE:
            raise ValueError(f"unexpected DiffSSD path: {rel}")
        if gen in LJ_VOICE:
            assert len(rel.parts) == 3, rel
            speaker = "LJSpeech"
        else:
            assert len(rel.parts) == 4 and rel.parts[2].startswith("speaker_"), rel
            speaker = rel.parts[2].removeprefix("speaker_")
        yield dict(file_name=rel.as_posix(), label="fake", generator=gen, speaker=speaker,
                   sentence_id=int(m.group(1)), style=m.group(2) or "", utterance_id="")
    for rel in paths:
        if rel.parts[0] == "real_speech" and rel.suffix.lower() == ".wav":
            yield dict(file_name=rel.as_posix(), label="real", generator="real",
                       speaker="LJSpeech", sentence_id=None, style="", utterance_id=rel.stem)


def main(root="data/DiffSSD", listing=None):
    root = Path(root)
    paths = [line.strip() for line in open(listing) if line.strip()] if listing else tree(root)
    df = pd.DataFrame(rows(paths)).sort_values("file_name", ignore_index=True)
    df["sentence_id"] = df.sentence_id.astype("Int64")
    df.to_csv(root / "metadata.csv", index=False)
    print(df.groupby(["label", "generator"]).size().to_string())
    print(f"{len(df)} rows -> {root / 'metadata.csv'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/DiffSSD")
    ap.add_argument("--listing")
    a = ap.parse_args()
    main(a.root, a.listing)
