"""Fetch real LibriSpeech utterances for the 10 DiffSSD voice-cloning reference speakers.

DiffSSD's voice-cloning generators (elevenlabs, openvoicev2, playht, unit_speech, xtts_v2,
your_tts) clone 10 LibriSpeech train-clean-360 speakers. Real audio from the same speakers
gives a speaker-matched bona fide set, so a detector can't separate real from fake by voice.

Only the parquet row groups that contain those speakers are downloaded (~20 row groups),
not the 120 GB corpus. Output: data/external/librispeech_10spk/<speaker>/<id>.flac + index.csv
"""
import io
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

SPEAKERS = {100, 1487, 2061, 3654, 4490, 5448, 6167, 6575, 7995, 8848}
# speaker -> (shard, row groups), found by scanning the speaker_id column of every train shard
HITS = {
    "all/train.clean.360/0000.parquet": [0, 1],
    "all/train.clean.360/0007.parquet": [0, 1, 4, 5],
    "all/train.clean.360/0014.parquet": [9, 10],
    "all/train.clean.360/0024.parquet": [2, 3],
    "all/train.clean.360/0025.parquet": [20, 21],
    "all/train.clean.360/0026.parquet": [0],
    "all/train.clean.360/0030.parquet": [16, 17],
    "all/train.clean.360/0032.parquet": [10, 11],
    "all/train.clean.360/0034.parquet": [7, 8],
    "all/train.clean.360/0039.parquet": [0, 1],
}


def main(out_dir="data/external/librispeech_10spk"):
    fs = HfFileSystem()
    rows = []
    for shard, rgs in HITS.items():
        pf = pq.ParquetFile(fs.open(f"datasets/openslr/librispeech_asr/{shard}", block_size=8 << 20))
        for rg in rgs:
            t = pf.read_row_group(rg, columns=["id", "speaker_id", "chapter_id", "text", "audio"]).to_pylist()
            for r in t:
                if r["speaker_id"] not in SPEAKERS:
                    continue
                spk_dir = os.path.join(out_dir, str(r["speaker_id"]))
                os.makedirs(spk_dir, exist_ok=True)
                path = os.path.join(spk_dir, f"{r['id']}.flac")
                if not os.path.exists(path):
                    with open(path, "wb") as fh:
                        fh.write(r["audio"]["bytes"])
                info = sf.info(path)
                rows.append(dict(file=os.path.relpath(path, out_dir), speaker=r["speaker_id"], chapter=r["chapter_id"],
                                 utt_id=r["id"], text=r["text"], sr=info.samplerate, dur=info.duration))
            print(shard, rg, len(rows), flush=True)
    df = pd.DataFrame(rows).drop_duplicates("utt_id")
    df.to_csv(os.path.join(out_dir, "index.csv"), index=False)
    print(df.groupby("speaker").agg(n=("utt_id", "size"), hours=("dur", lambda d: d.sum() / 3600)).round(3))


if __name__ == "__main__":
    main(*sys.argv[1:])
