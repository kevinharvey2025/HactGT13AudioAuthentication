"""Container / file forensics. Diagnostics only: never classifier inputs.

Reports what the file claims and whether the claims agree with each other:
extension vs sniffed container, header vs decoded duration, codec, rate,
bit rate, RIFF chunks and INFO tags (e.g. ISFT encoder string), MP3/ID3
encoder strings. Encoder tags, timestamps and MAC times are trivially spoofed
and, in the training data, perfectly confounded with the label, so they are
kept out of scoring.
"""

from __future__ import annotations

import os
import re
import struct

from .common import OK, ModuleResult

EXPECTED_EXT = {"wav": {".wav", ".wave"}, "flac": {".flac"}, "ogg": {".ogg", ".oga", ".opus"},
                "aiff": {".aif", ".aiff", ".aifc"}, "mp4": {".m4a", ".mp4", ".aac", ".3gp", ".mov"},
                "mp3": {".mp3"}, "aac_adts": {".aac"}, "asf": {".wma", ".asf"},
                "matroska": {".mka", ".mkv", ".webm"}}


def riff_chunks(path: str, max_chunks: int = 64) -> dict:
    out = {"chunks": [], "info": {}}
    with open(path, "rb") as fh:
        head = fh.read(12)
        if len(head) < 12 or head[8:12] != b"WAVE":
            return out
        for _ in range(max_chunks):
            hdr = fh.read(8)
            if len(hdr) < 8:
                break
            cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            name = cid.decode("latin-1", errors="replace")
            out["chunks"].append({"id": name, "size": size})
            if cid == b"LIST" and size >= 4 and size < 1 << 20:
                body = fh.read(size + (size & 1))
                if body[:4] == b"INFO":
                    pos = 4
                    while pos + 8 <= len(body):
                        sid = body[pos:pos + 4].decode("latin-1", errors="replace")
                        ssz = struct.unpack("<I", body[pos + 4:pos + 8])[0]
                        val = body[pos + 8:pos + 8 + ssz].split(b"\x00")[0]
                        out["info"][sid] = val.decode("utf-8", errors="replace")
                        pos += 8 + ssz + (ssz & 1)
            else:
                fh.seek(size + (size & 1), os.SEEK_CUR)
    return out


def mp3_encoder_strings(path: str) -> list[str]:
    with open(path, "rb") as fh:
        head = fh.read(1 << 16)
    found = set()
    for m in re.finditer(rb"(LAME\d\.\d{2,3}[a-z]?|Lavc\d+\.\d+|Lavf\d+\.\d+\.\d+|Xing|Info)", head):
        found.add(m.group(0).decode("latin-1"))
    return sorted(found)


def container_module(path: str, provenance: dict) -> ModuleResult:
    sniffed = provenance.get("sniffed_format", "unknown")
    ext = provenance.get("extension", "")
    diag = {
        "diag.container.sniffed_format": sniffed,
        "diag.container.extension": ext,
        "diag.container.extension_mismatch": bool(ext not in EXPECTED_EXT.get(sniffed, {ext})),
        "diag.container.format": provenance.get("container"),
        "diag.container.codec": provenance.get("codec"),
        "diag.container.native_sr": provenance.get("native_sr"),
        "diag.container.channels": provenance.get("channels"),
        "diag.container.bit_rate": provenance.get("bit_rate"),
        "diag.container.n_audio_streams": provenance.get("n_audio_streams"),
        "diag.container.tags": provenance.get("tags", {}),
    }
    if sniffed == "wav":
        rc = riff_chunks(path)
        diag["diag.container.riff_chunks"] = [c["id"] for c in rc["chunks"]]
        diag["diag.container.riff_info"] = rc["info"]
    if sniffed == "mp3":
        diag["diag.container.mp3_encoder_strings"] = mp3_encoder_strings(path)
    params = {"note": "diagnostic only; excluded from classification by design"}
    return ModuleResult(status=OK, diagnostics=diag, params=params)
