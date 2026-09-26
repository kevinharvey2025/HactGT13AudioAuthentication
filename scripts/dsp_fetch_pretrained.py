"""Download the official ASVspoof 2021 LA pretrained LFCC-GMM (MATLAB .mat) for the
"evaluate unchanged" experiment. Verifies the archive's SHA-256 before extracting.

Source: https://www.asvspoof.org/asvspoof2021/pre_trained_LA_LFCC-GMM.zip
(referenced by LA/Baseline-LFCC-GMM/matlab/LFCC_GMM_ASVspoof_2021_baseline.m, BSD-3-Clause).
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.asvspoof.org/asvspoof2021/pre_trained_LA_LFCC-GMM.zip"
SHA256 = "72126106b99bfb9cdeb46913ddba5dd914373eadbcdb6f42e03d15d652bd0da1"
OUT = Path(__file__).resolve().parents[1] / "cache/dsp/external"


def main() -> int:
    target = OUT / "pre_trained_LA_LFCC-GMM.mat"
    if target.exists():
        print(f"already present: {target}")
        return 0
    data = urllib.request.urlopen(URL, timeout=120).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"checksum mismatch: {digest} != {SHA256}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extract("pre_trained_LA_LFCC-GMM.mat", OUT)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
