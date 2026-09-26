import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hearsay_dsp.config import load_config  # noqa: E402

from dsp_testutils import SR  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"run": {"cache_dir": str(tmp_path / "cache"), "n_jobs": 1}})


@pytest.fixture
def write_wav(tmp_path):
    def _w(x, sr=SR, name="x.wav", subtype="PCM_16"):
        p = tmp_path / name
        sf.write(p, x, sr, subtype=subtype)
        return str(p)
    return _w

