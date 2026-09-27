"""provenance.json next to every generated result (docs/ROADMAP.md R6): the code version, the inputs by SHA-256, the
package versions, and the command, so a number in results/ can be traced to exactly what produced it."""
import hashlib
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

from . import config

PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "torch", "transformers", "cobweb")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def code_version():
    """The mpcdf snapshot name (<commit>[-dirty-<hash>]) on Raven, else the checkout's HEAD."""
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(config.REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return config.REPO.name


def write(out_dir, inputs=(), **extra):
    files = sorted({Path(p) for p in inputs if Path(p).is_file()})
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    rec = dict(code_version=code_version(), command=" ".join(sys.argv), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               python=platform.python_version(), packages=versions,
               inputs={str(p.relative_to(config.REPO) if p.is_relative_to(config.REPO) else p): sha256(p) for p in files}, **extra)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(Path(out_dir) / "provenance.json", "w"), indent=1)
    return rec
