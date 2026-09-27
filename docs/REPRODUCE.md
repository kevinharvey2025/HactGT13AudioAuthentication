# Reproducing everything

## 1. Predictions only (CPU or GPU)

```bash
conda env create -f environment.yml && conda activate hearsay      # Python 3.11 + ffmpeg + requirements.txt
python predict.py --input /path/to/audio --output out/ \
    --template /path/to/HGT_Hearsay_score_template.csv            # -> out/SideQuests_predictions_final.tsv, traces.jsonl
```

`predict.py` needs the weights in `artifacts/diffusion/`: `fusion.json` plus `xlsr2b_d6rall/`, `xlsr1b_d6rall/` and
`mms1b_d6rall/`, each holding a `best.pt`. They total about 7.7 GB, are produced by `mpcdf/final.sh`, and are not in
git.

Two precisions:
- **`--precision fp32`** (default, the reference) peaks at about 14.5 GB. On 8 CPU threads the NSA test set takes
  about 2 hours ([results/runtime.md](../results/runtime.md)).
- **`--precision bf16`** agrees with fp32 (100 NSA test clips: Spearman 0.998, every decision at P > 0.2
  identical, max |ΔP| 0.043) but is not a memory fix on CPUs without native bf16: on Raven's IceLake nodes it peaked
  at 13.2 GB (fp32: 14.4 GB) and took 1.6× as long. Budget about 15 GB either way.

Every file is scored whole and alone, so a score never depends on the other files in the folder.

### Docker

```bash
docker build -t sidequests-hearsay .                               # bakes artifacts/diffusion/ in if present
docker run --rm --network none --memory 16g -v /path/to/audio:/data/input:ro -v $PWD/out:/data/output sidequests-hearsay
# without baked weights: mount them (Docker Desktop: raise the VM's memory to >= 16 GB first)
docker run --rm --network none --memory 16g -v /path/to/weights:/app/artifacts/diffusion:ro \
    -v /path/to/audio:/data/input:ro -v $PWD/out:/data/output sidequests-hearsay
```

- **Pinned inputs:** the base image is pinned by digest, torch is the CPU wheel, and the other packages come from
  `requirements-docker.txt`.
- **Minimal context:** `.dockerignore` limits the build context to the files the image needs.
- **Offline:** the image runs with `--network none`.

**Verified on MPCDF Raven.** Docker does not run there, but Apptainer runs a saved Docker image unchanged:

```bash
docker buildx build --platform linux/amd64 -t sidequests-hearsay:exp1-amd64 --load .
docker save sidequests-hearsay:exp1-amd64 -o hearsay-amd64.tar                 # ~670 MB without weights
rsync hearsay-amd64.tar raven:/ptmp/$USER/hearsay/docker/
mpcdf.py submit raven mpcdf/run.sbatch -- neural "bash mpcdf/docker_verify.sh /ptmp/$USER/hearsay/docker/hearsay-amd64.tar"
```

`mpcdf/docker_verify.sh` does three things:
1. It converts the image to a SIF and runs the image's own entrypoint on 20 test clips, with the weights
   bind-mounted. `--pwd /app` honours the image's working directory and `--cleanenv` keeps the host environment out.
2. It scores the whole NSA test set with the image.
3. It compares the result with the reference CPU run and writes `verification.json` (PASS when the maximum absolute
   difference is ≤ 1e-4).

## 2. Tests

```bash
pytest tests --ignore=tests/dsp      # metric parity with the organizers' code (needs data/*HackGTMinDCF*), TSV, splits, TTCG, views
pytest tests/dsp                     # DSP track (requirements-dsp.txt)
```

`.github/workflows/tests.yml` runs both suites and a Docker build smoke test on every push. They need no data or
weights; the data-dependent tests skip there and run on Raven.

**Provenance.** Every generated results directory has a `provenance.json`: the code version, the SHA-256 of every
input, package versions, and the command.

The cobweb-private test runs only where the lab's COBWEB (revision 5012d51b or later) is installed. The view tests
need `soundfile` and ffmpeg.

## 3. The full pipeline on MPCDF Raven

Jobs run from immutable code snapshots made by the mpcdf helper. Everything stateful lives in
`/ptmp/$USER/hearsay/{data,cache,runs,artifacts}`, and `mpcdf/env.sh` links it into each snapshot. Submit with
`mpcdf.py submit raven <script> [-- args]`. In order:

| Step | Job | Output | Time |
|---|---|---|---|
| 0 | `bash mpcdf/setup_envs.sh all` (login node) | venv-neural, venv-dsp, venv-concepts | ~20 min |
| 1 | `mpcdf/setup_data.sbatch` | organizers' archives unpacked, LJSpeech, LibriSpeech (10 cloned speakers), metadata rebuilt | ~1 h |
| 2 | `mpcdf/forensics.sbatch` | the forensic audit (`runs/forensics/`) | ~1 h |
| 3 | `mpcdf/prep.sbatch -- neural dsp` (after `scripts/fetch_extra_reals.sh`) | 16 kHz cache, triage tables, test durations, DSP features | ~2 h |
| 4 | `mpcdf/gpu.sbatch -- "python scripts/run_d6r.py --vocoder hifigan_16k" ...`, then `run_d6r.py --merge` | 11,900 copy-synthesis fakes | ~1 h |
| 5 | `mpcdf/finetune.sbatch -- "--backbone adf_xlsr_2b --name xlsr2b_d6rall --epochs 3 --extra-fakes d6r --grad-ckpt --freeze-layers 24 --batch 16 --max-eval 2500 --shares lj_voice=0.2,clone=0.2,resynth=0.1,real_lj=0.2,real_libri=0.15,real_extra=0.15" "..."` | fine-tuned detectors, scored every epoch | 3-5 h per model |
| 6 | `mpcdf/gpu.sbatch -- "bash mpcdf/final.sh xlsr2b_d6rall@3 xlsr1b_d6rall@2 mms1b_d6rall@3"` | `fusion.json`, `artifacts/diffusion/`, the final TSV through `predict.py` | ~20 min |
| 6b | `mpcdf/run.sbatch -- neural "bash mpcdf/predict_cpu.sh data/hearsay_test <template> runs/diffusion/predict_final_v3 final 4"` | the same path on CPU in 4 shards (what the Docker image runs) | ~21 min |
| 7 | `mpcdf/concepts.sbatch` | concept formation + TTCG + their tests (`runs/diffusion/concepts/`) | ~1 h |
| 8 | `mpcdf/evaluate.sbatch` | unit tests + `scripts/evaluate.py` -> `runs/eval/` (copied to `results/`) | ~10 min |

Other runs (zero-shot, plain fine-tuning, WiSE-FT, seed replicate) use the same `finetune.sbatch` with
`--epochs 0`, without `--extra-fakes`, or with `--wise-from ... --wise-alpha a --epochs 0`. Exact per-run arguments
are stored in `runs/diffusion/ft/<run>/config.json`.

Raven notes:
- Whole 4-GPU nodes start far sooner than single-GPU jobs, so `finetune.sbatch` and `gpu.sbatch` run up to four
  commands, one per GPU.
- The login shell is zsh; the job scripts are bash.
- Compute nodes have no internet: model weights are fetched on the login node, into `HF_HOME`.

## 4. Environments

| Environment | Python | File | Used by |
|---|---|---|---|
| neural | 3.11 | `requirements.txt` | everything under `hearsay/` and `scripts/` except DSP and concepts |
| concepts | 3.11 (uv-managed, needs headers) | `requirements.txt` + cobweb-private (`pip install -e`, GCC 13, Eigen) | `scripts/run_concepts.py`, `tests/test_concepts.py` |
| dsp | 3.13 | `requirements-dsp.txt` | `hearsay_dsp/`, `scripts/dsp_*`, `tests/dsp/` |
| docker | 3.11-slim | `requirements-docker.txt` + CPU torch | `predict.py` |

cobweb-private is the Teachable AI Lab's private repository. It is never committed here (`.gitignore`); clone it
next to the workspace and install it with `pip install -e .`.
