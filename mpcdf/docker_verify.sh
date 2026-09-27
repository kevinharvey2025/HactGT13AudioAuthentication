#!/bin/bash
# Verify the Docker image on Raven. Docker does not run there, but Apptainer runs a saved Docker image unchanged.
#   (laptop)  docker buildx build --platform linux/amd64 -t sidequests-hearsay:TAG --load . && docker save ... -o image.tar
#   (Raven)   mpcdf.py submit raven mpcdf/run.sbatch -- neural "bash mpcdf/docker_verify.sh /ptmp/$USER/hearsay/docker/image.tar"
# 1. convert the image to a SIF; 2. run the image's own ENTRYPOINT on 20 NSA test clips; 3. score the whole NSA test
# set with the image (N shards) and compare with the reference CPU run (runs/diffusion/predict_final_v3).
# The weights are bind-mounted at /app/artifacts/diffusion; --pwd /app honours the image's WORKDIR and --cleanenv keeps
# the host's Python environment out, so the code and packages that run are the image's own.
set -eo pipefail
TAR=$1; OUT=${2:-runs/docker_verify}; N=${3:-4}; REF=${4:-runs/diffusion/predict_final_v3/SideQuests_predictions_final.tsv}
module load apptainer/1.5.2
mkdir -p "$OUT"
OUT=$(readlink -f "$OUT"); W=$(readlink -f artifacts/diffusion); T=$(readlink -f data/hearsay_test)
SIF=$OUT/hearsay.sif
echo "=== build SIF from $TAR"
[ "$SIF" -nt "$TAR" ] || apptainer build --force "$SIF" "docker-archive://$TAR" 2>&1 | tail -2
THREADS=$(( ${SLURM_CPUS_PER_TASK:-72} / N ))

echo "=== 2. the image's ENTRYPOINT on 20 clips"
rm -rf "$OUT/smoke_in" "$OUT/smoke_out" && mkdir -p "$OUT/smoke_in" "$OUT/smoke_out"
wavs=("$T"/*.wav)                               # (a pipe through head would end in SIGPIPE under pipefail)
cp "${wavs[@]:0:20}" "$OUT/smoke_in/"
apptainer run --cleanenv --pwd /app --env OMP_NUM_THREADS=$THREADS \
    --bind "$W:/app/artifacts/diffusion:ro" --bind "$OUT/smoke_in:/data/input:ro" --bind "$OUT/smoke_out:/data/output" "$SIF" 2>&1 | tail -2
ls "$OUT/smoke_out"

echo "=== 3. the NSA test set with the image, $N shards"
python - "$T/HGT_Hearsay_score_template.csv" "$OUT" "$N" <<'PY'
import sys, pandas as pd
tpl, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = pd.read_csv(tpl, sep="\t")
import os; os.makedirs(f"{out}/shards", exist_ok=True)
for i in range(n):
    t.iloc[i::n].to_csv(f"{out}/shards/template_{i}.tsv", sep="\t", index=False)
PY
pids=()
for i in $(seq 0 $((N - 1))); do
  /usr/bin/time -v apptainer exec --cleanenv --pwd /app --env OMP_NUM_THREADS=$THREADS \
      --bind "$W:/app/artifacts/diffusion:ro" --bind "$T:/data/input:ro" --bind "$OUT:/data/output" "$SIF" \
      python predict.py --input /data/input --output /data/output/shards/$i --template /data/output/shards/template_$i.tsv \
      --device cpu > "$OUT/shards/log_$i.txt" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
grep -h "Elapsed\|Maximum resident" "$OUT"/shards/log_*.txt

IMAGE_ID=$(cat "$(dirname "$TAR")/image_id.txt" 2>/dev/null || echo unknown)
python - "$T/HGT_Hearsay_score_template.csv" "$OUT" "$N" "$REF" "$IMAGE_ID" <<'PY'
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from hearsay import submission
tpl, out, n, ref, image = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
name = "SideQuests_predictions_final.tsv"
parts = pd.concat([pd.read_csv(out / "shards" / str(i) / name, sep="\t") for i in range(n)])
submission.write(parts.set_index("filename")["cm-score"], out / name, template=tpl)
a = pd.read_csv(out / name, sep="\t").set_index("filename")["cm-score"]
b = pd.read_csv(ref, sep="\t").set_index("filename")["cm-score"].loc[a.index]
res = dict(rows=len(a), max_abs_diff=float((a - b).abs().max()), spearman=float(a.corr(b, method="spearman")),
           decisions_agree_at_0_2=float(((a > 0.2) == (b > 0.2)).mean()), reference=ref, image=image,
           verdict="PASS" if (a - b).abs().max() <= 1e-4 else "CHECK")
json.dump(res, open(out / "verification.json", "w"), indent=1)
print("docker image vs reference:", res)
PY
