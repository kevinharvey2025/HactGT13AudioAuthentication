# Runtime and memory of the shipped inference path (`predict.py`, CPU, fp32)

Measured on MPCDF Raven CPU nodes (2 x Intel Xeon IceLake-SP 8360Y, 72 cores, 256 GB) with the three-model final
ensemble. Each run includes loading the three models; per-clip times are approximate.

| Run | Clips | Threads | Wall time | Per clip | Peak memory (RSS) |
|---|---|---|---|---|---|
| NSA test set, 4 parallel processes (`mpcdf/predict_cpu.sh`) | 1,671 | 4 x 18 | 20 min 55 s | 0.75 s node throughput | 14.5-15.0 GB per process |
| In-the-Wild, 6 parallel processes | 4,000 | 6 x 12 | 57 min 42 s | 0.87 s node throughput | 15.3-15.5 GB per process |
| 100 test clips, one process ("laptop-like") | 100 | 8 | 8 min 03 s | ~4.2 s after loading | 14.4 GB |
| 100 test clips, earlier version (all models resident, batch-cropped) | 100 | 8 | 6 min 22 s | ~3.5 s | 22.4 GB |
| 100 test clips, earlier version | 100 | 72 | 2 min 23 s | ~1.0 s | 22.4 GB |

- **Whole NSA test set on 8 CPU threads:** about 2 hours. Give Docker at least 16 GB of memory.
- **Order independence:** scoring 100 test clips alone and as part of all 1,671 differs by at most 2.0e-5 in
  P(synthetic).
- **The earlier version** scored batches cropped to their shortest clip, so its scores depended on the other clips
  in the folder (up to 0.18 difference on the same 100 clips).
