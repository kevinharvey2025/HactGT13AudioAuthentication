# DSP run report: suite_v1

## Development ablations

Dev rows: 5797 (1573 real / 4224 synthetic); decode failures excluded: 0. All numbers are 5-fold grouped out-of-fold (OOF) on dev rows only.

| Configuration | AUC [95% CI] | EER [95% CI] | Balanced log loss | Genuine FPR @0.5 | Coverage | Module runtime (s/file) |
|---|---|---|---|---|---|---|
| gmm:official_repro(0-4k,c0,all,512c,10it) | 0.898 [0.828, 0.964] | 0.172 [0.122, 0.264] | 0.435 | 0.186 | 1.000 | n/a |
| gmm:official(0-4k,c0,nonsilent,64c) | 0.878 [0.816, 0.942] | 0.212 [0.145, 0.296] | 0.447 | 0.252 | 1.000 | n/a |
| gmm:custom(all,64c) | 0.920 [0.887, 0.953] | 0.192 [0.153, 0.217] | 0.366 | 0.222 | 1.000 | n/a |
| gmm:custom(nonsilent,64c) | 0.909 [0.867, 0.953] | 0.198 [0.153, 0.247] | 0.396 | 0.243 | 1.000 | n/a |
| gmm:custom(active,64c) | 0.888 [0.835, 0.940] | 0.218 [0.159, 0.279] | 0.435 | 0.269 | 1.000 | n/a |
| gmm:custom(nonsilent,16c) | 0.908 [0.875, 0.945] | 0.200 [0.154, 0.231] | 0.389 | 0.219 | 1.000 | n/a |
| gmm:custom(nonsilent,32c) | 0.906 [0.864, 0.949] | 0.201 [0.151, 0.249] | 0.400 | 0.238 | 1.000 | n/a |
| gmm:custom(nonsilent,128c) | 0.916 [0.871, 0.960] | 0.188 [0.143, 0.236] | 0.382 | 0.226 | 1.000 | n/a |
| lr:cep | 0.902 [0.856, 0.940] | 0.175 [0.133, 0.216] | 0.414 | 0.174 | 1.000 | 0.035 |
| lr:spec | 0.913 [0.861, 0.955] | 0.168 [0.113, 0.209] | 0.394 | 0.181 | 1.000 | 0.052 |
| lr:lpc | 0.829 [0.740, 0.920] | 0.237 [0.145, 0.333] | 0.532 | 0.245 | 1.000 | 0.759 |
| lr:bg | 0.731 [0.634, 0.799] | 0.308 [0.267, 0.363] | 0.821 | 0.217 | 1.000 | 0.006 |
| lr:phase | 0.773 [0.668, 0.856] | 0.272 [0.221, 0.340] | 0.659 | 0.285 | 1.000 | 0.035 |
| lr:pros | 0.812 [0.785, 0.839] | 0.257 [0.233, 0.287] | 0.536 | 0.275 | 1.000 | 0.095 |
| lr:all | 0.982 [0.966, 0.995] | 0.060 [0.029, 0.099] | 0.180 | 0.073 | 1.000 | 0.983 |
| lr:all-minus-cep | 0.949 [0.892, 0.989] | 0.103 [0.048, 0.183] | 0.339 | 0.116 | 1.000 | 0.948 |
| lr:all-minus-spec | 0.966 [0.929, 0.991] | 0.088 [0.043, 0.153] | 0.267 | 0.104 | 1.000 | 0.931 |
| lr:all-minus-lpc | 0.961 [0.914, 0.991] | 0.090 [0.044, 0.159] | 0.306 | 0.096 | 1.000 | 0.224 |
| lr:all-minus-bg | 0.980 [0.963, 0.994] | 0.067 [0.032, 0.100] | 0.189 | 0.081 | 1.000 | 0.977 |
| lr:all-minus-phase | 0.983 [0.967, 0.995] | 0.061 [0.035, 0.094] | 0.179 | 0.071 | 1.000 | 0.948 |
| lr:all-minus-pros | 0.981 [0.965, 0.995] | 0.063 [0.030, 0.101] | 0.184 | 0.078 | 1.000 | 0.888 |
| lr:core(cep+spec) | 0.961 [0.932, 0.983] | 0.104 [0.066, 0.142] | 0.263 | 0.113 | 1.000 | 0.087 |
| hgb:all | 0.980 [0.957, 0.998] | 0.086 [0.022, 0.130] | 0.206 | 0.112 | 1.000 | n/a |
| fusion:gmm:custom(all,64c)+lr:all | 0.982 [0.963, 0.996] | 0.064 [0.028, 0.104] | 0.201 | 0.074 | 1.000 | n/a |
| fusion:gmm:custom(all,64c)+lr:core(cep+spec) | 0.966 [0.940, 0.986] | 0.102 [0.063, 0.136] | 0.250 | 0.104 | 1.000 | n/a |

**Paired AUC deltas vs `lr:all` (same rows, group bootstrap 95% CI):**

| Configuration | ΔAUC | 95% CI |
|---|---|---|
| lr:cep | -0.0807 | [-0.1205, -0.0499] |
| lr:spec | -0.0691 | [-0.1177, -0.0372] |
| lr:lpc | -0.1532 | [-0.2298, -0.0747] |
| lr:bg | -0.2514 | [-0.3317, -0.1918] |
| lr:phase | -0.2089 | [-0.2974, -0.1349] |
| lr:pros | -0.1699 | [-0.2069, -0.1373] |
| lr:all-minus-cep | -0.0332 | [-0.0755, -0.0054] |
| lr:all-minus-spec | -0.0159 | [-0.0369, -0.0000] |
| lr:all-minus-lpc | -0.0214 | [-0.0509, -0.0030] |
| lr:all-minus-bg | -0.0028 | [-0.0074, -0.0002] |
| lr:all-minus-phase | +0.0002 | [-0.0017, 0.0021] |
| lr:all-minus-pros | -0.0008 | [-0.0045, 0.0015] |
| lr:core(cep+spec) | -0.0212 | [-0.0424, -0.0087] |
| hgb:all | -0.0026 | [-0.0097, 0.0039] |

paired_fusion_vs_lr_all: ΔAUC +0.0000, 95% CI [-0.0029, 0.0030]

paired_fusion_vs_gmm: ΔAUC +0.0625, 95% CI [0.0366, 0.0878]

**Per-generator AUC (all dev reals vs that generator's fakes):**

| Generator | gmm:custom(nonsilent,64c) | lr:all | lr:all-minus-cep | lr:all-minus-spec | lr:all-minus-lpc | lr:all-minus-bg | lr:all-minus-phase | lr:all-minus-pros | hgb:all | fusion:gmm:custom(all,64c)+lr:all | fusion:gmm:custom(all,64c)+lr:core(cep+spec) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diffgan_tts | 1.000 | 0.993 | 0.953 | 0.988 | 0.982 | 0.992 | 0.993 | 0.992 | 0.998 | 0.997 | 0.997 |
| elevenlabs | 0.823 | 0.960 | 0.902 | 0.930 | 0.922 | 0.954 | 0.957 | 0.957 | 0.958 | 0.955 | 0.904 |
| grad_tts | 0.600 | 0.964 | 0.923 | 0.930 | 0.929 | 0.960 | 0.964 | 0.960 | 0.930 | 0.950 | 0.880 |
| openvoicev2 | 0.991 | 0.975 | 0.942 | 0.970 | 0.948 | 0.971 | 0.975 | 0.973 | 0.982 | 0.976 | 0.957 |
| playht | 0.863 | 0.979 | 0.938 | 0.953 | 0.958 | 0.978 | 0.979 | 0.980 | 0.966 | 0.975 | 0.967 |
| pro_diff | 0.916 | 0.998 | 0.989 | 0.990 | 0.991 | 0.998 | 0.998 | 0.998 | 0.997 | 0.997 | 0.995 |
| unit_speech | 0.964 | 0.988 | 0.959 | 0.980 | 0.968 | 0.985 | 0.989 | 0.990 | 0.987 | 0.987 | 0.978 |
| wavegrad2 | 0.994 | 0.989 | 0.957 | 0.975 | 0.972 | 0.983 | 0.991 | 0.989 | 0.998 | 0.988 | 0.986 |
| xtts_v2 | 0.965 | 0.983 | 0.953 | 0.974 | 0.960 | 0.981 | 0.982 | 0.984 | 0.982 | 0.999 | 0.999 |
| your_tts | 0.999 | 0.991 | 0.970 | 0.971 | 0.974 | 0.990 | 0.993 | 0.989 | 0.998 | 1.000 | 1.000 |

**Leave-one-generator-out (generator's fakes excluded from all training):**

| Held-out generator | n fake | GMM AUC (seen) | LR AUC (seen) | Fusion AUC (seen) |
|---|---|---|---|---|
| diffgan_tts | 474 | 0.983 (1.000) | 0.986 (0.993) | 0.984 (0.997) |
| elevenlabs | 388 | 0.701 (0.831) | 0.587 (0.960) | 0.610 (0.955) |
| grad_tts | 474 | 0.541 (0.600) | 0.787 (0.964) | 0.701 (0.950) |
| openvoicev2 | 388 | 0.958 (0.995) | 0.869 (0.975) | 0.887 (0.976) |
| playht | 388 | 0.825 (0.882) | 0.940 (0.979) | 0.932 (0.975) |
| pro_diff | 474 | 0.666 (0.951) | 0.997 (0.998) | 0.996 (0.997) |
| unit_speech | 388 | 0.942 (0.971) | 0.977 (0.988) | 0.976 (0.987) |
| wavegrad2 | 474 | 0.741 (0.998) | 0.979 (0.989) | 0.971 (0.988) |
| xtts_v2 | 388 | 1.000 (1.000) | 0.945 (0.983) | 0.993 (0.999) |
| your_tts | 388 | 1.000 (1.000) | 0.989 (0.991) | 1.000 (1.000) |

**Shortcut audit (OOF AUC of excluded properties used alone):**

| Property set | Columns | OOF AUC |
|---|---|---|
| stream_properties | diag.decode.native_sr, diag.decode.channels, diag.decode.native_duration_s | 0.758 |
| levels_and_silence | diag.decode.rms_dbfs, diag.decode.peak_dbfs, q.act.digital_silence_frac, diag.act.floor_dbfs | 0.713 |
| bandwidth | diag.spec.native_occupied_bw_hz, diag.spec.native_edge_drop_db, q.spec.occupied_bw_hz | 0.836 |
| container_flags | diag.container.extension_mismatch | 0.546 |
| missingness_indicators | 13 indicator columns | 0.535 |

**Speaker / recording-chain audit (reals never seen by the DiffSSD-only model):**

DiffSSD-only training rows: 4417 (193 real).

| Real source | n | DiffSSD-only model: genuine FPR@0.5 (median p) | Extended model OOF: genuine FPR@0.5 (median p) |
|---|---|---|---|
| librispeech_train_clean_360 | 585 | 0.998 (1.000) | 0.176 (0.092) |
| ljspeech11_original_22k | 795 | 0.020 (0.000) | 0.008 (0.001) |

**Mean module runtime (s/file, measured on this machine under load):**

| Module | s/file |
|---|---|
| activity | 0.0089 |
| background | 0.0060 |
| compression | 0.0000 |
| container | 0.0003 |
| enf | 0.0100 |
| lfcc | 0.0354 |
| lpc | 0.7591 |
| phase | 0.0353 |
| prosody | 0.0954 |
| spectral | 0.0518 |
