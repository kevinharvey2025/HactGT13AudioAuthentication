# Shortcut and leakage checks (plan section 13)

Grouped 5-fold CV AUC of a small gradient-boosting model on non-content features. 0.5 = no information. *raw* = decoded clip as delivered; *canonical* = the view every model in this track sees (trim, test-like crop, 7 kHz low-pass, peak-norm, dither).

| group                                 | view      | protocol   |   auc |   eer |
|:--------------------------------------|:----------|:-----------|------:|------:|
| container (sr, codec, bitrate, tags)  | raw       | text-CV    | 0.897 | 0.212 |
| container (sr, codec, bitrate, tags)  | raw       | speaker-CV | 0.899 | 0.206 |
| filesystem MAC times                  | raw       | text-CV    | 1     | 0     |
| filesystem MAC times                  | raw       | speaker-CV | 1     | 0     |
| duration                              | raw       | text-CV    | 0.696 | 0.362 |
| duration                              | raw       | speaker-CV | 0.697 | 0.36  |
| duration                              | canonical | text-CV    | 0.528 | 0.483 |
| duration                              | canonical | speaker-CV | 0.529 | 0.483 |
| level (peak, RMS, clipping, DC)       | raw       | text-CV    | 0.959 | 0.097 |
| level (peak, RMS, clipping, DC)       | raw       | speaker-CV | 0.91  | 0.159 |
| level (peak, RMS, clipping, DC)       | canonical | text-CV    | 0.834 | 0.246 |
| level (peak, RMS, clipping, DC)       | canonical | speaker-CV | 0.793 | 0.266 |
| edge silence + digital zeros          | raw       | text-CV    | 0.948 | 0.12  |
| edge silence + digital zeros          | raw       | speaker-CV | 0.923 | 0.149 |
| edge silence + digital zeros          | canonical | text-CV    | 0.659 | 0.383 |
| edge silence + digital zeros          | canonical | speaker-CV | 0.629 | 0.401 |
| bandwidth (cutoff, HF drop, HF ratio) | raw       | text-CV    | 0.811 | 0.271 |
| bandwidth (cutoff, HF drop, HF ratio) | raw       | speaker-CV | 0.773 | 0.293 |
| bandwidth (cutoff, HF drop, HF ratio) | canonical | text-CV    | 0.668 | 0.379 |
| bandwidth (cutoff, HF drop, HF ratio) | canonical | speaker-CV | 0.643 | 0.391 |
| ALL signal-trivial features           | canonical | speaker-CV | 0.85  | 0.227 |

## Raw clips by generator (medians)

| generator   |   native_sr | codec     | encoder       |    dur |   peak |   rms_dbfs |   lead_sil |   trail_sil |   zero_run_s |   hf_7k_drop_db |   cutoff_hz |
|:------------|------------:|:----------|:--------------|-------:|-------:|-----------:|-----------:|------------:|-------------:|----------------:|------------:|
| diffgan_tts |       22050 | pcm_s16le | -             |  5.851 |  0.776 |    -19.971 |       0    |        0.04 |        0     |           3.843 |     8000    |
| elevenlabs  |       44100 | mp3       | -             |  7.419 |  0.533 |    -25.028 |       0.16 |        0.36 |        0.018 |           8.597 |     8000    |
| grad_tts    |       22050 | pcm_s16le | -             |  6.014 |  0.516 |    -24.112 |       0.02 |        0.08 |        0     |           2.619 |     8000    |
| openvoicev2 |       22050 | pcm_s16le | -             |  7.454 |  0.435 |    -25.66  |       0.16 |        0.26 |        0     |          35.143 |     7625    |
| playht      |       24000 | mp3       | Lavf58.29.100 |  7.445 |  0.652 |    -24.274 |       0.02 |        0.16 |        0.002 |           7.088 |     8000    |
| pro_diff    |       22050 | pcm_s16le | -             |  6.142 |  0.467 |    -24.866 |       0    |        0.1  |        0     |           8.045 |     8000    |
| real_libri  |       16000 | flac      | -             | 14     |  0.536 |    -24.347 |       0.2  |        0.18 |        0     |           2.531 |     8000    |
| real_lj     |       22050 | pcm_s16le | -             |  6.938 |  0.547 |    -23.924 |       0    |        0.08 |        0     |           3.804 |     8000    |
| real_lj_nsa |       16000 | pcm_s16le | Lavf58.29.100 |  7.095 |  0.537 |    -23.991 |       0    |        0.08 |        0     |           3.721 |     8000    |
| unit_speech |       22050 | pcm_s16le | -             |  7.732 |  0.379 |    -26.756 |       0.04 |        0.08 |        0     |          14.447 |     8000    |
| unknown     |       16000 | pcm_s16le | Lavf58.29.100 |  3.413 |  0.998 |    -18.072 |       0    |        0    |        0     |          43.511 |     7484.38 |
| wavegrad2   |       22050 | pcm_s16le | -             |  6.286 |  0.802 |    -21.242 |       0    |        0.34 |        0     |           0.986 |     8000    |
| xtts_v2     |       24000 | pcm_s16le | -             |  7.862 |  0.999 |    -17.189 |       0    |        0.56 |        0.417 |           6.727 |     8000    |
| your_tts    |       16000 | pcm_s16le | -             |  9.026 |  1     |    -18.539 |       0.02 |        0.68 |        0.625 |           6.14  |     8000    |

## Canonical view by generator (medians)

| generator   |   decoded_duration |   peak |   rms_dbfs |   lead_sil |   trail_sil |   zero_run_s |   hf_7k_drop_db |   cutoff_hz |
|:------------|-------------------:|-------:|-----------:|-----------:|------------:|-------------:|----------------:|------------:|
| diffgan_tts |              3.39  |  0.985 |    -16.9   |          0 |           0 |            0 |          71.266 |     7062.5  |
| elevenlabs  |              3.437 |  0.985 |    -18.146 |          0 |           0 |            0 |          60.022 |     7046.88 |
| grad_tts    |              3.413 |  0.985 |    -17.672 |          0 |           0 |            0 |          71.52  |     7062.5  |
| openvoicev2 |              3.436 |  0.985 |    -15.946 |          0 |           0 |            0 |          65.405 |     7046.88 |
| playht      |              3.424 |  0.986 |    -19.464 |          0 |           0 |            0 |          59.433 |     7046.88 |
| pro_diff    |              3.39  |  0.986 |    -17.306 |          0 |           0 |            0 |          70.622 |     7062.5  |
| real_libri  |              3.413 |  0.985 |    -17.513 |          0 |           0 |            0 |          60.804 |     7046.88 |
| real_lj     |              3.367 |  0.985 |    -17.848 |          0 |           0 |            0 |          71.047 |     7062.5  |
| real_lj_nsa |              3.413 |  0.986 |    -17.896 |          0 |           0 |            0 |          70.727 |     7062.5  |
| unit_speech |              3.413 |  0.985 |    -17.758 |          0 |           0 |            0 |          61.775 |     7046.88 |
| unknown     |              3.204 |  0.985 |    -17.856 |          0 |           0 |            0 |          58.429 |     7046.88 |
| wavegrad2   |              3.39  |  0.984 |    -18.189 |          0 |           0 |            0 |          67.157 |     7062.5  |
| xtts_v2     |              3.437 |  0.984 |    -16.852 |          0 |           0 |            0 |          61.947 |     7046.88 |
| your_tts    |              3.437 |  0.984 |    -17.523 |          0 |           0 |            0 |          66.029 |     7062.5  |
