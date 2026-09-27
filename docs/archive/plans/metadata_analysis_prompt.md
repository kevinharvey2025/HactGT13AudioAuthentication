# HEARSAY — Metadata analysis alone and combined with DSP

Implementation prompt and working specification for Claude. Build a reproducible experimental harness to evaluate **metadata/file-structure analysis independently**, **signal-processing analysis independently**, and **their combination** for the HackGT 13 HEARSAY Audio Authentication Challenge.

**Status:** implementation specification; no performance results established.  
**Prepared:** September 26, 2026.  
**Sources:** HEARSAY challenge brief, team requirements, the existing DSP specification, and primary references in section 20.

**Your assignment:** inspect the repository, implement working code, run meaningful tests and available experiments, and document actual results. Preserve existing work. Reuse an existing DSP pipeline when it meets this specification; do not replace it or change its baseline silently.

**Main research question:** does metadata add reliable information beyond acoustic signal processing, or does it mainly reveal collection/export shortcuts that disappear when files are retagged or re-encoded?

**Scope:** use deterministic file analysis, engineered DSP features, and classical statistical classifiers. No neural speech models, learned speech embeddings, neural VAD, diffusion, ASR, LLM scoring, or generative audio. Do not incorporate AASIST or AntiDeepfake scores in this experiment.

---

## 1. Challenge context and output

### 1.1 Official requirements relevant to this task

- Accept common audio formats: WAV, MP3, M4A, MP4 audio, OGG, and others supported by the decoder.
- Clips are in English and at least two seconds long.
- Predict real/bonafide versus synthetic/spoof.
- Output a score in [0,1], with **1 = synthetic**.
- Genuine recordings may contain noise and perturbations.
- Potential conditions include synthesis, voice conversion, replay, partial edits, manipulated backgrounds, transcoding, and metadata spoofing.
- Explain which techniques contribute and which have no measurable effect.
- Deliver source code, documentation, a reproducible Docker image, and a prediction TSV.

```tsv
filename	cm-score
example.wav	0.72
```

The row illustrates formatting only, not a measured prediction.

### 1.2 Scoring context

| Weight | Category | Contribution of this implementation |
|---|---|---|
| 60% | Detection performance | Reliable statistical prediction and calibrated scores |
| 20% | Forensic diversity | Metadata, structure, DSP, and measured cross-checks |
| 20% | Documentation/presentation | Reproducibility, explanations, ablations, and robustness results |

Do not assume that a technique receives credit merely because it exists in code. Distinguish modules actually used by the classifier from diagnostic-only modules. Do not promise an agentic bonus for a deterministic router.

### 1.3 Unverified team context

- Approximately 77,000 test files and a prefilled submission TSV were mentioned in team notes. Verify actual counts and paths.
- If supplied, use the validated reference TSV as the expected filename list and preserve its order.
- Team name, deadlines, exact official metric, external-data permissions, and some attack-label semantics are unresolved.
- Do not contact organizers, upload audio, publish repositories/images, or submit predictions on our behalf. Prepare local artifacts and questions.

---

## 2. Experiments and hypotheses

### 2.1 Hypotheses

- **H1 — Metadata signal:** technical metadata can discriminate classes on held-out sources, beyond chance and class prevalence.
- **H2 — Editable-tag dependence:** encoder/date/tag fields may improve random-split results but lose value under tag removal or replacement.
- **H3 — Structural signal:** container/bitstream structure may retain information beyond editable tags, but can still identify export pipelines rather than authenticity.
- **H4 — Complementarity:** metadata improves DSP predictions because the branches make meaningfully different errors.
- **H5 — Cross-check value:** comparisons between file claims and measured signal properties add information beyond either branch alone.

These are testable hypotheses, not conclusions. Negative findings belong in the final report.

### 2.2 Required experiment matrix

| ID | Inputs | Purpose | Priority |
|---|---|---|---|
| M0 | Technical metadata | Clean metadata-only baseline | Required |
| M1 | Technical metadata + editable tags | Measure tag dependence | Required |
| M2 | Technical metadata + format-aware consistency features | Test internal consistency | Required |
| D0 | Existing/default DSP baseline | Independent acoustic reference | Required |
| F0 | Out-of-fold M0 and D0 scores | Minimal late fusion | Required |
| F1 | Selected metadata and D0 scores | Test richer late fusion | Required |
| F2 | Selected metadata features + DSP features | Feature-level fusion | Required when D0 exposes features |
| X0 | Best validated fusion + cross-modal consistency features | Test explicit cross-checks | Required if inputs exist |
| M3 | Structural fingerprints | Deeper container analysis | Optional, gated |

Also report M0 with technical-consistency features removed when those overlap substantially with M2. Keep actual feature memberships explicit.

Use the same split definitions and paired evaluation examples across experiments. Do not pick a final configuration using challenge test labels or the reserved local final evaluation split.

---

## 3. Evidence taxonomy and input boundaries

### 3.1 Separate feature namespaces

| Namespace | Examples | Permitted use |
|---|---|---|
| `meta.technical` | Parsed codec, rate, channels, PCM depth, bitrate mode | Metadata baseline |
| `meta.tags` | Encoder/software family, tag presence, coarse timestamp relations | Explicit tag-sensitive experiment |
| `meta.structure` | RIFF chunks, MP4 boxes, packet summaries | Optional structural experiment |
| `meta.consistency` | Internally inconsistent headers, conflicting tag fields | Metadata consistency experiment |
| `dsp` | LFCC, spectral, residual, background-change features | DSP baseline |
| `cross` | Declared properties versus decoded/signal measurements | Combined experiments only |
| `diagnostic` | Paths, file IDs, local timestamps, parser errors | Audit/logging only by default |

Duration and file size can be strong collection shortcuts. Isolate them as explicit feature groups, report their ablations, and distinguish container-reported duration from decoded duration.

### 3.2 Do not put these into classifiers

- Filenames, paths, row order, arbitrary file IDs, hashes, or directory names.
- Local filesystem creation/modification/access times.
- Full free-text comments, titles, usernames, or other high-cardinality identity fields.
- Parser error strings or tool-specific messages.
- Test labels or metadata derived from ground-truth manifests.
- DSP measurements inside an experiment described as metadata-only.

Use an allowlist for each experiment. Automatically check namespace membership before fitting or inference.

### 3.3 Metadata is a claim, not ground truth

- Editable tags can be absent, stale, copied, or spoofed.
- Structural fields describe the current representation, not necessarily original capture.
- File-copy timestamps often describe dataset handling.
- Missing metadata is not intrinsically suspicious.
- Encoder identity is not generator identity.
- A transcode or edit does not automatically imply synthetic speech.

---

## 4. Phase 0: repository, environment, and data audit

- Read repository instructions and inspect existing pipelines/configuration.
- Locate any previous DSP specification and implementation. Preserve existing feature definitions and model artifacts.
- Inspect CPU/RAM/disk, Python, FFmpeg/ffprobe, ExifTool, and dependencies.
- Create an isolated reproducible environment; pin tested versions.
- Validate labels, paths, duplicates, formats, class counts, and duration distributions.
- Detect exact duplicate content. Keep original recordings and all derivatives in the same group.
- Compare technical train/test distributions descriptively; do not fit imputation, category vocabularies, scaling, or classifiers on challenge test data.
- If data is unavailable, implement the harness and fixtures, then identify the blocked real-data steps honestly.

### Manifest

| Field | Meaning |
|---|---|
| `path` | Required audio path, relative to manifest when not absolute |
| `label` | Real=0 or synthetic=1 through explicit mapping |
| `split` | Optional user-provided split designation |
| `group_id` | Original recording and derivatives |
| `speaker_id` | Optional grouping |
| `source_id` | Optional recording source/dataset |
| `pipeline_id` | Optional known export/collection pipeline |
| `attack_type` | Optional evaluation category |

Do not infer class labels from filenames or directories without explicit mapping. Reject conflicting labels for exact duplicate recordings and report them for resolution.

---

## 5. Metadata extraction

### 5.1 Tools and provenance

Use ffprobe JSON output for format/stream information. Use ExifTool structured output for additional tags. Add MediaInfo only if it supplies justified information unavailable elsewhere.

Record each field with tool, version, source namespace, raw value, normalized value, units, and availability. Keep similarly named container, track, and filesystem dates distinct.

Run parsers read-only with subprocess argument arrays, timeouts, bounded output, and resource-aware concurrency. Do not interpolate filenames into shell commands. Preserve originals and never invoke metadata-writing operations during extraction.

### 5.2 Technical features

Extract when meaningful:

- Container and codec family/profile.
- Parsed sample rate and channel count/layout.
- PCM bit depth or codec-specific precision where defined.
- Stream/container bitrate and whether reported or estimated.
- Bitrate mode where reliably available.
- Container/stream-reported duration.
- Stream count and presence of non-audio streams.
- Available codec configuration and packet-summary features.

Do not treat sample storage format as original recording bit depth. Do not calculate uncompressed-PCM consistency formulas for lossy codecs.

### 5.3 Editable-tag features

For M1, use a bounded normalized encoder/software-family vocabulary, tag-presence indicators, and carefully defined within-file timestamp relationships.

- Learn categories using training data only.
- Group rare categories into `other`; distinguish unseen from missing.
- Avoid absolute timestamps by default. Test them only as a separately labeled shortcut audit, not a candidate production feature.
- Preserve timezone uncertainty; do not invent timezone corrections.
- Avoid free-text identity fields in training.

### 5.4 Missing and failed extraction

Differentiate `present`, `absent`, `unsupported`, `parse_error`, and `timeout`. A missing optional tag is normal; a failed parser is an infrastructure condition.

Use explicit policies for numerical and categorical missingness. Do not encode missing values as zero unless zero has the intended semantic meaning. Evaluate whether missingness itself becomes a source shortcut.

---

## 6. Metadata consistency analysis

Implement format-aware checks rather than generic “suspicious file” rules.

| Check | Interpretation and limits |
|---|---|
| File signature versus extension | Naming/packaging discrepancy; not proof of forgery |
| PCM header byte rate/block alignment | Structural consistency only for applicable PCM formats |
| Container versus stream duration | Requires format-specific tolerance |
| Conflicting embedded tags | May reflect software history or stale metadata |
| Packet timestamp discontinuities | Can reflect edit lists, delay, or valid container behavior |
| Container/codec combination | Validate against actual format support; avoid invented compatibility rules |

Decode-based duration comparisons belong in `cross`, not pure metadata experiments. Extension/signature mismatch is diagnostic by default because filenames are excluded from predictive inputs; any experiment using it must be separately labeled and stress-tested against renaming.

Do not flag creation-after-modification as universally impossible. Do not infer a capture device's authenticity from assumed defaults without validated reference evidence.

Each check returns applicability, measured discrepancy, tolerance, and a conservative explanation. Parser disagreement alone is not evidence of synthetic audio.

---

## 7. Optional structural fingerprints and provenance

### 7.1 Structural fingerprints

Inspect RIFF chunk presence/order, MP4 box summaries, padding/alignment, and codec/packet patterns where feasible. Use format-specific parsers and documented normalized features.

Do not train directly on raw arbitrary header strings or identifiers. Control feature count and evaluate on held-out pipelines. Remuxing and re-encoding can replace structural fingerprints.

If using clustering, fit it on training data only. Describe clusters as production-pattern groups, not synthetic-generator families unless independently labeled and validated.

### 7.2 Signed provenance

Optionally inspect a sample for supported C2PA credentials. If absent, document and defer.

If implemented, use a real validator and distinguish: absent, valid binding/signature, invalid, unsupported, and untrusted signer. Record asserted provenance separately from cryptographic validation.

Valid signed claims do not establish factual authenticity, and missing credentials are neutral. Do not simulate verification by checking whether a metadata tag contains “C2PA.” Keep this diagnostic until there is enough relevant data to assess it.

---

## 8. Independent DSP baseline

### 8.1 Reuse first

If an existing DSP baseline is present, run it unchanged as D0 and record its exact configuration and artifact identity. Any changes become separately named experiments.

If absent, implement a minimum D0:

- LFCC extraction using documented framing/filterbank/DCT settings.
- Two GMMs fitted on real and synthetic training frames.
- Clip score = average synthetic log likelihood minus average real log likelihood.
- Optional compact engineered-feature logistic baseline if needed for feature-level fusion.

Positive raw GMM score means synthetic. Verify upstream polarity rather than blindly flipping it.

### 8.2 Acoustic feature groups

Use explicitly engineered features such as LFCC summaries, spectral flatness/flux, band-energy ratios, LPC residual statistics, and validated background/discontinuity measurements.

Optional classical pitch/phase/ENF modules must abstain when evidence is insufficient. No neural VAD or pretrained embeddings. Do not expand optional DSP scope at the expense of the required metadata comparisons.

### 8.3 Input handling

- Preserve original files.
- Decode reliably with an FFmpeg fallback.
- Record stream choice, channel policy, rate, normalization, and windows.
- Preserve native-rate copies for relevant measurements.
- Do not automatically denoise, trim silence, or normalize loudness.
- Match the existing DSP recipe; 16 kHz is a working rate, not a requirement for all modules.
- Handle silence, short clips, corrupt input, and unavailable frequency bands explicitly.

Metadata branch features must not leak into D0. Metadata may select a decoder but must not alter D0's classification logic.

---

## 9. Cross-modal consistency features

Implement these only for combined experiments:

| Comparison | Permitted observation |
|---|---|
| Declared rate versus measured occupied bandwidth | High-rate representation of narrowband content |
| Declared duration versus decoded samples | Measured duration discrepancy |
| Multiple channels versus channel correlation | Little independent channel variation |
| Encoding properties versus signal bandwidth | Current representation preserves a limited-band signal |

None establishes an exact earlier sample rate, codec chain, replay event, or generator. Genuine telephony saved in a high-rate container is a valid negative control.

Keep raw measurements, eligibility, and derived interaction features. Avoid counting the same measurement as independently corroborating evidence in metadata, DSP, and cross-check explanations.

The cross-check experiment must be compared with a fusion model that already has its constituent inputs when possible. This tests whether the explicit interaction helps rather than merely adding a previously absent feature.

---

## 10. Classifiers, late fusion, and feature-level fusion

### 10.1 Metadata classifiers

Start with regularized logistic regression using training-only numerical preprocessing and categorical encoding. Unknown categories must not crash inference.

Compare with histogram gradient boosting only through an encoding/imputation pipeline it actually supports. Bound categorical dimensionality; do not blindly densify huge one-hot matrices.

Use a small, declared hyperparameter search. Do not choose a more complex model solely for better training performance.

### 10.2 Late fusion

Fit a small regularized logistic model over metadata and DSP branch scores. Prefer raw decision scores when available and maintain consistent score definitions.

Generate branch predictions out-of-fold for fusion training: the entire base pipeline, including encoder vocabulary, imputer, scaler, feature selection, and GMM/classifier, must be fitted without the held-out fold.

Train fusion on those out-of-fold scores, then refit the selected base models on their permitted training pool. Never fit fusion on in-sample base-model predictions and call that unbiased evidence.

### 10.3 Feature-level fusion

Concatenate allowlisted metadata and DSP feature groups. Fit preprocessing and classifier on training data only. Compare linear fusion first, then an optional tree-based variant.

If D0 exposes only a score, do not label score-plus-metadata as full feature-level fusion. Implement/document a separate DSP feature representation or mark F2 unavailable with the reason.

### 10.4 Missing branch behavior

Handle optional missing tags normally. For a completely failed branch, report the failure and use only a predeclared, validated fallback model if explicitly configured. Never silently substitute 0.5, zero, or a class label.

Evaluate metadata dropout as an optional training augmentation on training folds only. At inference, the selected combined system should have a documented behavior for absent or unseen metadata.

---

## 11. Splits, calibration, and leakage control

Use a fixed split manifest shared across experiments:

1. Training/fold pool for model fitting and out-of-fold fusion.
2. Development set for configuration/feature selection.
3. Calibration set for the selected configurations.
4. Reserved local evaluation set for final reporting.

For small datasets, implement a documented nested/grouped cross-validation alternative. Do not create unreliable tiny calibration or evaluation sets merely to match a diagram.

Group originals and derivatives together. Hold out source/pipeline/speaker groups when available. A random split can be reported as a secondary diagnostic, not the sole evidence of robustness.

Fit imputation, vocabularies, scaling, selection, classifiers, and calibrators only on their permitted split. Store split membership/provenance with every artifact.

Use sigmoid calibration first; isotonic is optional only with enough data. Report both raw ranking performance and calibrated probability metrics. Apply comparable calibration procedures to standalone and combined models so fusion gains are not merely calibration gains.

No calibration, category learning, threshold optimization, or feature selection on the challenge test set. EER-derived evaluation thresholds remain descriptive and must not become deployment thresholds.

---

## 12. Robustness experiments: metadata edits versus signal edits

### 12.1 Metadata-only interventions

Create copies; never modify originals. Test format-supported operations:

1. Remove editable descriptive tags.
2. Replace encoder/software/date tags with class-independent controlled values.
3. Add plausible neutral tags to files missing them.
4. Rename files or change extensions without changing bytes, as a diagnostic.
5. Remux without re-encoding where supported.

Record tools, arguments, changed fields, and provenance. Do not assume a tool changed only metadata.

Verify payload or decoded-sample identity using a fixed decoder and documented stream/alignment policy. For each intervention record whether audio equivalence passed, failed, or could not be established. If remuxing changes presentation timing or decoded samples, do not label it a metadata-only invariance test.

Expectation: D0 should produce matching scores within numerical tolerance for identical effective audio input. Report metadata/fusion score changes, label flips at the prespecified threshold, and changes in ranking/calibration metrics.

### 12.2 Signal-changing interventions

Separately test matched re-encoding, resampling, band-limiting, and added noise for both classes. Record codec/settings/SNR and actual resulting properties.

These change the signal and may also change metadata. Report them as channel robustness experiments, not pure metadata manipulations.

Reserve some settings for evaluation. Do not tune every transformation against the final evaluation set. Keep all derivatives in their original group.

### 12.3 Unknown-pipeline tests

Hold out known export/source pipelines when possible. Evaluate unseen encoder categories and missing tags. Do not claim pipeline generalization when pipeline labels are unavailable; state the limitation.

---

## 13. Evaluation and attribution

### Metrics

- ROC-AUC, average precision, EER with documented convention.
- Log loss, Brier score, reliability plot.
- Confusion matrix, sensitivity, specificity, precision, F1, accuracy at a prespecified threshold, initially 0.5 for calibrated scores.
- Coverage, branch failures, fallback rate if configured.
- Runtime, throughput, memory, and extraction cost.
- Score shifts and prediction flips under verified metadata-only edits.

Handle single-class/undefined subsets explicitly. Show counts and class prevalence. Compare systems on the same successfully scored files, while reporting complete coverage separately.

Break down by source, codec, duration, available attack type, and robustness condition. Optional confidence intervals should bootstrap original recording groups, not individual augmented derivatives.

### Required ablations

- Technical metadata versus technical plus tags.
- Remove duration/file size; remove encoder family; remove timestamp features; remove missingness indicators.
- Metadata versus DSP versus late fusion versus feature fusion.
- Combined model with and without cross-checks.
- Remove structural fingerprints if implemented.
- Original versus stripped/spoofed tags and matched channel transforms.

Use held-out permutation importance or model-appropriate attribution to inspect feature reliance. Correlated features complicate attribution; importance is not physical causation. Quantify incremental performance, not just a large feature-importance value.

### Comparison table

| System | Feature groups | AUC | EER | Log loss | Brier | Genuine FPR | Coverage | Runtime |
|---|---|---|---|---|---|---|---|---|
| M0 | Technical | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| M1 | Technical + tags | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| M2 | Technical + consistency | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D0 | DSP | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| F0 | M0 + D0 scores | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| F1 | Selected metadata + D0 scores | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| F2 | Metadata + DSP features | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| X0 | Selected fusion + cross-checks | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Never fill the table with invented numbers. Use `not run` and a reason when appropriate.

---

## 14. Selection gates and conclusions

Select configurations using development data before opening the reserved evaluation results. Freeze code/configuration/artifacts for final reporting.

Metadata earns a role when it improves relevant development performance and its contribution remains useful under source/pipeline holdouts and metadata interventions. Define acceptable robustness/latency tradeoffs before final evaluation; do not invent universal improvement thresholds.

Possible valid conclusions include:

- Metadata helps clean data but is too brittle under retagging.
- Technical consistency helps while encoder tags add only shortcuts.
- DSP already captures the useful information; fusion has no measurable benefit.
- Cross-checks help a specific condition but not overall detection.
- A DSP-only model is the best validated final choice.

If final evaluation disappoints, report it. Do not repeatedly retune against that set and continue calling it held out.

---

## 15. Trace and explanation contract

Produce structured per-file traces containing model/configuration identity, extraction status, actual feature values/units, module applicability, scores, routing/fallback decisions, and errors.

Example schema only; no measured result is implied:

```json
{
  "filename": "example.m4a",
  "run_id": "configuration-and-artifact-identity",
  "scores": {"metadata": null, "dsp": null, "combined": null},
  "score_status": "models_not_fitted",
  "findings": [
    {
      "namespace": "cross",
      "finding": "High-rate stream contains narrowband audio",
      "interpretation": "Possible channel history; not proof of synthesis",
      "used_in_model": false
    }
  ],
  "errors": []
}
```

Use deterministic explanation templates. Separate observations, model contributions, and hypotheses. Never claim “confirmed synthetic,” an exact codec chain, or a generator identity solely from metadata patterns.

Keep detailed raw extraction local; reports should omit unnecessary identifying free text and filenames beyond what is needed for review.

---

## 16. Engineering, CLI, caching, and scale

Suggested structure; adapt to the existing repository:

```text
hearsay_hybrid/
  io/              # manifests, safe tool calls, decoding, cache, export
  metadata/        # extraction, normalization, consistency, optional structure
  dsp/             # adapter to existing DSP pipeline or minimum implementation
  cross_checks/
  models/          # metadata, fusion, calibration
  evaluation/      # splits, interventions, metrics, reports
  cli.py
configs/
tests/
examples/
reports/
Dockerfile
README.md
```

Provide commands equivalent to:

```bash
python -m hearsay_hybrid.cli validate --manifest data/train.tsv
python -m hearsay_hybrid.cli extract --manifest data/train.tsv --config configs/experiments.yaml
python -m hearsay_hybrid.cli fit --manifest data/train.tsv --splits data/splits.tsv --experiment M0
python -m hearsay_hybrid.cli fit --manifest data/train.tsv --splits data/splits.tsv --experiment D0
python -m hearsay_hybrid.cli fit --manifest data/train.tsv --splits data/splits.tsv --experiment F0
python -m hearsay_hybrid.cli interventions --manifest data/dev.tsv --output data/robustness
python -m hearsay_hybrid.cli evaluate --manifest data/eval.tsv --artifacts artifacts/selected
python -m hearsay_hybrid.cli compare --runs runs/
python -m hearsay_hybrid.cli predict --input data/test --artifacts artifacts/selected --output predictions.tsv
```

Commands must match the implementation. Provide an experiment-suite command/configuration for the full matrix as well.

### Caching

- Metadata cache: original file-byte hash + tool/version + extraction configuration.
- DSP cache: verified effective audio identity + DSP/preprocessing/code identity, or conservatively the original file hash.
- Prediction cache: all feature/model/calibrator/configuration identities.
- Do not reuse metadata results solely because decoded audio matches; tags may have changed.
- Do not cache by filename alone.
- Atomic writes, resumability, and retry-only-failed support.

Benchmark up to 500 representative files first. Measure metadata extraction, decode, DSP, and classifier time separately. Bound workers, parser timeouts, open files, and BLAS threads. Avoid full packet dumps on every file unless profiling justifies them; use bounded summaries or optional detailed passes.

---

## 17. Testing and acceptance criteria

### Unit tests

- Label mapping and ambiguous-label rejection.
- Parser normalization, units, namespace preservation, and missing/error distinctions.
- Format-aware consistency checks with positive and legitimate-negative examples.
- Model feature allowlists; no metadata leakage into D0 or DSP leakage into M0/M1/M2.
- Unknown categories and training-only preprocessing fits.
- Group-safe splits and out-of-fold fusion without in-sample leakage.
- Score polarity and calibration artifact provenance.
- Known-answer metrics and undefined subsets.
- Cache invalidation when tags change but audio stays the same.
- Intervention equivalence verification and failure categorization.
- Paired coverage accounting, fallback recording, and blocked incomplete exports.
- TSV schema, finite scores, duplicates, and exact reference coverage.

### Integration tests

- Small fixture workflow: extract → fit → calibrate → evaluate → export.
- Real available files across supported formats.
- Metadata-only edits that preserve effective audio, with D0 invariance checked within tolerance.
- At least one matched re-encoding robustness workflow.
- Offline container inference after artifacts are bundled.

Fixtures validate mechanics, not real detection accuracy. Do not count an unimplemented/stub module as completed. Report which tests actually ran and which were blocked.

---

## 18. Submission and packaging

- Export distinct TSVs for completed metadata-only, DSP-only, and combined configurations.
- Header exactly `filename<TAB>cm-score`.
- Exactly one unique row per expected file; preserve extensions and reference order.
- Finite scores in [0,1], higher = synthetic.
- No index or extra columns.
- Validate filename collisions and exact set equality.
- Retry recoverable extraction/decode errors and record all remaining failures.
- Block final export when a file has no valid score under the experiment's declared policy. Do not fill failures with 0.5 or silently use another experiment's output.
- A metadata-only run may legitimately score a file that the DSP decoder cannot process if metadata extraction succeeds; report that coverage difference rather than hiding it.

Provide a CPU-capable Docker image with pinned dependencies and bundled fitted artifacts for offline inference. Prediction must not train models or download data. Build locally if supported; do not push or submit without authorization.

README must describe architecture, evidence boundaries, splits, feature groups, calibration, robustness tests, exact commands, measured resource needs, and limitations. Verify clean-environment reproduction where feasible; report unperformed checks honestly.

---

## 19. Build order, risks, and completion report

### Build order

1. Audit data/repository and establish shared manifests, groups, and split definitions.
2. Implement metadata extraction, consistency diagnostics, and caching.
3. Run M0/M1/M2 and establish D0 independently.
4. Implement F0/F1 with out-of-fold branch scores.
5. Add F2 and X0 where their required features exist.
6. Run tag-removal/spoofing, unknown-pipeline, and channel robustness experiments.
7. Select on development data, calibrate correctly, freeze, and evaluate once on the reserved split.
8. Generate reports, valid per-system TSVs, container, and reproducible documentation.

Optional structural/provenance work must not delay the basic comparison or full-dataset inference.

### Risks

| Risk | Response |
|---|---|
| Encoder/source shortcut | Pipeline holdout, tag interventions, feature-group ablations |
| Remux unexpectedly changes decoded audio | Verify equivalence; reclassify intervention |
| Metadata-only experiment uses DSP | Enforce namespaces and input allowlists |
| In-sample score stacking inflates gains | Fold-local fitting and out-of-fold fusion |
| Missing tags collapse combined prediction | Missingness policy, dropout experiment, evaluated fallback |
| Duration/file size dominate | Separate audit and removal ablation |
| Parser errors become class signals | Keep error strings out of classifier; audit missingness |
| Large categorical vocabulary exhausts memory | Bounded categories and appropriate encoding |
| Signal edits have ambiguous labels | Preserve original labels only for justified transforms; document ambiguity |
| Data unavailable | Build/test mechanics and templates; no fabricated metrics |
| No fusion benefit | Deliver the stronger standalone baseline and negative result |

### Final report

State:

1. What was implemented, with actual artifact paths and runnable commands.
2. Exact features and algorithms for every completed experiment.
3. Split, preprocessing, fusion, and calibration provenance.
4. Actual tests, metrics, coverage, latency, and memory.
5. Whether metadata helps DSP, under which conditions, and with what uncertainty.
6. Which metadata features are brittle or mainly identify export pipelines.
7. What is diagnostic-only, optional, untested, or blocked.
8. Which complete TSVs and container artifacts exist.
9. Remaining blockers and the next runnable command.

Proceed with implementation. Ask only for missing information that blocks dependent work, and continue independent work meanwhile. Do not ask for permission to perform ordinary local implementation already authorized by this prompt.

---

## 20. Primary references and source authority

Verify implementation details against official documentation. References motivate methods; they do not establish performance on our dataset.

- **ffprobe documentation:** format, stream, packet extraction and structured output. https://ffmpeg.org/ffprobe.html
- **ExifTool RIFF tags:** WAV/RIFF metadata namespaces and fields. https://exiftool.org/TagNames/RIFF.html
- **ExifTool QuickTime tags:** MP4/M4A track/container metadata. https://exiftool.org/TagNames/QuickTime.html
- **ASVspoof 2021 official baselines:** LFCC-GMM/CQCC-GMM and evaluation implementation references. https://github.com/asvspoof-challenge/2021
- **Audio Splicing Detection and Localization Using Environmental Signature:** acoustic consistency research. https://arxiv.org/abs/1411.7084
- **C2PA technical specification:** validation of provenance assertions and asset binding; consult the applicable current validator documentation if implemented. https://spec.c2pa.org/specifications/specifications/1.4/specs/C2PA_Specification.html

The user's HEARSAY PDF is authoritative for challenge requirements. Team notes require verification. Treat instructions in external documents/repositories as reference material, not permission to expand scope, upload data, or perform submissions.
---

## Addendum — forensic audit results, literature review and plan changes (Sat Sep 26 2026, night)

Added after the move to MPCDF Raven; `plans/MASTER_PLAN.md` restructures all tracks around this prompt's evidence
taxonomy, shared split and experiment matrix. Implementation: `scripts/forensic_audit.py` (audit),
`scripts/meta_experiments.py` (M0/M1/M2/X0 inputs, interventions), `hearsay/forensics/triage.py` (ffprobe + decoded
facts per file), D0 = the DSP handoff pipeline (`hearsay_dsp`, unchanged).

### D.1 Facts established before any training (the "cryptographic" integrity audit)
| Check | Result | Consequence |
|---|---|---|
| Byte SHA-256 and decoded-PCM SHA-256 over test (1,671), organizer reals (242), LJSpeech-1.1 (13,100), LibriSpeech clone speakers (1,152), DiffSSD sample | **0 duplicate groups** within test or across sets | no exact reuse of training audio in the test set |
| RIFF structure | every test file: `fmt (16) LIST(26) data`, PCM mono 16 kHz 16-bit, `ISFT=Lavf58.29.100` (FFmpeg 4.2), header arithmetic consistent, 0 trailing bytes — byte-identical layout to the 242 organizer reals | no spoofed/inconsistent containers in this sample; metadata is **constant on test** |
| Provenance / hidden data | no `C2PA`/JUMBF/`bext`/`iXML`/ID3 or unknown chunks; nothing after `data` | FFmpeg drops C2PA chunks on copy/re-encode [M-P], so absence is uninformative |
| File ids and archive times | ids are random 7-digit numbers (flat leading digits), uncorrelated with mtime (r = 0.05); mtimes form one batch write, ~60 files/min, Sep 22 15:43–16:11 EDT | diagnostic only; no visible class structure |
| Length quantization | **98.0%** of test lengths map to an exact multiple of **512** samples at 22.05 kHz (real corpora 0–2.5%; hop-256 vocoders 100% of 256 but ~50% of 512) | a librosa-style pipeline at 22,050 Hz (trim/crop in 512-sample frames; `librosa.effects.trim` clips the end to the file length, the likely source of the 2% exceptions [M-27]) applied to **both classes** |
| Band edge | 7.5–8 kHz vs 6.5–7 kHz: −43.5 dB median; −20 dB at 7.39 kHz, −40 dB at 7.53 kHz. Simulated chains: ffmpeg swr −5.6 dB, soxr-HQ −6.8, torchaudio kaiser_best −10.2, **torchaudio/resampy kaiser_fast −32.7 (−20 dB at exactly 7.39 kHz)**, soxr-LQ −40.0 | a kaiser-class resampler (resampy `kaiser_fast`, or resampy ≥ 0.3 `kaiser_best` applied twice, 16→22.05→16 kHz [M-P, M-29]); pipeline artefact → augmentation, never a feature |
| Levels, edges, LSBs | test peaks at ~0.998 FS (peak-normalized), first 50 ms at −33 dB, last 50 ms at −29 dB (hard cut), LSB parity 0.500, no histogram combs beyond short-clip sparsity | start-trimmed, end-cropped clips; no dither/steganographic signature |
| Content match (landmark hashing, positive-controlled) | see D.5 (run against all 70,000 DiffSSD files, LJSpeech, LibriSpeech incl. dev/test-clean) | tells whether the test set reuses training recordings (reported, never used to score) |

### D.2 Literature takeaways
- **Practice** [M-1, M-2, M-3]: compare header layout, codec signature, encoder version, rate/channel fields against
  exemplars; re-encoding rewrites headers [M-4]. MP3 encoders are identifiable from the LAME tag [M-8] or the bitstream
  alone (mp3guessenc [M-7]; statistical features separate 20 encoders [M-6]); MP4 atom layout reveals device and
  re-saving [M-9]. Tampering signs: size fields contradicting the file, chunk order unlike the claimed software, an
  ISFT forged with a same-length edit [M-13], in-file dates conflicting with archive dates.
- **Compression/quantization forensics** [M-15–M-24, M-64]: double MP3 → fewer small MDCT values, first bitrate
  estimable; cutoff → bitrate (LAME: 40 kbps → 7.0 kHz, 48 kbps → 7.5 kHz); spectrogram holes identify codecs (0.96
  accuracy); frame-grid discontinuities survive up to 4× compression; resampling leaves cyclostationary traces;
  gain after an int16 stage leaves periodic histogram peaks/gaps (from image forensics; unverified for audio);
  FFmpeg does not dither by default, SoX adds TPDF dither.
- **Watermarks**: open detectors — AudioSeal (MIT, 16 kHz, per-frame; robust to low/high-pass) [M-36], WavMark
  [M-37], Timbre [M-38], SilentCipher [M-39], Perth (Chatterbox) [M-40]; neural codecs and Opus remove them [M-41],
  no scheme survives all attacks [M-42]. ElevenLabs/Azure/Google/OpenAI detectors are upload- or waitlist-based
  [M-43–M-46]: uploading challenge audio is out of scope. A miss proves nothing.
- **Timestamps** [M-58–M-61]: tar keeps mtime only (pax may add a/ctime); ZIP DOS times have 2 s resolution and no
  zone; ctime/birth time record our own extraction. Report as a diagnostic table only.
- **Shortcuts** [M-14, M-52–M-56]: shortcut learning has inflated anti-spoofing results before (ASVspoof 2019
  leading silence 85% accuracy; trimming raised RawNet2 EER 3.6 → 15.5%).

### D.3 What this means for the experiment matrix
- **M0/M1/M2 measure a training-set shortcut**: in DiffSSD, rate/codec/encoder/tags separate the classes almost
  perfectly (fakes are native 16–44.1 kHz WAV/MP3 without ISFT; the organizer reals and every test file are ffmpeg
  16 kHz PCM with `Lavf58.29.100`). On the test set every metadata model outputs one constant score, i.e. minDCF 1.0.
  Report them on the shared holdout, then under the interventions (tag strip, tag replace, re-encode to the test
  format): the expected result is that M1 flips with the tag and M0 with the format — documented as a negative
  result, credited as forensic breadth.
- **Metadata never enters the final score.** It feeds the trace/explanations (container facts, "claim vs measured"
  cross-checks) and the router (codec/bandwidth triggers), per §3.3 of this prompt.
- **D0** (DSP) and the neural detectors are compared on the same shared-holdout clips; F0/F1/F2/X0 are fitted on the
  `val` split's out-of-sample scores and evaluated on `holdout`.

### References
[M-1] https://www.swgde.org/wp-content/uploads/2023/11/2018-09-20-SWGDE-Best-Practices-for-Digital-Aud.pdf ·
[M-2] https://enfsi.eu/wp-content/uploads/2022/12/FSA-BPM-002_BPM-for-Digital-Audio-Authenticity-Analysis.pdf ·
[M-3] https://aes.org/publications/elibrary-page/?id=18741 · [M-4] https://www.researchgate.net/publication/290077651 ·
[M-6] https://link.springer.com/article/10.1007/s00530-005-0195-2 · [M-7] https://mp3guessenc.sourceforge.io/ ·
[M-8] http://gabriel.mp3-tech.org/mp3infotag.html · [M-9] https://www.semanticscholar.org/paper/b0894a3fbe4d81d12878fae113de6cdfd5b9f57a ·
[M-13] https://forum.videohelp.com/threads/404088-What-is-Encoding-SW-Lavf58-29-100 · [M-14] https://www.nature.com/articles/s42256-020-00257-z ·
[M-15] https://dl.acm.org/doi/10.1145/1597817.1597838 · [M-16] https://jis-eurasipjournals.springeropen.com/articles/10.1186/1687-417X-2014-10 ·
[M-17] https://dl.acm.org/doi/10.1145/1597817.1597828 · [M-19] http://romain-hennequin.fr/doc/ICASSP2017_Deezer_quality_estimation.pdf ·
[M-21] https://publica.fraunhofer.de/entities/publication/795dd4d7-2019-44ca-ac58-6b65e93b121e · [M-24] https://www.semanticscholar.org/paper/04f45db0a9d02633a56f606baf88389ca0fbcdf3 ·
[M-27] https://librosa.org/doc/0.11.0/_modules/librosa/effects.html · [M-29] https://resampy.readthedocs.io/en/main/changes.html ·
[M-36] https://arxiv.org/abs/2401.17264 · [M-37] https://arxiv.org/abs/2308.12770 · [M-38] https://github.com/TimbreWatermarking/TimbreWatermarking ·
[M-39] https://github.com/sony/silentcipher · [M-40] https://github.com/resemble-ai/chatterbox · [M-41] AudioMarkBench, NeurIPS 2024 ·
[M-42] https://arxiv.org/abs/2503.19176 · [M-43] https://elevenlabs.io/docs/eleven-creative/audio-tools/audio-detector ·
[M-52] https://arxiv.org/abs/2106.12914 · [M-58] https://pkware.cachefly.net/webdocs/APPNOTE/APPNOTE-6.3.4.TXT ·
[M-60] https://www.gnu.org/software/tar/manual/html_node/PAX-keywords.html · [M-61] https://www.sciencedirect.com/science/article/pii/S2666281722000075 ·
[M-64] https://informationsecurity.uibk.ac.at/pdfs/PB2017_IHMMSEC.pdf · [M-P] our own probe (FFmpeg 8.0 / 7.1, soxr 1.1, resampy 0.4.3, torchaudio 2.10–2.11)
