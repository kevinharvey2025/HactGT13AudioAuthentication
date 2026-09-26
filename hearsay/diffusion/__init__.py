"""Track D (plans/diffusion_cf_prompt.md section 7): diffusion-model features and explanations.

Implemented but not yet run. Each module is optional: fusion treats a missing feature block as
"not run", and every block has to pass the go/no-go gate (section 7.4) before it joins the model.

    ddpm.py        D2  DDPM on SSL embeddings of bona fide speech -> mode-path features (H2)
    prototypes.py  D5  Gaussian prototypes, per-dimension submodular composition, basic level
    resynth.py     D1  vocoder / codec resynthesis residuals (H1, H3)
    latent.py      D3  latent audio diffusion (AudioLDM 2) denoising-loss curve
    score_se.py    D4  score-based speech enhancement (SGMSE+) features, exploratory
    generate.py    D6  training-data generation: copy-synthesis, partial fakes, laundering
"""
