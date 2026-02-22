# chronos-lens
Geometric latent space analysis of Joint Embedding Predictive Architectures (JEPA) for clinical temporal mental health prediction.

Repository for preprint "Geometric Latent Space Analysis of JEPA for Clinical Mental Health Risk Prediction (UT Austin, 2026).

### Motivation
Clinical ML models operating on temporal patient data demand interpretability, yet the representations they learn remain opaque. Standard mechanistic interpretability techniques, done on autoregressive transformers, assume prediction in token space - JEPAs break this by predicting in a learned latent space with no discrete vocabulary. Furthermore, these techniques assume a lossy translation layer to human words - that is: residual stream $\rightarrow$ logits $\rightarrow$ vocabulary $\rightarrow$ human words. This project aims to utilize JEPA's prediction of it's latent embedding space as a medium of mechanistic interpretability (MI) analysis directly, rather than force human labels and discretize them out of the model:
- what fraction of the latent space's geometric structure can be explained by clinical metadata via LASSO regression?
- How does the predictor learn the *structure* of the embedding space?
- Can we map latent space embedding predictions to untampered human labels?
- where in the encoder do clinically meaningful representations emerge?
- do principal component axes correspond to stable traits vs. dynamic state changes?
- Simply, *can the latent space of a [JEPA] model be explained?*

The JEPA encodes patient encounter sequences (ICD codes, active medications) and predicts the embedding of a masked future encounter. A *ViT-Tiny* context encoder processes the sequence; an EMA-updated target encoder provides training signal. The predictor maps context representations to target representations in latent space, producing the displacement field $\Delta$ that is the primary object of analysis. Because both $z_{pred}$ and $z_{context}$ live in the same embedding space by construction, their difference is geometrically meaningful - this is the architectural property that makes the analysis possible.

### Analysis Pipeline
1. Displacement field construction: Compute $\Delta_i$ = $z_{pred} - z_{context}$ per patient, decompose into magnitude (uncertainty/volatility) and direction.
2. PCA decomposition: Eigenvalue spectrum for intrinsic dimensionality, Tracy-Widom test to separate structure from noise, UMAP comparison to validate linearity.
3. Divergence analysis: Identify patient pairs with similar histories but divergent predictions; test whether divergence vectors cluster along principal axes.
4. Stability analysisn: ICC of per-patient PC projections across encounter windows to classify axes as trait-like (high ICC) vs. state-like (low ICC).
5. Partial labeling bridge: LASSO regression with stability selection regressing clinical metadata against PC scores. Residual variance quantifies structure the model learned beyond the clinical concept space.

Supporting MI techniques (linear probing, activation patching, CKA, and SAE encoding) provide additional context but are secondary to the aim of the project.

### Reproduce
```
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

This project uses MIMIC-IV via BigQuery. Reproducing requires:
1. Physionet credentials, from https://physionet.org/content/mimiciv/
2. Link your PhysioNet account to a GCP project: https://physionet.org/settings/cloud/
3. Authenticate locally:
```bash
   gcloud auth application-default login
   gcloud config set project 
```
1. Update `BQ_PROJECT_ID` in the config
