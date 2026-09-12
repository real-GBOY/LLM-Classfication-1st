# Experiment Log

Format per the project's reporting convention. Every entry here is a REAL
completed result — never fill this in with a predicted/expected number.

---

## Baseline: DeBERTa-v3-xsmall (first run)
- Configuration: `notebooks/03_deberta_train.py`, CONFIG as committed (lr=2e-5, batch_size=16, epochs=2, weight_decay=0.01, warmup_ratio=0.1, sequence_length=512, no clipnorm, no label smoothing)
- Train loss @ best epoch (1): 1.0898
- Val log loss @ best epoch (1): **1.0784**
- Val accuracy @ best epoch (1): 0.3948
- Epoch 2 (rejected by checkpointing): train loss 1.0695, val log loss 1.0869, val accuracy 0.4063 — accuracy up, log loss up = overconfidence/miscalibration, not underfitting
- Training time: not recorded (add next time)
- Decision: KEEP as reference baseline. Best checkpoint = epoch 1 (`deberta_xsmall_best.keras`), correctly selected by `ModelCheckpoint(monitor="val_loss")`.

## Reference: TF-IDF + Logistic Regression (C=0.1)
- Val log loss: 1.0803
- Decision: reference baseline, superseded by the blend below.

## Reference: length-features-only Logistic Regression
- Val log loss: 1.0747
- Decision: reference baseline, superseded by the blend below.

## Experiment: TF-IDF + length-only blend (alpha search)
- Configuration: `P = alpha * P_tfidf + (1-alpha) * P_length`, alpha swept 0.00-1.00 step 0.05 on the same held-out val split as DeBERTa (group-split by prompt, seed=42, test_size=0.1)
- Best alpha: 0.35
- Val log loss: **1.0654**
- Val accuracy: n/a (optimizing log loss directly, not accuracy)
- Training time: ~15s CPU (no GPU)
- Delta vs DeBERTa baseline (1.0784): **-0.0130 (better)**
- Delta vs TF-IDF alone (1.0803): -0.0149 (better)
- Delta vs length-only alone (1.0747): -0.0093 (better)
- Decision: **KEEP — current best result in the project.**
- Reproduce: `python notebooks/07_ensemble.py`
- Note: curve has a broad flat minimum (alpha 0.30-0.40 all within 0.0005 of best), not a spiky single-point optimum — not an artifact of the ~5.8k-row validation set.

## Experiment: exp1_lr1e-5 (DeBERTa-v3-xsmall, lr=1e-5)
- Configuration: `notebooks/06_deberta_experiments.py`, identical to baseline except `learning_rate: 2e-5 -> 1e-5`. Same seed (42), same split, same text construction, same batch size/epochs/weight_decay/warmup.
- Train loss @ best epoch (2): 1.0552
- Val log loss @ best epoch (2): **1.0583**
- Val accuracy @ best epoch (2): 0.4426
- Training time: 39.2 min (T4, single-GPU as before)
- Delta vs DeBERTa baseline (1.0784): **-0.0201 (better)**
- Delta vs TF-IDF+length blend (1.0654): **-0.0071 (better)**
- Decision: **KEEP — new best single model, and new best result overall in the project.**
- Confirms the diagnosis: at lr=1e-5, epoch 2 IMPROVED over epoch 1 (unlike baseline, where epoch 2 got worse) — the original 2e-5 LR was genuinely too aggressive, causing the overconfidence/miscalibration pattern. Lowering it fixed the failure mode directly rather than needing gradient clipping or label smoothing as a workaround.
- Checkpoint: `/kaggle/working/deberta_xsmall_exp1_lr1e-5.keras`; val predictions saved to `val_proba_exp1_lr1e-5.npy` + `val_ids_exp1_lr1e-5.csv` for the 3-way ensemble test (not yet run — needs these files downloaded to `outputs/deberta_preds/`).

---

## Next steps (not yet run)

1. **3-way ensemble (DeBERTa exp1 + TF-IDF + length)** — free, local, no GPU. Download `val_proba_exp1_lr1e-5.npy` + `val_ids_exp1_lr1e-5.csv` from `/kaggle/working/` into `outputs/deberta_preds/`, set `DEBERTA_EXPERIMENT = "exp1_lr1e-5"` in `notebooks/07_ensemble.py`, run it. Highest expected value next step given exp1 alone already beats the previous best blend.
2. **exp2_clipnorm1 / exp3_label_smoothing** — lower priority now that exp1 already fixed the diagnosed problem directly. Worth trying only if there's still a calibration gap after the ensemble step, or as an additive stack on top of lr=1e-5 (e.g. `exp4_combined`: lr=1e-5 + label_smoothing=0.05) if there's reason to believe further calibration headroom remains.

Explicitly NOT tested, with reasons (avoiding low-value GPU spend):
- **Higher LR (e.g. 3e-5)**: evidence points the opposite direction; testing it first would likely make things worse.
- **sequence_length=768**: backbone's own `config.json` confirms `max_sequence_length=512` — this is architecturally unsupported without position-embedding surgery, not just "unlikely to help."
- **Input reformatting (Priority 2)**: local analysis shows the achievable signal is currently dominated by length + weak lexical cues (see below), not by how clearly fields are delimited in an already-explicit format. Revisit only if exp1-4 still leave DeBERTa behind the blend.

## Analysis: why does TF-IDF ≈ DeBERTa?

Computed locally (`scripts` reproduced in chat, 2026-09-12):
- TF-IDF's P(A wins) has ~zero correlation with response length difference (r=-0.011) — it is NOT just re-deriving the length signal.
- On the top quartile of |length difference| (length clearly favors one side): length-only log loss 1.038 beats TF-IDF's 1.070.
- On the bottom quartile (length uninformative): TF-IDF's 1.074 beats length-only's 1.098.
- Conclusion: length and lexical/TF-IDF signal are **complementary**, each covering the region where the other is weak. This is exactly why the blend beats both individually, and is the reason DeBERTa (which should in principle capture both plus more) hasn't yet pulled ahead after only one good epoch of fine-tuning.

## Per-class diagnostics (TF-IDF, same val split; DeBERTa's own breakdown not yet run)

Confusion matrix (rows=true, cols=pred): A/B/tie
```
           pred_A  pred_B  pred_tie
true_A       970     707      418
true_B       926     681      378
true_tie     669     473      601
```
Per-class log loss: A wins 1.047, B wins 1.068, **tie 1.134 (worst)**.
Tie is the smallest class (30.9%) and the hardest to call — consistent with tie being the most subjective human judgment in the underlying Chatbot Arena data.

`notebooks/04_error_analysis.py` produces the equivalent breakdown for DeBERTa but has not been run yet (needs a completed checkpoint on Kaggle).
