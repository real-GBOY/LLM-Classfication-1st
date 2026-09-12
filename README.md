# LLM Classification Finetuning

Predicts which of two LLM responses a human would prefer (`winner_model_a` /
`winner_model_b` / `winner_tie`), scored by multiclass log loss.
Competition: https://www.kaggle.com/competitions/llm-classification-finetuning

Model: `deberta_v3/keras/deberta_v3_extra_small_en` (KerasHub/KerasNLP preset),
fine-tuned for 3-way sequence classification.

## Project layout

```
src/data.py           loading, JSON-turn parsing, leakage-safe grouped train/val split
src/preprocessing.py  builds "[PROMPT] ... [RESPONSE A] ... [RESPONSE B] ..." text,
                       including the word-budget truncation used for DeBERTa
src/evaluate.py        multiclass log loss + submission sanity checks
configs/deberta_base.yaml  documented hyperparameter defaults for the DeBERTa run

notebooks/01_eda.py             class balance, length distributions, model win rates, leakage checks
notebooks/02_baseline_tfidf.py  TF-IDF + Logistic Regression reference (runs locally, no GPU)
notebooks/03_deberta_train.py   first successful DeBERTa run (val log loss 1.0784) -- KAGGLE GPU ONLY, do not re-run/overwrite
notebooks/04_error_analysis.py  confusion matrix, high-confidence errors, length/model-pair breakdowns (KAGGLE GPU ONLY)
notebooks/05_predict_submit.py  test prediction + verified submission.csv (KAGGLE GPU ONLY)
notebooks/06_deberta_experiments.py  controlled experiment runner (edit EXPERIMENT_NAME + one CONFIG line per run, never overwrites a checkpoint) -- KAGGLE GPU ONLY
notebooks/07_ensemble.py        TF-IDF + length + DeBERTa probability blending (runs locally, no GPU)

outputs/eda/          saved EDA charts (PNG)
outputs/baseline/     baseline results.txt
outputs/experiments/  experiment_log.md -- every completed experiment, real results only
outputs/deberta_preds/  val_proba_<experiment>.npy + val_ids_<experiment>.csv downloaded from Kaggle, for 07_ensemble.py
```

`notebooks/*.py` use `# %%` cell markers (Jupytext/VS Code interactive
format) -- open them directly in VS Code, or paste cell-by-cell into a
Jupyter/Kaggle notebook.

## Why the local/Kaggle split

This machine has pandas/scikit-learn but no TensorFlow/KerasHub/GPU, so:
- **Runs locally**: `01_eda.py`, `02_baseline_tfidf.py` (both plain Python/sklearn).
- **Must run on Kaggle** (needs GPU + the DeBERTa preset attached):
  `03_deberta_train.py`, `04_error_analysis.py`, `05_predict_submit.py`.
  These are self-contained (duplicate the small parsing/truncation helpers
  from `src/` inline) so they can be pasted into a fresh Kaggle notebook
  without uploading the rest of the repo -- keep them in sync with `src/`
  manually if those files change.

Before running `03_deberta_train.py` on Kaggle: attach the competition
dataset and the `deberta_v3/keras/deberta_v3_extra_small_en` model via
"+ Add Input", and set the accelerator to a GPU (T4 or P100).

**Path note (confirmed 2026-09):** Kaggle nests competition inputs under
`/kaggle/input/competitions/<slug>/`, not directly under `/kaggle/input/<slug>/`
as older examples show. `src/data.py` and the notebook scripts check both
paths so this survives environment differences.

## Key data facts (see notebooks/01_eda.py for full detail)

- `train.csv`: 57,477 rows, no missing values. `prompt`/`response_a`/`response_b`
  are JSON-encoded lists of conversation turns, not plain strings.
- `test.csv` in this repo is a 3-row stub (normal for this competition) --
  it has **no `model_a`/`model_b` columns**, so model identity can never be
  used as an input feature.
- Class balance: A 34.9% / B 34.2% / tie 30.9% -- roughly balanced.
- 3,118 unique prompts repeat 2-13x across rows -> train/val split is
  **grouped by prompt** (`src/data.py:group_split`), not plain stratified,
  to avoid leaking a prompt seen in training into validation.
- Median total example length ~2,360 characters (~590 tokens); DeBERTa's
  typical 512-token cap means most examples need truncation before they
  even reach the tokenizer -- handled via a per-segment word budget in
  `src/preprocessing.py:build_truncated_compare_text` (head+tail truncation
  for responses, most-recent-turn-first for multi-turn prompts) rather than
  naive right-truncation, which would silently drop `response_b` on long
  examples.
- Non-tie examples: the longer response wins 61.6% of the time -- a real,
  legitimate signal (not leakage), worth tracking in error analysis so the
  model doesn't just become a length-counter.

## Results so far (see `outputs/experiments/experiment_log.md` for full detail)

| model | val log loss |
|---|---|
| naive (class priors) | 1.0961 |
| length-features-only (logistic regression) | 1.0747 |
| TF-IDF (1-2gram) + Logistic Regression, C=0.1 | 1.0803 |
| DeBERTa-v3-xsmall, lr=2e-5 (first run) | 1.0784 |
| TF-IDF + length-only blend (alpha=0.35) | 1.0654 |
| **DeBERTa-v3-xsmall, lr=1e-5 (exp1)** | **1.0583 -- current best** |

Note: TF-IDF at the sklearn-default `C=1.0` actually scored *worse* than
guessing class priors (1.118) -- 50k sparse features overfit a genuinely
weak word-overlap signal. Regularizing harder (`C=0.1`) fixed it. Analysis
showed TF-IDF's predictions have ~zero correlation with response length
(r=-0.011) -- length and lexical signal are complementary, not redundant,
which is why blending them beats both individually AND currently beats
DeBERTa. DeBERTa's epoch 1 -> epoch 2 pattern (accuracy up, log loss up) is
a calibration/overconfidence issue, not underfitting -- see the experiment
log for the targeted fixes being tested (lower LR, gradient clipping, label
smoothing) before revisiting input formatting or model size.

## Reproducing

```
python notebooks/01_eda.py            # local, no GPU needed
python notebooks/02_baseline_tfidf.py # local, no GPU needed
python notebooks/07_ensemble.py       # local, no GPU needed -- current best result
```
`03_deberta_train.py` (already run, produced the 1.0784 baseline -- do not
re-run/overwrite) and `06_deberta_experiments.py` (edit `EXPERIMENT_NAME` +
one CONFIG line per experiment) run in a Kaggle GPU notebook session. After
each experiment, download `val_proba_<name>.npy` + `val_ids_<name>.csv` from
`/kaggle/working/` into `outputs/deberta_preds/` and set `DEBERTA_EXPERIMENT`
in `07_ensemble.py` to get the 3-way blend.
