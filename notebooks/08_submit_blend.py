# %% [markdown]
# # Submission: TF-IDF + length-only blend (current best, val log loss 1.0654)
#
# Runs LOCALLY (no GPU needed). This is the current best-validated model in
# the project (see outputs/experiments/experiment_log.md) -- use this for a
# submission today; swap in a better ensemble once a DeBERTa experiment from
# 06_deberta_experiments.py beats 1.0784 and is blended via 07_ensemble.py.
#
# Final-fit note: alpha=0.35 was chosen on the held-out validation split.
# For the actual submission we refit both underlying models on 100% of
# train.csv (not just the 90% train split) so the final models see the most
# labeled data possible -- refitting on more data after the config is
# already chosen (not before) is standard practice, not leakage: alpha was
# never tuned using test.csv or any information outside train.csv.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_train, load_test, label_to_class, TARGET_COLS
from src.preprocessing import build_compare_text, render_prompt_cell

SEED = 42
ALPHA = 0.35  # chosen via validation sweep in notebooks/07_ensemble.py

BASE = Path(__file__).resolve().parent.parent

train = load_train()
test = load_test()
sample_sub = pd.read_csv(BASE / "sample_submission.csv")

y_full = label_to_class(train).to_numpy()

# --- TF-IDF (refit on 100% of train) ---
train_text = [build_compare_text(p, a, b) for p, a, b in zip(train.prompt, train.response_a, train.response_b)]
test_text = [build_compare_text(p, a, b) for p, a, b in zip(test.prompt, test.response_a, test.response_b)]

vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)
clf_tfidf = LogisticRegression(max_iter=200, C=0.1, random_state=SEED)
clf_tfidf.fit(X_train, y_full)
proba_tfidf_test = clf_tfidf.predict_proba(X_test)

# --- length-only (refit on 100% of train) ---
def char_len(raw):
    return len(render_prompt_cell(raw))

len_train = np.array([[char_len(a), char_len(b), char_len(a) - char_len(b)] for a, b in zip(train.response_a, train.response_b)])
len_test = np.array([[char_len(a), char_len(b), char_len(a) - char_len(b)] for a, b in zip(test.response_a, test.response_b)])
clf_len = LogisticRegression(max_iter=500, random_state=SEED)
clf_len.fit(len_train, y_full)
proba_len_test = clf_len.predict_proba(len_test)

# --- blend + submission ---
proba_test = ALPHA * proba_tfidf_test + (1 - ALPHA) * proba_len_test

submission = pd.DataFrame({
    "id": test["id"],
    "winner_model_a": proba_test[:, 0],
    "winner_model_b": proba_test[:, 1],
    "winner_tie": proba_test[:, 2],
})

assert list(submission.columns) == list(sample_sub.columns), f"column mismatch: {list(submission.columns)} vs {list(sample_sub.columns)}"
assert len(submission) == len(sample_sub), f"row count mismatch: {len(submission)} vs {len(sample_sub)}"
assert (submission["id"].values == sample_sub["id"].values).all(), "id order/values mismatch"
assert not submission.isna().any().any(), "NaNs present in submission"
prob_cols = TARGET_COLS
assert (submission[prob_cols].to_numpy() >= 0).all() and (submission[prob_cols].to_numpy() <= 1).all()
row_sums = submission[prob_cols].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), f"rows don't sum to 1: min={row_sums.min()}, max={row_sums.max()}"

print("all submission checks passed")
print(submission.head())

out_path = BASE / "outputs" / "submission_tfidf_length_blend.csv"
submission.to_csv(out_path, index=False)
print(f"\nsaved to {out_path}")
