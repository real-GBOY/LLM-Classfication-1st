# %% [markdown]
# # Baseline: TF-IDF + Logistic Regression
#
# Purpose: a fast, non-neural reference point. If DeBERTa can't beat this by
# a meaningful margin, something is wrong with the fine-tuning setup, not the
# problem itself.

# %%
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_train, group_split, label_to_class, TARGET_COLS
from src.preprocessing import build_compare_text
from src.evaluate import multiclass_log_loss, assert_valid_probabilities

SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Data + split
# Reuses the same grouped split as EDA so the baseline's validation score is
# directly comparable to DeBERTa's later -- same held-out rows, same leakage
# guarantees.

# %%
train = load_train()
tr_df, va_df = group_split(train, test_size=0.1, seed=SEED)
print(f"train: {tr_df.shape}, val: {va_df.shape}")

t0 = time.time()
tr_text = [build_compare_text(p, a, b) for p, a, b in zip(tr_df.prompt, tr_df.response_a, tr_df.response_b)]
va_text = [build_compare_text(p, a, b) for p, a, b in zip(va_df.prompt, va_df.response_a, va_df.response_b)]
print(f"text building took {time.time()-t0:.1f}s")

y_tr = label_to_class(tr_df)
y_va = label_to_class(va_df)

# %% [markdown]
# ## TF-IDF vectorization
# word 1-2 grams, capped vocabulary -- enough for a linear baseline without
# it becoming its own multi-minute preprocessing step.

# %%
t0 = time.time()
vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)
X_tr = vectorizer.fit_transform(tr_text)
X_va = vectorizer.transform(va_text)
print(f"vectorization took {time.time()-t0:.1f}s, X_tr shape={X_tr.shape}")

# %% [markdown]
# ## Fit + evaluate
# Multinomial logistic regression trained directly on the 3-way label
# (argmax of the one-hot targets), evaluated with the competition's actual
# metric (multiclass log loss), not accuracy.
#
# NOTE on C=0.1: an initial run at the sklearn default (C=1.0) with 50k
# sparse TF-IDF features actually scored WORSE than a naive class-prior
# guess (1.118 vs 1.096) -- classic overfitting to noise when the true
# signal-to-feature-count ratio is low. A small C-sweep (0.001/0.01/0.1/1.0)
# showed C=0.1 is the sweet spot. This matters because it shows raw
# word-overlap carries only weak signal for "who wins" -- model_a/model_b
# position is randomly assigned, so words don't correlate with the label the
# way they would in a topic-classification task.

# %%
t0 = time.time()
clf = LogisticRegression(max_iter=200, C=0.1, random_state=SEED)
clf.fit(X_tr, y_tr)
print(f"fit took {time.time()-t0:.1f}s")

proba_va = clf.predict_proba(X_va)
# LogisticRegression orders columns by clf.classes_ -- confirm it matches [0,1,2]
assert list(clf.classes_) == [0, 1, 2], f"unexpected class order: {clf.classes_}"

assert_valid_probabilities(proba_va)
val_logloss = multiclass_log_loss(va_df, proba_va)
val_acc = (proba_va.argmax(axis=1) == y_va.to_numpy()).mean()

# reference: naive uniform-probability baseline (always predict train class priors)
priors = tr_df[TARGET_COLS].mean().to_numpy()
naive_proba = np.tile(priors, (len(va_df), 1))
naive_logloss = multiclass_log_loss(va_df, naive_proba)

# reference: length-only features (isolates how much of the TF-IDF score is
# just re-deriving the "longer response tends to win" signal from EDA)
from src.preprocessing import render_prompt_cell

def char_len(raw):
    return len(render_prompt_cell(raw))

len_feats_tr = np.array([
    [char_len(a), char_len(b), char_len(a) - char_len(b)]
    for a, b in zip(tr_df.response_a, tr_df.response_b)
])
len_feats_va = np.array([
    [char_len(a), char_len(b), char_len(a) - char_len(b)]
    for a, b in zip(va_df.response_a, va_df.response_b)
])
clf_len = LogisticRegression(max_iter=500, random_state=SEED)
clf_len.fit(len_feats_tr, y_tr)
len_logloss = multiclass_log_loss(va_df, clf_len.predict_proba(len_feats_va))

print(f"\nNaive (class-prior) val log loss:  {naive_logloss:.4f}")
print(f"Length-features-only val log loss: {len_logloss:.4f}")
print(f"TF-IDF + LogReg val log loss:       {val_logloss:.4f}")
print(f"TF-IDF + LogReg val accuracy:       {val_acc:.4f}")

with open(OUT_DIR / "results.txt", "w") as f:
    f.write(f"naive_logloss={naive_logloss:.4f}\n")
    f.write(f"length_only_logloss={len_logloss:.4f}\n")
    f.write(f"tfidf_logreg_val_logloss={val_logloss:.4f}\n")
    f.write(f"tfidf_logreg_val_acc={val_acc:.4f}\n")

print(f"\nResults saved to {OUT_DIR / 'results.txt'}")
