# %% [markdown]
# # Ensembling: TF-IDF + length + DeBERTa
#
# Runs LOCALLY (no GPU needed) -- combines already-computed probabilities.
# Requires: after any DeBERTa experiment on Kaggle, download
# /kaggle/working/val_proba_<experiment_name>.npy and
# /kaggle/working/val_ids_<experiment_name>.csv into outputs/deberta_preds/,
# then set DEBERTA_EXPERIMENT below to match.
#
# Why local: ensembling is just combining two already-computed probability
# arrays -- it costs zero additional GPU time, so there's no reason to do
# this search on Kaggle.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.data import load_train, group_split, label_to_class, TARGET_COLS
from src.preprocessing import build_compare_text, render_prompt_cell

SEED = 42
DEBERTA_EXPERIMENT = None  # e.g. "exp1_lr1e-5" -- set once you've downloaded predictions
PRED_DIR = Path(__file__).resolve().parent.parent / "outputs" / "deberta_preds"

train = load_train()
tr_df, va_df = group_split(train, test_size=0.1, seed=SEED)

tr_text = [build_compare_text(p, a, b) for p, a, b in zip(tr_df.prompt, tr_df.response_a, tr_df.response_b)]
va_text = [build_compare_text(p, a, b) for p, a, b in zip(va_df.prompt, va_df.response_a, va_df.response_b)]
y_tr = label_to_class(tr_df).to_numpy()
y_va = label_to_class(va_df).to_numpy()

# --- TF-IDF ---
vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X_tr = vectorizer.fit_transform(tr_text)
X_va = vectorizer.transform(va_text)
clf_tfidf = LogisticRegression(max_iter=200, C=0.1, random_state=SEED)
clf_tfidf.fit(X_tr, y_tr)
proba_tfidf = clf_tfidf.predict_proba(X_va)

# --- length-only ---
def char_len(raw):
    return len(render_prompt_cell(raw))
len_tr = np.array([[char_len(a), char_len(b), char_len(a) - char_len(b)] for a, b in zip(tr_df.response_a, tr_df.response_b)])
len_va = np.array([[char_len(a), char_len(b), char_len(a) - char_len(b)] for a, b in zip(va_df.response_a, va_df.response_b)])
clf_len = LogisticRegression(max_iter=500, random_state=SEED)
clf_len.fit(len_tr, y_tr)
proba_len = clf_len.predict_proba(len_va)

print(f"TF-IDF alone:      {log_loss(y_va, proba_tfidf, labels=[0,1,2]):.4f}")
print(f"length-only alone: {log_loss(y_va, proba_len, labels=[0,1,2]):.4f}")


def two_way_sweep(name_a, proba_a, name_b, proba_b):
    best_ll, best_alpha = 999, None
    for alpha in np.arange(0.0, 1.01, 0.05):
        blend = alpha * proba_a + (1 - alpha) * proba_b
        ll = log_loss(y_va, blend, labels=[0, 1, 2])
        if ll < best_ll:
            best_ll, best_alpha = ll, alpha
    print(f"best blend {name_a}*a + {name_b}*(1-a): alpha={best_alpha:.2f}  log_loss={best_ll:.4f}")
    return best_alpha, best_ll


print("\n--- TF-IDF + length ---")
alpha_tl, ll_tl = two_way_sweep("tfidf", proba_tfidf, "length", proba_len)

if DEBERTA_EXPERIMENT is None:
    print(
        "\nDEBERTA_EXPERIMENT not set -- skipping 3-way blend. "
        "Download val_proba_<exp>.npy + val_ids_<exp>.csv from Kaggle's "
        "/kaggle/working/ into outputs/deberta_preds/, then set "
        "DEBERTA_EXPERIMENT and re-run."
    )
else:
    proba_deberta_raw = np.load(PRED_DIR / f"val_proba_{DEBERTA_EXPERIMENT}.npy")
    deberta_ids = pd.read_csv(PRED_DIR / f"val_ids_{DEBERTA_EXPERIMENT}.csv")["id"].to_numpy()

    # align DeBERTa's val predictions to THIS script's va_df row order by id
    # (both came from the same group_split with the same seed/test_size, so
    # the SET of ids should match exactly -- this assert catches any drift,
    # e.g. if CONFIG["val_fraction"] or the seed ever differs between the
    # Kaggle script and this one).
    id_to_row = {vid: i for i, vid in enumerate(deberta_ids)}
    assert set(id_to_row) == set(va_df["id"]), (
        "val id sets don't match between local split and Kaggle experiment -- "
        "check both used group_split(seed=42, test_size=0.1) on the same train.csv"
    )
    reorder = [id_to_row[vid] for vid in va_df["id"]]
    proba_deberta = proba_deberta_raw[reorder]

    print(f"\nDeBERTa ({DEBERTA_EXPERIMENT}) alone: {log_loss(y_va, proba_deberta, labels=[0,1,2]):.4f}")

    print("\n--- DeBERTa + length ---")
    two_way_sweep("deberta", proba_deberta, "length", proba_len)

    print("\n--- DeBERTa + TF-IDF ---")
    two_way_sweep("deberta", proba_deberta, "tfidf", proba_tfidf)

    print("\n--- 3-way grid: w_deberta + w_tfidf + w_length = 1 ---")
    best_ll, best_w = 999, None
    step = 0.05
    for w_d in np.arange(0, 1.0001, step):
        for w_t in np.arange(0, 1.0001 - w_d, step):
            w_l = 1 - w_d - w_t
            blend = w_d * proba_deberta + w_t * proba_tfidf + w_l * proba_len
            ll = log_loss(y_va, blend, labels=[0, 1, 2])
            if ll < best_ll:
                best_ll, best_w = ll, (round(w_d, 2), round(w_t, 2), round(w_l, 2))
    print(f"best (w_deberta, w_tfidf, w_length) = {best_w}  log_loss={best_ll:.4f}")
    print(f"vs DeBERTa alone: delta = {best_ll - log_loss(y_va, proba_deberta, labels=[0,1,2]):+.4f}")
    print(f"vs current best (TF-IDF+length, {ll_tl:.4f}): delta = {best_ll - ll_tl:+.4f}")
