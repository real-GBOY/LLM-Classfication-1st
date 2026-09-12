# %% [markdown]
# # Error analysis for the DeBERTa-v3-extra-small run
#
# Run this AFTER notebooks/03_deberta_train.py in the same Kaggle session (it
# reuses tr_df/va_df/va_text/best_model/val_proba from that script's memory).
# If running in a fresh session, first reload the checkpoint from
# /kaggle/working/deberta_xsmall_best.keras and recompute val_proba as shown
# in 03's "Validation reporting" cell.
#
# Purpose: find out WHY the model is wrong, to decide which of the stage-11
# improvement experiments (truncation, formatting, class weighting, etc.) is
# actually worth trying, instead of guessing.

# %%
import numpy as np
import pandas as pd

CLASS_NAMES = ["A wins", "B wins", "tie"]

analysis_df = va_df.copy()
analysis_df["pred_class"] = val_proba.argmax(axis=1)
analysis_df["true_class"] = y_va
analysis_df["p_a"] = val_proba[:, 0]
analysis_df["p_b"] = val_proba[:, 1]
analysis_df["p_tie"] = val_proba[:, 2]
analysis_df["confidence"] = val_proba.max(axis=1)
analysis_df["correct"] = analysis_df["pred_class"] == analysis_df["true_class"]
analysis_df["example_logloss"] = -np.log(np.clip(val_proba[np.arange(len(va_df)), y_va], 1e-15, 1.0))
analysis_df["total_chars"] = (
    va_df["prompt"].str.len() + va_df["response_a"].str.len() + va_df["response_b"].str.len()
)

print(f"overall accuracy: {analysis_df['correct'].mean():.4f}")
print(f"overall mean per-example log loss: {analysis_df['example_logloss'].mean():.4f}")

# %% [markdown]
# ## Confusion matrix

# %%
confusion = pd.crosstab(
    analysis_df["true_class"].map(dict(enumerate(CLASS_NAMES))),
    analysis_df["pred_class"].map(dict(enumerate(CLASS_NAMES))),
    rownames=["true"], colnames=["pred"],
)
print(confusion)

# %% [markdown]
# ## High-confidence incorrect predictions
# These are the most informative errors -- the model was CERTAIN and wrong,
# meaning it latched onto a misleading pattern rather than being uncertain
# on a genuinely ambiguous example.

# %%
high_conf_wrong = analysis_df[~analysis_df["correct"]].sort_values("confidence", ascending=False).head(15)
for _, row in high_conf_wrong.iterrows():
    print(f"\nid={row['id']}  true={CLASS_NAMES[row['true_class']]}  pred={CLASS_NAMES[row['pred_class']]}  "
          f"conf={row['confidence']:.3f}  chars={row['total_chars']}")
    print("  prompt:", str(row["prompt"])[:150])

# %% [markdown]
# ## A-vs-B confusion specifically (excluding ties)
# Checks whether errors are symmetric (equally likely to mistake A-wins for
# B-wins and vice versa) or whether the model has a systematic position bias
# -- e.g. always leaning toward predicting A regardless of content.

# %%
non_tie = analysis_df[analysis_df["true_class"] != 2]
a_true = non_tie[non_tie["true_class"] == 0]
b_true = non_tie[non_tie["true_class"] == 1]
print(f"When true=A wins: predicted A {np.mean(a_true['pred_class']==0):.2%}, "
      f"B {np.mean(a_true['pred_class']==1):.2%}, tie {np.mean(a_true['pred_class']==2):.2%}")
print(f"When true=B wins: predicted A {np.mean(b_true['pred_class']==0):.2%}, "
      f"B {np.mean(b_true['pred_class']==1):.2%}, tie {np.mean(b_true['pred_class']==2):.2%}")

# %% [markdown]
# ## Tie predictions specifically
# Ties are the hardest class (smallest, most subjective). Check both how
# often we predict tie when we shouldn't, and how often we miss a real tie.

# %%
tie_true = analysis_df[analysis_df["true_class"] == 2]
print(f"Recall on true ties: {np.mean(tie_true['pred_class']==2):.2%} "
      f"({np.mean(tie_true['pred_class']==2)*len(tie_true):.0f}/{len(tie_true)})")
tie_pred = analysis_df[analysis_df["pred_class"] == 2]
print(f"Precision on predicted ties: {np.mean(tie_pred['true_class']==2):.2%}")

# %% [markdown]
# ## Very long examples
# Checks whether truncation is actually hurting accuracy on the longest
# conversations -- if error rate climbs sharply with length, the truncation
# budget (CONFIG["max_words"]) is a lever worth pulling in stage 11.

# %%
analysis_df["length_bucket"] = pd.qcut(analysis_df["total_chars"], q=5, labels=["shortest", "short", "medium", "long", "longest"])
print(analysis_df.groupby("length_bucket", observed=True).agg(
    n=("correct", "size"),
    accuracy=("correct", "mean"),
    mean_logloss=("example_logloss", "mean"),
))

# %% [markdown]
# ## Errors by model pair
# Uses model_a/model_b -- available in train/val (NOT usable as a model
# input feature since test.csv lacks it) but perfectly fine for diagnostics:
# are certain model matchups systematically harder to judge?

# %%
analysis_df["pair"] = analysis_df["model_a"] + " vs " + analysis_df["model_b"]
pair_stats = analysis_df.groupby("pair").agg(n=("correct", "size"), accuracy=("correct", "mean"), mean_logloss=("example_logloss", "mean"))
pair_stats = pair_stats[pair_stats["n"] >= 20].sort_values("mean_logloss", ascending=False)
print("\nHardest model pairs (min 20 examples, highest log loss):")
print(pair_stats.head(15))
