# %% [markdown]
# # Test prediction + submission
#
# Run this in the SAME Kaggle session as 03_deberta_train.py (reuses
# best_model, CONFIG, and the text-building helpers from memory), or reload
# the checkpoint fresh with:
#   best_model = keras.models.load_model("/kaggle/working/deberta_xsmall_best.keras")
#
# NOTE ON RETRAINING FOR THE FINAL SUBMISSION: this script predicts using the
# model trained on the 90% train split from 03_deberta_train.py. Once you've
# picked a final config via the stage-11 experiments, the standard last step
# is to retrain on 100% of train.csv (train+val combined) with the same
# config so the submitted model has seen the most data possible -- do this
# only AFTER validation-based model selection is finished, never before,
# or you lose your ability to check the config against a held-out set.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

def _resolve_data_dir():
    for candidate in (
        Path("/kaggle/input/competitions/llm-classification-finetuning"),
        Path("/kaggle/input/llm-classification-finetuning"),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("competition input dir not found under /kaggle/input -- check 'Add Input' attached the competition dataset")


DATA_DIR = _resolve_data_dir()

test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
print("test shape:", test_df.shape)

test_text = [
    build_truncated_compare_text(p, a, b, CONFIG["max_words"], CONFIG["prompt_frac"], CONFIG["response_frac"])
    for p, a, b in zip(test_df.prompt, test_df.response_a, test_df.response_b)
]

# %% [markdown]
# ## Predict
# best_model outputs logits (activation=None, see stage 7) -- apply softmax
# explicitly to get valid probabilities.

# %%
test_logits = best_model.predict(test_text, batch_size=CONFIG["batch_size"] * 2)
test_proba = tf.nn.softmax(test_logits, axis=-1).numpy()

# %% [markdown]
# ## Build + verify submission
# Checks every requirement from the task spec before writing the file:
# correct rows, correct columns in the correct order, no NaNs, valid
# probability ranges, rows summing to ~1, and an exact column-name match
# against sample_submission.csv (not just "looks right").

# %%
submission = pd.DataFrame({
    "id": test_df["id"],
    "winner_model_a": test_proba[:, 0],
    "winner_model_b": test_proba[:, 1],
    "winner_tie": test_proba[:, 2],
})

assert list(submission.columns) == list(sample_sub.columns), \
    f"column mismatch: {list(submission.columns)} vs {list(sample_sub.columns)}"
assert len(submission) == len(sample_sub), \
    f"row count mismatch: {len(submission)} vs {len(sample_sub)}"
assert (submission["id"].values == sample_sub["id"].values).all(), "id order/values mismatch"
assert not submission.isna().any().any(), "NaNs present in submission"

prob_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
assert (submission[prob_cols].to_numpy() >= 0).all() and (submission[prob_cols].to_numpy() <= 1).all(), \
    "probabilities out of [0, 1] range"
row_sums = submission[prob_cols].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-3), f"rows don't sum to 1: min={row_sums.min()}, max={row_sums.max()}"

print("all submission checks passed")
print(submission.head())

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nsaved to /kaggle/working/submission.csv")
