# %% [markdown]
# # DeBERTa-v3-extra-small fine-tuning
#
# Run this in a KAGGLE NOTEBOOK with a GPU accelerator (T4 or P100) attached.
# Before running:
#   1. Add Input -> Competition -> "LLM Classification Finetuning"
#   2. Add Input -> Models -> search "deberta_v3", framework "keras",
#      variation "deberta_v3_extra_small_en" (handle: deberta_v3/keras/deberta_v3_extra_small_en)
#   3. Settings -> Accelerator -> GPU T4 x2 (or P100)
#
# This script is intentionally self-contained (data/preprocessing helpers are
# duplicated from src/ below, not imported) so it can be pasted into a fresh
# Kaggle notebook without uploading the rest of the repo. Keep it in sync
# with src/data.py and src/preprocessing.py if those change.
#
# Mirrors configs/deberta_base.yaml -- update both if you change a default.

# %%
import json
import ast
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import keras_hub as kh
except ImportError:
    import keras_nlp as kh  # older Kaggle images name the package keras_nlp

import keras
import tensorflow as tf

CONFIG = dict(
    preset="deberta_v3_extra_small_en",
    max_words=340,
    sequence_length=512,   # verified against the loaded preprocessor below -- adjust if it prints something else
    prompt_frac=0.2,
    response_frac=0.4,
    seed=42,
    val_fraction=0.1,
    batch_size=16,
    epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    label_smoothing=0.0,
    mixed_precision=True,
)

def _resolve_data_dir():
    # Kaggle nests competition inputs under /kaggle/input/competitions/<slug>/
    # (confirmed 2026-09) rather than directly under /kaggle/input/<slug>/ as
    # older examples show -- check both so this survives environment changes.
    for candidate in (
        Path("/kaggle/input/competitions/llm-classification-finetuning"),
        Path("/kaggle/input/llm-classification-finetuning"),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("competition input dir not found under /kaggle/input -- check 'Add Input' attached the competition dataset")


DATA_DIR = _resolve_data_dir()
WORK_DIR = Path("/kaggle/working")
TARGET_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]

keras.utils.set_random_seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print("GPUs visible:", tf.config.list_physical_devices("GPU"))
if CONFIG["mixed_precision"] and tf.config.list_physical_devices("GPU"):
    keras.mixed_precision.set_global_policy("mixed_float16")
    print("mixed precision enabled: mixed_float16")
else:
    print("mixed precision NOT enabled (no GPU detected or disabled in config)")

# %% [markdown]
# ## Data + text-building helpers (duplicated from src/, see module docstring)

# %%
def parse_turns(raw):
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw]
    return parsed if isinstance(parsed, list) else [parsed]


NULL_TURN_PLACEHOLDER = "[NO RESPONSE]"
TRUNCATED_MARKER = " ... [TRUNCATED] ... "


def render_turns(turns):
    parts = [t if isinstance(t, str) else NULL_TURN_PLACEHOLDER for t in turns]
    return " ".join(parts)


def _truncate_words_head_tail(text, word_budget, head_frac=0.7):
    words = text.split()
    if len(words) <= word_budget:
        return text
    head_n = max(1, int(word_budget * head_frac))
    tail_n = max(0, word_budget - head_n)
    if tail_n == 0:
        return " ".join(words[:head_n])
    return " ".join(words[:head_n]) + TRUNCATED_MARKER + " ".join(words[-tail_n:])


def _truncate_prompt_turns(turns, word_budget):
    rendered = [t if isinstance(t, str) else NULL_TURN_PLACEHOLDER for t in turns]
    kept, remaining = [], word_budget
    for turn in reversed(rendered):
        turn_words = turn.split()
        if len(turn_words) <= remaining:
            kept.append(turn)
            remaining -= len(turn_words)
        else:
            if remaining > 0:
                kept.append(_truncate_words_head_tail(turn, remaining, head_frac=1.0))
            break
    kept.reverse()
    return " ".join(kept)


def build_truncated_compare_text(prompt_raw, response_a_raw, response_b_raw, max_words, prompt_frac, response_frac):
    prompt_budget = int(max_words * prompt_frac)
    response_budget = int(max_words * response_frac)
    prompt = _truncate_prompt_turns(parse_turns(prompt_raw), prompt_budget)
    resp_a = _truncate_words_head_tail(render_turns(parse_turns(response_a_raw)), response_budget)
    resp_b = _truncate_words_head_tail(render_turns(parse_turns(response_b_raw)), response_budget)
    return f"[PROMPT] {prompt} [RESPONSE A] {resp_a} [RESPONSE B] {resp_b}"


def group_split(df, test_size, seed):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df["prompt"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


# %%
train_df = pd.read_csv(DATA_DIR / "train.csv")
print("train shape:", train_df.shape)

tr_df, va_df = group_split(train_df, test_size=CONFIG["val_fraction"], seed=CONFIG["seed"])
overlap = set(tr_df["prompt"]).intersection(set(va_df["prompt"]))
assert len(overlap) == 0, "leakage: prompts overlap between train/val"
print(f"train: {tr_df.shape}, val: {va_df.shape}, prompt overlap: {len(overlap)}")

t0 = time.time()
tr_text = [
    build_truncated_compare_text(p, a, b, CONFIG["max_words"], CONFIG["prompt_frac"], CONFIG["response_frac"])
    for p, a, b in zip(tr_df.prompt, tr_df.response_a, tr_df.response_b)
]
va_text = [
    build_truncated_compare_text(p, a, b, CONFIG["max_words"], CONFIG["prompt_frac"], CONFIG["response_frac"])
    for p, a, b in zip(va_df.prompt, va_df.response_a, va_df.response_b)
]
print(f"text building took {time.time()-t0:.1f}s")

y_tr = tr_df[TARGET_COLS].to_numpy().argmax(axis=1)
y_va = va_df[TARGET_COLS].to_numpy().argmax(axis=1)

# %% [markdown]
# ## 6. Tokenization
#
# What: load the DeBERTa-v3-extra-small preprocessor (tokenizer + packing
# into [CLS]/[SEP] format) and confirm its true sequence_length/vocab size
# instead of assuming 512 from memory.
# Why: if the preset's actual max differs from our assumption, CONFIG["max_words"]
# (which was chosen to fit under an assumed 512-token cap) needs adjusting
# BEFORE we waste a training run on badly-truncated inputs.

# %%
preprocessor = kh.models.TextClassifierPreprocessor.from_preset(
    CONFIG["preset"], sequence_length=CONFIG["sequence_length"]
)
print("tokenizer vocab size:", preprocessor.tokenizer.vocabulary_size())
print("configured sequence_length:", preprocessor.sequence_length)

# sanity check: tokenize a few examples and inspect actual token counts,
# to see how often we're hitting the truncation ceiling.
sample_out = preprocessor(tr_text[:200])
token_ids = sample_out["token_ids"] if isinstance(sample_out, dict) else sample_out[0]["token_ids"]
pad_id = preprocessor.tokenizer.pad_token_id if hasattr(preprocessor.tokenizer, "pad_token_id") else 0
lengths = np.sum(token_ids.numpy() != pad_id, axis=1)
print(f"sample token-length stats (n=200): mean={lengths.mean():.0f}, p90={np.percentile(lengths,90):.0f}, max={lengths.max()}")
print(f"fraction hitting the {CONFIG['sequence_length']}-token ceiling: {(lengths >= CONFIG['sequence_length']).mean():.1%}")

# %% [markdown]
# ## 7. Model
#
# What: load the pretrained DeBERTa-v3-extra-small backbone with a fresh
# 3-way classification head on top (via the keras_hub Task API), configured
# to output raw logits (activation=None) rather than probabilities.
# Why logits, not softmax, in the model itself: `from_logits=True` losses
# are numerically more stable (avoids computing log(softmax(x)) as two
# lossy floating-point steps), which matters more once mixed_float16 is on.
# We apply softmax explicitly at prediction time to get P(A), P(B), P(tie).

# %%
model = kh.models.TextClassifier.from_preset(
    CONFIG["preset"],
    preprocessor=preprocessor,
    num_classes=3,
    activation=None,  # logits
)
model.summary()

# %% [markdown]
# ## 8. Fine-tuning
#
# Conservative defaults for a small encoder on ~50k rows: AdamW, small LR
# (2e-5) with linear warmup then decay, weight decay 0.01, 2 epochs. Loss is
# SparseCategoricalCrossentropy(from_logits=True) evaluated on integer class
# ids 0/1/2 -- its mean over a batch IS the competition's multiclass log
# loss (natural log, same formula), so val_loss during training already
# tracks what we're being scored on; we don't need a custom metric.

# %%
steps_per_epoch = len(tr_text) // CONFIG["batch_size"]
total_steps = steps_per_epoch * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=CONFIG["learning_rate"],
    decay_steps=max(1, total_steps - warmup_steps),
    warmup_target=CONFIG["learning_rate"],
    warmup_steps=warmup_steps,
)

optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=CONFIG["weight_decay"],
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, label_smoothing=CONFIG["label_smoothing"]
    ),
    metrics=["sparse_categorical_accuracy"],
)

checkpoint_path = WORK_DIR / "deberta_xsmall_best.keras"
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",   # log loss, NOT accuracy -- competition metric
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=1, restore_best_weights=True),
]

t0 = time.time()
history = model.fit(
    x=tr_text,
    y=y_tr,
    validation_data=(va_text, y_va),
    batch_size=CONFIG["batch_size"],
    epochs=CONFIG["epochs"],
    callbacks=callbacks,
)
train_time = time.time() - t0
print(f"\ntraining took {train_time/60:.1f} min")

# %% [markdown]
# ## 9. Validation reporting
#
# What: report val log loss (= best val_loss from training), val accuracy,
# runtime, and confirm the saved checkpoint is the best-log-loss one, not
# just the last epoch.

# %%
best_epoch = int(np.argmin(history.history["val_loss"]))
print(f"best epoch: {best_epoch+1}/{CONFIG['epochs']}")
print(f"train loss @ best epoch: {history.history['loss'][best_epoch]:.4f}")
print(f"val loss @ best epoch (= val log loss): {history.history['val_loss'][best_epoch]:.4f}")
print(f"val accuracy @ best epoch: {history.history['val_sparse_categorical_accuracy'][best_epoch]:.4f}")
print(f"total training time: {train_time/60:.1f} min for {CONFIG['epochs']} epoch(s)")

print("\nreference log losses from the TF-IDF baseline notebook (outputs/baseline/results.txt):")
print("  naive class-prior: 1.0961")
print("  length-features-only: 1.0747")
print("  TF-IDF + LogReg (C=0.1): 1.0803")
print(f"  DeBERTa-v3-xsmall (this run): {history.history['val_loss'][best_epoch]:.4f}")

# reload best checkpoint explicitly (EarlyStopping already restores best
# weights in-memory, but this confirms the saved file on disk is usable)
best_model = keras.models.load_model(checkpoint_path)
val_logits = best_model.predict(va_text, batch_size=CONFIG["batch_size"] * 2)
val_proba = tf.nn.softmax(val_logits, axis=-1).numpy()

from sklearn.metrics import log_loss as sk_log_loss
recomputed_ll = sk_log_loss(y_va, val_proba, labels=[0, 1, 2])
print(f"\nrecomputed val log loss from saved checkpoint (should match above): {recomputed_ll:.4f}")

assert not np.isnan(val_proba).any()
assert np.allclose(val_proba.sum(axis=1), 1.0, atol=1e-3)
print("checkpoint saved to:", checkpoint_path)
