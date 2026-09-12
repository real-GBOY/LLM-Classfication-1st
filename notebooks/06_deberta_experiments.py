# %% [markdown]
# # DeBERTa-v3-extra-small -- controlled experiment runner
#
# Run this in the SAME Kaggle notebook as 03_deberta_train.py (same Input
# attachments: competition dataset + deberta_v3_extra_small_en model, GPU
# accelerator). This is a SEPARATE cell/script from 03 so the first
# successful run (val log loss 1.0784, epoch 1) and its checkpoint are never
# touched -- every experiment here saves to its own checkpoint file and
# appends one row to a JSON-lines experiment log instead of overwriting
# anything.
#
# WHY these specific experiments (not a random grid): the baseline run
# showed train acc UP + val acc UP but val LOG LOSS UP from epoch 1 to
# epoch 2 (1.0784 -> 1.0869). Accuracy improving while log loss worsens is
# the signature of the model becoming overconfident/miscalibrated, not of
# underfitting. All three experiments below are targeted fixes for THAT
# specific failure mode, evidence-first rather than "try random knobs":
#   Exp 1: lower LR      -- tests whether the step size itself is too large
#   Exp 2: gradient clip -- tests whether large individual gradient spikes
#                            (not LR magnitude) are the driver
#   Exp 3: label smoothing -- directly targets overconfidence: it caps how
#                            close to 0/1 the model is allowed to push a
#                            probability, which is exactly what a log-loss-
#                            optimizing overconfidence fix should do
#
# NOT tested, with reasons (avoiding low-value GPU spend per the brief):
#   - Higher LR: evidence points the OPPOSITE direction (current LR already
#     looks too aggressive) -- testing higher LR first would very likely
#     make things worse and burns quota for a low-information result.
#   - sequence_length=768: the DeBERTa-v3-xsmall backbone's own config
#     (confirmed by reading its config.json on Kaggle) has
#     max_sequence_length=512 -- going past that isn't "increasing a
#     setting," it's running the model outside its pretrained position
#     range, which needs position-embedding surgery to do safely. Skipped
#     as infeasible for a first pass, not just low-value.
#   - Input reformatting (Priority 2): local analysis (see chat) shows the
#     achievable signal in this dataset is currently dominated by response
#     length + weak lexical cues (TF-IDF+length blend already beats the
#     current best DeBERTa checkpoint). The bottleneck right now is
#     training dynamics, not whether the model can parse field boundaries
#     in an already-explicit "[PROMPT]...[RESPONSE A]...[RESPONSE B]..."
#     format. Revisit AFTER Exp 1-3 if DeBERTa still underperforms.

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
    import keras_nlp as kh

import keras
import tensorflow as tf

# ============================================================
# EDIT THIS BLOCK ONLY, then re-run the whole script, per experiment.
# ============================================================
EXPERIMENT_NAME = "exp1_lr1e-5"   # <-- change per run: exp1_lr1e-5 / exp2_clipnorm1 / exp3_label_smoothing / exp4_combined
CONFIG = dict(
    preset="deberta_v3_extra_small_en",
    max_words=340,
    sequence_length=512,
    prompt_frac=0.2,
    response_frac=0.4,
    seed=42,               # keep fixed across experiments for a fair comparison
    val_fraction=0.1,
    batch_size=16,
    epochs=2,
    learning_rate=1e-5,    # baseline was 2e-5
    weight_decay=0.01,
    warmup_ratio=0.1,
    clipnorm=None,         # Exp 2: set to 1.0
    label_smoothing=0.0,   # Exp 3: set to 0.05 (switches loss to CategoricalCrossentropy w/ one-hot labels)
    mixed_precision=True,
)
# ============================================================

DATA_DIR = None
for candidate in (
    Path("/kaggle/input/competitions/llm-classification-finetuning"),
    Path("/kaggle/input/llm-classification-finetuning"),
):
    if candidate.exists():
        DATA_DIR = candidate
        break
if DATA_DIR is None:
    raise FileNotFoundError("competition input dir not found under /kaggle/input")

WORK_DIR = Path("/kaggle/working")
LOG_PATH = WORK_DIR / "experiment_log.jsonl"
TARGET_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]

keras.utils.set_random_seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print(f"=== EXPERIMENT: {EXPERIMENT_NAME} ===")
print("GPUs visible:", tf.config.list_physical_devices("GPU"))
if CONFIG["mixed_precision"] and tf.config.list_physical_devices("GPU"):
    keras.mixed_precision.set_global_policy("mixed_float16")

# %% [markdown]
# ## Data + text-building helpers (identical to 03_deberta_train.py)

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
tr_df, va_df = group_split(train_df, test_size=CONFIG["val_fraction"], seed=CONFIG["seed"])
overlap = set(tr_df["prompt"]).intersection(set(va_df["prompt"]))
assert len(overlap) == 0, "leakage: prompts overlap between train/val"
print(f"train: {tr_df.shape}, val: {va_df.shape}, prompt overlap: {len(overlap)}")

tr_text = [
    build_truncated_compare_text(p, a, b, CONFIG["max_words"], CONFIG["prompt_frac"], CONFIG["response_frac"])
    for p, a, b in zip(tr_df.prompt, tr_df.response_a, tr_df.response_b)
]
va_text = [
    build_truncated_compare_text(p, a, b, CONFIG["max_words"], CONFIG["prompt_frac"], CONFIG["response_frac"])
    for p, a, b in zip(va_df.prompt, va_df.response_a, va_df.response_b)
]

y_tr = tr_df[TARGET_COLS].to_numpy().argmax(axis=1)
y_va = va_df[TARGET_COLS].to_numpy().argmax(axis=1)

use_label_smoothing = CONFIG["label_smoothing"] > 0
if use_label_smoothing:
    # CategoricalCrossentropy needs one-hot targets, unlike the sparse
    # integer labels used everywhere else in this project.
    y_tr_fit = tr_df[TARGET_COLS].to_numpy().astype("float32")
    y_va_fit = va_df[TARGET_COLS].to_numpy().astype("float32")
else:
    y_tr_fit = y_tr
    y_va_fit = y_va

# %% [markdown]
# ## Tokenizer (identical setup to 03, sequence_length unchanged at 512)

# %%
preprocessor = kh.models.TextClassifierPreprocessor.from_preset(
    CONFIG["preset"], sequence_length=CONFIG["sequence_length"]
)

# %% [markdown]
# ## Model + compile
#
# activation=None -> logits, matching 03_deberta_train.py. Loss switches to
# CategoricalCrossentropy(label_smoothing=...) only when CONFIG["label_smoothing"] > 0
# (Exp 3); otherwise identical SparseCategoricalCrossentropy(from_logits=True)
# as the baseline run, so Exp 1/2 are true single-variable changes vs baseline.

# %%
model = kh.models.TextClassifier.from_preset(
    CONFIG["preset"], preprocessor=preprocessor, num_classes=3, activation=None,
)

steps_per_epoch = len(tr_text) // CONFIG["batch_size"]
total_steps = steps_per_epoch * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=CONFIG["learning_rate"],
    decay_steps=max(1, total_steps - warmup_steps),
    warmup_target=CONFIG["learning_rate"],
    warmup_steps=warmup_steps,
)

optimizer_kwargs = dict(learning_rate=lr_schedule, weight_decay=CONFIG["weight_decay"])
if CONFIG["clipnorm"] is not None:
    optimizer_kwargs["clipnorm"] = CONFIG["clipnorm"]
optimizer = keras.optimizers.AdamW(**optimizer_kwargs)

if use_label_smoothing:
    loss = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=CONFIG["label_smoothing"])
    acc_metric = "categorical_accuracy"
else:
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric = "sparse_categorical_accuracy"

model.compile(optimizer=optimizer, loss=loss, metrics=[acc_metric])

# %% [markdown]
# ## Fine-tune -- checkpoint path is UNIQUE PER EXPERIMENT, never overwrites

# %%
checkpoint_path = WORK_DIR / f"deberta_xsmall_{EXPERIMENT_NAME}.keras"
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path), monitor="val_loss", mode="min",
        save_best_only=True, verbose=1,
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=1, restore_best_weights=True),
]

t0 = time.time()
history = model.fit(
    x=tr_text, y=y_tr_fit,
    validation_data=(va_text, y_va_fit),
    batch_size=CONFIG["batch_size"], epochs=CONFIG["epochs"], callbacks=callbacks,
)
train_time = time.time() - t0

# %% [markdown]
# ## Report + append to machine-readable experiment log
#
# Every run appends ONE line to /kaggle/working/experiment_log.jsonl instead
# of overwriting a single results file. Download this file (or paste its
# contents back) after each experiment so results accumulate across runs.

# %%
best_epoch = int(np.argmin(history.history["val_loss"]))
best_val_loss = history.history["val_loss"][best_epoch]
best_train_loss = history.history["loss"][best_epoch]
best_val_acc = history.history[f"val_{acc_metric}"][best_epoch]

print(f"\n=== {EXPERIMENT_NAME} RESULTS ===")
print(f"config: {CONFIG}")
print(f"best epoch: {best_epoch+1}/{CONFIG['epochs']}")
print(f"train loss @ best epoch: {best_train_loss:.4f}")
print(f"val log loss @ best epoch: {best_val_loss:.4f}")
print(f"val accuracy @ best epoch: {best_val_acc:.4f}")
print(f"training time: {train_time/60:.1f} min")
print(f"\nvs baseline (1.0784): delta = {best_val_loss - 1.0784:+.4f}")
print(f"vs TF-IDF+length blend (1.0654): delta = {best_val_loss - 1.0654:+.4f}")
print(f"decision: {'KEEP' if best_val_loss < 1.0784 else 'REJECT (worse than baseline)'}")

log_entry = {
    "experiment": EXPERIMENT_NAME,
    "config": CONFIG,
    "best_epoch": best_epoch + 1,
    "train_loss": round(float(best_train_loss), 4),
    "val_log_loss": round(float(best_val_loss), 4),
    "val_accuracy": round(float(best_val_acc), 4),
    "train_time_min": round(train_time / 60, 1),
    "delta_vs_baseline_1_0784": round(float(best_val_loss - 1.0784), 4),
    "checkpoint": str(checkpoint_path),
}
with open(LOG_PATH, "a") as f:
    f.write(json.dumps(log_entry) + "\n")
print(f"\nappended to {LOG_PATH}")

# Also save this experiment's val predictions -- needed for the 3-way
# ensembling test (DeBERTa + TF-IDF + length) once a good checkpoint exists.
best_model = keras.models.load_model(checkpoint_path)
val_logits = best_model.predict(va_text, batch_size=CONFIG["batch_size"] * 2)
val_proba = tf.nn.softmax(val_logits, axis=-1).numpy()
np.save(WORK_DIR / f"val_proba_{EXPERIMENT_NAME}.npy", val_proba)
va_df[["id"]].to_csv(WORK_DIR / f"val_ids_{EXPERIMENT_NAME}.csv", index=False)
print(f"val_proba saved to val_proba_{EXPERIMENT_NAME}.npy ({val_proba.shape}) for later ensembling")
