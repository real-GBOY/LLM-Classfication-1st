# %% [markdown]
# # EDA: LLM Classification Finetuning
#
# Goal: understand class balance, response length effects, and model-vs-model
# win patterns before designing the DeBERTa input format. Every chart here is
# meant to answer a modeling question, not just describe the data for its own
# sake.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_train, add_parsed_columns, TARGET_COLS

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

train = load_train()
train = add_parsed_columns(train)
print("train shape:", train.shape)

# %% [markdown]
# ## 1. Class balance
# Confirms the target is roughly balanced (~35/34/31), so we don't need
# aggressive class weighting, but log loss still rewards getting the tie
# class calibrated correctly since it's the smallest.

# %%
class_counts = train[TARGET_COLS].sum()
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["model_a", "model_b", "tie"], class_counts.values, color=["#4C72B0", "#DD8452", "#55A868"])
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 300, f"{v}\n({v/len(train):.1%})", ha="center")
ax.set_title("Class balance")
ax.set_ylabel("count")
fig.tight_layout()
fig.savefig(OUT_DIR / "01_class_balance.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 2. Length distributions
# We compute character length of the joined turns for prompt, response_a,
# response_b. This directly informs max_sequence_length for DeBERTa and
# whether truncation will hurt the prompt or the responses more.

# %%
def joined_len(turns):
    return sum(len(t) for t in turns if isinstance(t, str))

train["prompt_len"] = train["prompt_turns"].apply(joined_len)
train["resp_a_len"] = train["response_a_turns"].apply(joined_len)
train["resp_b_len"] = train["response_b_turns"].apply(joined_len)
train["total_len"] = train["prompt_len"] + train["resp_a_len"] + train["resp_b_len"]

print(train[["prompt_len", "resp_a_len", "resp_b_len", "total_len"]].describe())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(train["prompt_len"].clip(upper=3000), bins=60, alpha=0.6, label="prompt")
axes[0].hist(train["resp_a_len"].clip(upper=3000), bins=60, alpha=0.6, label="response_a")
axes[0].hist(train["resp_b_len"].clip(upper=3000), bins=60, alpha=0.6, label="response_b")
axes[0].set_title("Char length (clipped at 3000)")
axes[0].set_xlabel("characters")
axes[0].legend()

axes[1].hist(train["total_len"].clip(upper=15000), bins=60, color="#55A868")
for q in [0.5, 0.9, 0.95, 0.99]:
    val = train["total_len"].quantile(q)
    axes[1].axvline(val, linestyle="--", color="black", linewidth=0.8)
    axes[1].text(val, axes[1].get_ylim()[1] * 0.9, f"p{int(q*100)}", rotation=90, fontsize=8)
axes[1].set_title("Total example length (prompt+A+B, clipped at 15000 chars)")
axes[1].set_xlabel("characters")
fig.tight_layout()
fig.savefig(OUT_DIR / "02_length_distributions.png", dpi=150)
plt.close(fig)

print("\ntotal_len quantiles:")
for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"  p{int(q*100)}: {train['total_len'].quantile(q):.0f} chars")

# %% [markdown]
# ## 3. Response length vs winner
# Checks whether length alone predicts the winner -- a real, legitimate but
# potentially confounding signal we need to be aware of during error analysis
# (a model that just learns "longer wins" will plateau at a mediocre log
# loss).

# %%
train["len_diff"] = train["resp_a_len"] - train["resp_b_len"]
label_map = {0: "A wins", 1: "B wins", 2: "tie"}
train["winner_label"] = train[TARGET_COLS].to_numpy().argmax(axis=1)
train["winner_label"] = train["winner_label"].map(label_map)

fig, ax = plt.subplots(figsize=(6, 4))
data = [train.loc[train.winner_label == lbl, "len_diff"].clip(-3000, 3000) for lbl in ["A wins", "B wins", "tie"]]
ax.boxplot(data, tick_labels=["A wins", "B wins", "tie"], showfliers=False)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_ylabel("len(response_a) - len(response_b), chars (clipped)")
ax.set_title("Response length difference vs winner")
fig.tight_layout()
fig.savefig(OUT_DIR / "03_length_vs_winner.png", dpi=150)
plt.close(fig)

non_tie = train[train.winner_label != "tie"].copy()
non_tie["longer_is_a"] = non_tie["len_diff"] > 0
non_tie["a_won"] = non_tie["winner_label"] == "A wins"
longer_win_rate = (non_tie["longer_is_a"] == non_tie["a_won"]).mean()
print(f"\nFraction of non-tie examples where the LONGER response wins: {longer_win_rate:.1%}")

# %% [markdown]
# ## 4. Model-vs-model win patterns
# Aggregates win rate per model (as model_a or model_b) to sanity-check that
# some models are objectively stronger -- useful context even though we
# CANNOT use model identity as a feature at inference time (test.csv doesn't
# have model_a/model_b columns).

# %%
def win_rate_table(df):
    rows = []
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    for m in models:
        as_a = df[df.model_a == m]
        as_b = df[df.model_b == m]
        wins = as_a["winner_model_a"].sum() + as_b["winner_model_b"].sum()
        total = len(as_a) + len(as_b)
        if total >= 200:
            rows.append({"model": m, "n_appearances": total, "win_rate": wins / total})
    return pd.DataFrame(rows).sort_values("win_rate", ascending=False)

wr = win_rate_table(train)
print("\nTop 10 models by win rate (min 200 appearances):")
print(wr.head(10).to_string(index=False))
print("\nBottom 10 models by win rate:")
print(wr.tail(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 8))
top_bottom = pd.concat([wr.head(12), wr.tail(12)])
colors = ["#55A868" if x >= 0.5 else "#C44E52" for x in top_bottom["win_rate"]]
ax.barh(top_bottom["model"], top_bottom["win_rate"], color=colors)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("win rate (excludes ties from denominator... see note)")
ax.set_title("Model win rate: top 12 and bottom 12 (min 200 appearances)")
fig.tight_layout()
fig.savefig(OUT_DIR / "04_model_win_rates.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 5. Duplicate / leakage checks
# Confirms the group-split decision made earlier: prompts repeat across rows
# and must be kept together across train/val.

# %%
prompt_counts = train.groupby("prompt").size()
print(f"\nUnique prompts: {len(prompt_counts)} / {len(train)} rows")
print(f"Prompts appearing 2+ times: {(prompt_counts >= 2).sum()}")

dup_content = train.duplicated(subset=["prompt", "response_a", "response_b"], keep=False)
print(f"Rows with identical (prompt, response_a, response_b) elsewhere in train: {dup_content.sum()}")

print(f"\nAll figures saved to: {OUT_DIR}")
