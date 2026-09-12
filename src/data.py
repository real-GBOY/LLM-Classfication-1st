"""Data loading and parsing utilities for the LLM Classification Finetuning competition.

Kaggle paths (used when running in a Kaggle notebook):
    /kaggle/input/competitions/llm-classification-finetuning/train.csv
    /kaggle/input/competitions/llm-classification-finetuning/test.csv

(Confirmed on Kaggle 2026-09: competition inputs are nested under
/kaggle/input/competitions/<slug>/, not directly under /kaggle/input/<slug>/
as older Kaggle docs/examples show. We check both since this may vary by
notebook environment.)

Locally we point DATA_DIR at the project root instead.
"""
import json
import ast
from pathlib import Path

import pandas as pd

TARGET_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]

_SLUG = "llm-classification-finetuning"


def default_data_dir() -> Path:
    for candidate in (
        Path("/kaggle/input/competitions") / _SLUG,
        Path("/kaggle/input") / _SLUG,
    ):
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parent.parent


def parse_turns(raw):
    """Parse a train/test 'prompt'/'response_*' cell into a list[str | None].

    Cells are stored as JSON-encoded lists of strings (one entry per
    conversation turn). A small number of response turns are JSON `null`
    where the underlying model's output was content-filtered by the
    original Chatbot Arena logging -- we keep them as None and let the
    caller decide how to render them.
    """
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw]
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def load_train(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or default_data_dir()
    df = pd.read_csv(Path(data_dir) / "train.csv")
    return df


def load_test(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or default_data_dir()
    df = pd.read_csv(Path(data_dir) / "test.csv")
    return df


def add_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds *_turns list columns parsed from the raw JSON-string columns."""
    df = df.copy()
    df["prompt_turns"] = df["prompt"].apply(parse_turns)
    df["response_a_turns"] = df["response_a"].apply(parse_turns)
    df["response_b_turns"] = df["response_b"].apply(parse_turns)
    return df


def label_to_class(df: pd.DataFrame) -> pd.Series:
    """Collapses the three one-hot target columns into a single class label
    (0=model_a, 1=model_b, 2=tie), matching the column order the model's
    softmax output must follow.
    """
    arr = df[TARGET_COLS].to_numpy()
    return pd.Series(arr.argmax(axis=1), index=df.index, name="label")


def group_split(df: pd.DataFrame, test_size: float = 0.1, seed: int = 42):
    """Splits train into (train_df, val_df), keeping all rows that share the
    same raw `prompt` string on the same side of the split.

    Why grouped instead of a plain stratified split: 3,118 unique prompts in
    train.csv are repeated across 2-13 rows (same prompt judged against
    different model pairs). A plain random/stratified split would leak the
    same prompt into both train and val, letting the model partially
    "recognize" a validation prompt it saw in training and making the
    validation log loss look better than true generalization performance.
    """
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df["prompt"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


if __name__ == "__main__":
    train = load_train()
    test = load_test()
    print("train shape:", train.shape)
    print("test shape:", test.shape)

    tr, va = group_split(train)
    print("split ->", tr.shape, va.shape)
    overlap = set(tr["prompt"]).intersection(set(va["prompt"]))
    print("prompt overlap between train/val (should be 0):", len(overlap))

    label_dist_tr = label_to_class(tr).value_counts(normalize=True).sort_index()
    label_dist_va = label_to_class(va).value_counts(normalize=True).sort_index()
    print("\nlabel distribution train:\n", label_dist_tr)
    print("\nlabel distribution val:\n", label_dist_va)
