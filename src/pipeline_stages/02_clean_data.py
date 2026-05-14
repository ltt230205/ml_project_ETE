import json
import os
from pathlib import Path

import yaml


def load_config() -> dict:
    params_path = os.getenv("PARAMS_PATH", "params_staged_pipeline.yaml")
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

import numpy as np
import pandas as pd


def fit_cleaning_stats(train_df: pd.DataFrame, target: str) -> dict:
    feature_df = train_df.drop(columns=[target])

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    stats = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric": {},
        "categorical": {},
    }

    for col in numeric_cols:
        median = float(train_df[col].median())
        q1 = float(train_df[col].quantile(0.25))
        q3 = float(train_df[col].quantile(0.75))
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        stats["numeric"][col] = {
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": float(lower),
            "upper": float(upper),
        }

    for col in categorical_cols:
        mode_series = train_df[col].mode(dropna=True)
        fill_value = str(mode_series.iloc[0]) if len(mode_series) > 0 else "unknown"

        stats["categorical"][col] = {
            "fill_value": fill_value,
        }

    return stats


def apply_cleaning(df: pd.DataFrame, target: str, stats: dict) -> pd.DataFrame:
    out = df.copy()

    for col, col_stats in stats["numeric"].items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(col_stats["median"])
            out[col] = np.clip(out[col], col_stats["lower"], col_stats["upper"])

    for col, col_stats in stats["categorical"].items():
        if col in out.columns:
            out[col] = out[col].astype("object")
            out[col] = out[col].fillna(col_stats["fill_value"])
            out[col] = out[col].astype(str)

    return out


def main():
    config = load_config()
    target = config["preprocess"]["target_column"]

    train_raw_path = config["staged_pipeline"]["train_raw_path"]
    test_raw_path = config["staged_pipeline"]["test_raw_path"]

    train_clean_path = config["staged_pipeline"]["train_clean_path"]
    test_clean_path = config["staged_pipeline"]["test_clean_path"]
    cleaning_stats_path = config["staged_pipeline"]["cleaning_stats_path"]

    train_df = pd.read_csv(train_raw_path)
    test_df = pd.read_csv(test_raw_path)

    stats = fit_cleaning_stats(train_df, target)

    train_clean = apply_cleaning(train_df, target, stats)
    test_clean = apply_cleaning(test_df, target, stats)

    ensure_parent(train_clean_path)
    ensure_parent(test_clean_path)

    train_clean.to_csv(train_clean_path, index=False)
    test_clean.to_csv(test_clean_path, index=False)

    write_json(stats, cleaning_stats_path)

    print("✅ Stage clean_data done")
    print(f"Train clean: {train_clean_path}")
    print(f"Test clean: {test_clean_path}")
    print(f"Cleaning stats: {cleaning_stats_path}")


if __name__ == "__main__":
    main()
