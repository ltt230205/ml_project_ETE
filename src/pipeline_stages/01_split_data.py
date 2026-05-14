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

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    config = load_config()

    raw_path = config["raw_data"]["path"]
    target = config["preprocess"]["target_column"]

    train_raw_path = config["staged_pipeline"]["train_raw_path"]
    test_raw_path = config["staged_pipeline"]["test_raw_path"]
    split_summary_path = config["staged_pipeline"]["split_summary_path"]

    test_size = float(config.get("split", {}).get("test_size", 0.2))
    random_state = int(config.get("split", {}).get("random_state", 42))

    df = pd.read_csv(raw_path)

    id_cols = [col for col in df.columns if "id" in col.lower()]
    if id_cols:
        df = df.drop(columns=id_cols)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in raw data.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() == 2 else None,
    )

    train_df = X_train.copy()
    train_df[target] = y_train

    test_df = X_test.copy()
    test_df[target] = y_test

    ensure_parent(train_raw_path)
    ensure_parent(test_raw_path)

    train_df.to_csv(train_raw_path, index=False)
    test_df.to_csv(test_raw_path, index=False)

    summary = {
        "raw_path": raw_path,
        "target": target,
        "id_cols_dropped": id_cols,
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_path": train_raw_path,
        "test_path": test_raw_path,
    }
    write_json(summary, split_summary_path)

    print("✅ Stage split_data done")
    print(f"Train raw: {train_raw_path}")
    print(f"Test raw: {test_raw_path}")


if __name__ == "__main__":
    main()
