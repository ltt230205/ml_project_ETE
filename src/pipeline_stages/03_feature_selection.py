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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder


def encode_for_feature_selection(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()

    for col in out.columns:
        if out[col].dtype == "object":
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return out


def main():
    config = load_config()

    target = config["preprocess"]["target_column"]

    train_clean_path = config["staged_pipeline"]["train_clean_path"]
    test_clean_path = config["staged_pipeline"]["test_clean_path"]

    train_selected_path = config["staged_pipeline"]["train_selected_path"]
    test_selected_path = config["staged_pipeline"]["test_selected_path"]
    selected_features_path = config["staged_pipeline"]["selected_features_path"]

    k = config.get("feature_selection", {}).get("k", 20)

    train_df = pd.read_csv(train_clean_path)
    test_df = pd.read_csv(test_clean_path)

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    feature_cols = X_train.columns.tolist()

    if k == "all":
        selected_features = feature_cols
        scores_payload = []
    else:
        k = int(k)
        k = min(k, len(feature_cols))

        X_encoded = encode_for_feature_selection(X_train)

        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_encoded, y_train)

        mask = selector.get_support()
        selected_features = [col for col, keep in zip(feature_cols, mask) if keep]

        scores_payload = [
            {
                "feature": col,
                "score": None if pd.isna(score) else float(score),
            }
            for col, score in zip(feature_cols, selector.scores_)
        ]
        scores_payload = sorted(
            scores_payload,
            key=lambda x: -1 if x["score"] is None else x["score"],
            reverse=True,
        )

    train_selected = train_df[selected_features + [target]].copy()
    test_selected = test_df[selected_features + [target]].copy()

    ensure_parent(train_selected_path)
    ensure_parent(test_selected_path)

    train_selected.to_csv(train_selected_path, index=False)
    test_selected.to_csv(test_selected_path, index=False)

    payload = {
        "method": "SelectKBest(f_classif) on label-encoded categorical features",
        "k": k,
        "selected_features": selected_features,
        "feature_scores": scores_payload,
    }
    write_json(payload, selected_features_path)

    print("✅ Stage feature_selection done")
    print(f"Selected features: {selected_features}")
    print(f"Train selected: {train_selected_path}")
    print(f"Test selected: {test_selected_path}")


if __name__ == "__main__":
    main()
