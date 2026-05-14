import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def load_config() -> dict:
    params_path = os.getenv("PARAMS_PATH", "params_staged_pipeline.yaml")
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_running_inside_container() -> bool:
    cwd = str(Path.cwd())
    return (
        os.getenv("AIRFLOW_HOME") is not None
        or cwd.startswith("/opt/airflow")
        or Path("/opt/airflow").exists()
    )


def normalize_url_for_container(url: str | None, service_name: str, port: int) -> str:
    if not url:
        return f"http://localhost:{port}"

    if is_running_inside_container():
        url = url.replace(f"http://localhost:{port}", f"http://{service_name}:{port}")
        url = url.replace(f"http://127.0.0.1:{port}", f"http://{service_name}:{port}")

    return url


def setup_mlflow(config: dict) -> None:
    mlflow_config = config.get("mlflow", {})

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        mlflow_config.get("tracking_uri", "http://localhost:5000"),
    )
    s3_endpoint = os.getenv(
        "MLFLOW_S3_ENDPOINT_URL",
        mlflow_config.get("s3_endpoint", "http://localhost:9000"),
    )

    tracking_uri = normalize_url_for_container(tracking_uri, "mlflow", 5000)
    s3_endpoint = normalize_url_for_container(s3_endpoint, "minio", 9000)

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    mlflow.set_tracking_uri(tracking_uri)

    print(f"🔗 MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"🪣 MLflow S3 endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def detect_columns(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(scenario: str, numeric_cols: list[str], categorical_cols: list[str]):
    if scenario == "before_preprocessing":
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
    else:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def get_models_and_grids() -> dict[str, tuple[Any, dict]]:
    """
    Sử dụng 5 model:
    - Logistic_Regression
    - Random_Forest
    - Naive_Bayes
    - KNN
    - Decision_Tree

    Đã bỏ SVM.
    """
    return {
        "Logistic_Regression": (
            LogisticRegression(max_iter=1000, solver="liblinear"),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),

        "Random_Forest": (
            RandomForestClassifier(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__class_weight": [None, "balanced"],
            },
        ),

        "Naive_Bayes": (
            GaussianNB(),
            {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        ),

        "KNN": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 11],
                "model__weights": ["uniform", "distance"],
            },
        ),

        "Decision_Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 10],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }

def build_pipeline(model: Any, preprocessor):
    return ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", "passthrough"),
            ("model", model),
        ]
    )


def build_grid(scenario: str, model_grid: dict, config: dict) -> dict:
    grid = dict(model_grid)

    if scenario in ["after_preprocessing", "after_preprocessing_feature_selection"]:
        if config.get("grid_search", {}).get("try_smote", True):
            grid["sampler"] = ["passthrough", SMOTE(random_state=42)]
        else:
            grid["sampler"] = ["passthrough"]
    else:
        grid["sampler"] = ["passthrough"]

    return grid


def threshold_key(threshold: float) -> str:
    return str(threshold).replace(".", "_")


def evaluate_with_thresholds(model, X_test, y_test, thresholds: list[float]) -> tuple[dict, list[dict]]:
    y_pred_default = model.predict(X_test)

    flat_metrics = {
        "test_accuracy_default_predict": float(accuracy_score(y_test, y_pred_default)),
        "test_precision_churned_default_predict": float(
            precision_score(y_test, y_pred_default, zero_division=0)
        ),
        "test_recall_churned_default_predict": float(
            recall_score(y_test, y_pred_default, zero_division=0)
        ),
        "test_f1_churned_default_predict": float(
            f1_score(y_test, y_pred_default, zero_division=0)
        ),
    }

    long_rows = []

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    for threshold in thresholds:
        if y_prob is not None:
            y_pred = (y_prob >= threshold).astype(int)
        else:
            y_pred = y_pred_default

        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        accuracy = float(accuracy_score(y_test, y_pred))

        tk = threshold_key(threshold)
        flat_metrics.update(
            {
                f"test_accuracy_threshold_{tk}": accuracy,
                f"test_precision_churned_threshold_{tk}": precision,
                f"test_recall_churned_threshold_{tk}": recall,
                f"test_f1_churned_threshold_{tk}": f1,
            }
        )

        long_rows.append(
            {
                "threshold": float(threshold),
                "test_accuracy_threshold": accuracy,
                "test_precision_churned_threshold": precision,
                "test_recall_churned_threshold": recall,
                "test_f1_churned_threshold": f1,
            }
        )

    return flat_metrics, long_rows


def to_jsonable_params(params: dict) -> dict:
    return {k: str(v) for k, v in params.items()}


def load_dataset_pair(config: dict, scenario: str):
    target = config["preprocess"]["target_column"]

    if scenario == "before_preprocessing":
        train_path = config["staged_pipeline"]["train_raw_path"]
        test_path = config["staged_pipeline"]["test_raw_path"]
    elif scenario == "after_preprocessing":
        train_path = config["staged_pipeline"]["train_clean_path"]
        test_path = config["staged_pipeline"]["test_clean_path"]
    elif scenario == "after_preprocessing_feature_selection":
        train_path = config["staged_pipeline"]["train_selected_path"]
        test_path = config["staged_pipeline"]["test_selected_path"]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test


def main():
    config = load_config()
    setup_mlflow(config)

    thresholds = config.get("threshold", {}).get("values", [0.5, 0.35])
    thresholds = [float(t) for t in thresholds]

    selection_threshold = float(config.get("model_selection", {}).get("threshold", 0.35))
    selection_metric = config.get("model_selection", {}).get(
        "metric",
        "test_f1_churned_threshold",
    )

    experiment_name = config["mlflow"]["experiment_name"]

    cv_folds = int(config.get("grid_search", {}).get("cv_folds", 5))
    scoring = config.get("grid_search", {}).get("scoring", "f1")
    n_jobs = int(config.get("grid_search", {}).get("n_jobs", -1))

    mlflow.set_experiment(experiment_name)

    scenarios = config.get("comparison", {}).get(
        "scenarios",
        [
            "before_preprocessing",
            "after_preprocessing",
            "after_preprocessing_feature_selection",
        ],
    )

    models_and_grids = get_models_and_grids()

    wide_results = []
    long_results = []

    for scenario in scenarios:
        X_train, X_test, y_train, y_test = load_dataset_pair(config, scenario)
        numeric_cols, categorical_cols = detect_columns(X_train)

        print("\n==============================")
        print(f"Scenario: {scenario}")
        print(f"Thresholds: {thresholds}")
        print("==============================")

        preprocessor = build_preprocessor(scenario, numeric_cols, categorical_cols)

        for model_name, (model, model_grid) in models_and_grids.items():
            run_name = f"{scenario}_{model_name}"

            pipeline = build_pipeline(model=model, preprocessor=preprocessor)

            param_grid = build_grid(
                scenario=scenario,
                model_grid=model_grid,
                config=config,
            )

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv_folds,
                n_jobs=n_jobs,
                verbose=1,
                error_score="raise",
            )

            with mlflow.start_run(run_name=run_name) as run:
                grid.fit(X_train, y_train)

                best_estimator = grid.best_estimator_
                flat_metrics, threshold_rows = evaluate_with_thresholds(
                    model=best_estimator,
                    X_test=X_test,
                    y_test=y_test,
                    thresholds=thresholds,
                )

                params_to_log = {
                    "scenario": scenario,
                    "model_name": model_name,
                    "grid_scoring": scoring,
                    "cv_folds": cv_folds,
                    "thresholds_compared": json.dumps(thresholds),
                    "selection_threshold": selection_threshold,
                    "selection_metric": selection_metric,
                    "best_params": json.dumps(
                        to_jsonable_params(grid.best_params_),
                        ensure_ascii=False,
                    ),
                }

                mlflow.log_params(params_to_log)
                mlflow.log_metrics(
                    {
                        "cv_best_score": float(grid.best_score_),
                        **flat_metrics,
                    }
                )

                mlflow.sklearn.log_model(
                    sk_model=best_estimator,
                    artifact_path="model_file",
                    input_example=X_test.head(5),
                )

                wide_row = {
                    "scenario": scenario,
                    "model_name": model_name,
                    "run_id": run.info.run_id,
                    "mlflow_model_uri": f"runs:/{run.info.run_id}/model_file",
                    "cv_best_score": float(grid.best_score_),
                    "best_params": json.dumps(
                        to_jsonable_params(grid.best_params_),
                        ensure_ascii=False,
                    ),
                    **flat_metrics,
                }
                wide_results.append(wide_row)

                for tr in threshold_rows:
                    long_results.append(
                        {
                            "scenario": scenario,
                            "model_name": model_name,
                            "run_id": run.info.run_id,
                            "mlflow_model_uri": f"runs:/{run.info.run_id}/model_file",
                            "cv_best_score": float(grid.best_score_),
                            "best_params": wide_row["best_params"],
                            **tr,
                        }
                    )

                print(f"✅ Done: {run_name}")
                print(f"CV best score: {grid.best_score_:.4f}")

                for tr in threshold_rows:
                    print(
                        f"  threshold={tr['threshold']} | "
                        f"F1={tr['test_f1_churned_threshold']:.4f} | "
                        f"Precision={tr['test_precision_churned_threshold']:.4f} | "
                        f"Recall={tr['test_recall_churned_threshold']:.4f}"
                    )

    if not wide_results or not long_results:
        raise RuntimeError("No model results found.")

    artifact_dir = Path(config["staged_pipeline"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    wide_result_path = config["staged_pipeline"]["compare_results_path"]
    long_result_path = config["staged_pipeline"]["threshold_compare_results_path"]

    Path(wide_result_path).parent.mkdir(parents=True, exist_ok=True)
    Path(long_result_path).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(wide_results).to_csv(wide_result_path, index=False)
    pd.DataFrame(long_results).to_csv(long_result_path, index=False)

    selection_candidates = [
        row for row in long_results
        if float(row["threshold"]) == selection_threshold
    ]

    if not selection_candidates:
        print(
            f"⚠️ selection_threshold={selection_threshold} không nằm trong thresholds={thresholds}. "
            "Sẽ chọn best trên toàn bộ threshold."
        )
        selection_candidates = long_results

    best = sorted(
        selection_candidates,
        key=lambda x: x[selection_metric],
        reverse=True,
    )[0]

    best_summary = {
        "best_scenario": best["scenario"],
        "best_model_name": best["model_name"],
        "best_run_id": best["run_id"],
        "best_model_uri": best["mlflow_model_uri"],
        "selected_threshold": float(best["threshold"]),
        "selection_metric": selection_metric,
        "cv_best_score": best["cv_best_score"],
        "test_f1_churned_threshold": best["test_f1_churned_threshold"],
        "test_precision_churned_threshold": best["test_precision_churned_threshold"],
        "test_recall_churned_threshold": best["test_recall_churned_threshold"],
        "test_accuracy_threshold": best["test_accuracy_threshold"],
        "best_params": best["best_params"],
        "thresholds_compared": thresholds,
    }

    summary_path = config["staged_pipeline"]["best_summary_path"]
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(
        json.dumps(best_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n==============================")
    print("BEST MODEL")
    print("==============================")
    print(json.dumps(best_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
