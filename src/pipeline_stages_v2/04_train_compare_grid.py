import json
import logging
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


LOGGER = logging.getLogger(__name__)


def _parse_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _set_nested(config: dict, path: tuple[str, ...], value: Any) -> None:
    cur = config
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def apply_env_overrides(config: dict) -> dict:
    """
    Cho phép override params_staged_pipeline.yaml bằng biến môi trường trong .env.
    Nếu biến không tồn tại thì vẫn dùng YAML như bình thường.
    """
    overrides: list[tuple[str, tuple[str, ...], Any]] = [
        ("THREAD_POOL_MAX_WORKERS", ("thread_pool", "max_workers"), int),
        ("GRID_SEARCH_METHOD", ("grid_search", "method"), str),
        ("GRID_SEARCH_SCORING", ("grid_search", "scoring"), str),
        ("GRID_SEARCH_CV_FOLDS", ("grid_search", "cv_folds"), int),
        ("GRID_SEARCH_INNER_N_JOBS", ("grid_search", "inner_n_jobs"), int),
        ("GRID_SEARCH_VERBOSE", ("grid_search", "verbose"), int),
        ("GRID_SEARCH_MAX_EVALUATED_CANDIDATES", ("grid_search", "max_evaluated_candidates"), int),
        ("GRID_SEARCH_MIN_EVALUATED_CANDIDATES", ("grid_search", "min_evaluated_candidates"), int),
        ("GRID_SEARCH_EARLY_STOP_PATIENCE", ("grid_search", "early_stop_patience"), int),
        ("GRID_SEARCH_EARLY_STOP_MIN_DELTA", ("grid_search", "early_stop_min_delta"), float),
        ("GRID_SEARCH_RANDOM_STATE", ("grid_search", "random_state"), int),
        ("MODEL_SELECTION_THRESHOLD", ("model_selection", "threshold"), float),
        ("MODEL_SELECTION_METRIC", ("model_selection", "metric"), str),
        ("MLFLOW_EXPERIMENT_NAME", ("mlflow", "experiment_name"), str),
        ("MLFLOW_REGISTRY_NAME", ("mlflow", "registry_name"), str),
    ]

    for env_name, path, caster in overrides:
        raw = os.getenv(env_name)
        if raw is not None and raw != "":
            _set_nested(config, path, caster(raw))

    bool_overrides = [
        ("GRID_SEARCH_TRY_SMOTE", ("grid_search", "try_smote")),
        ("GRID_SEARCH_PARAM_SHUFFLE", ("grid_search", "param_shuffle")),
    ]
    for env_name, path in bool_overrides:
        parsed = _parse_bool(os.getenv(env_name))
        if parsed is not None:
            _set_nested(config, path, parsed)

    threshold_values = os.getenv("THRESHOLD_VALUES")
    if threshold_values:
        values = [float(x.strip()) for x in threshold_values.split(",") if x.strip()]
        if values:
            _set_nested(config, ("threshold", "values"), values)

    return config


def load_config() -> dict:
    params_path = os.getenv("PARAMS_PATH", "params_staged_pipeline.yaml")
    with open(params_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return apply_env_overrides(config)


def configure_logging(config: dict) -> None:
    level_name = str(config.get("logging", {}).get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
    )

    # Giảm log thừa từ thư viện để Airflow không bị phình log quá nhanh.
    for noisy_logger in [
        "mlflow",
        "urllib3",
        "botocore",
        "boto3",
        "s3transfer",
        "sklearn",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Giảm các warning không ảnh hưởng logic nhưng làm phình Airflow logs.
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
    warnings.filterwarnings("ignore", message="Distutils was imported before Setuptools.*")
    warnings.filterwarnings("ignore", message="Setuptools is replacing distutils.*")
    warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column.*")


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

    LOGGER.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    LOGGER.info("MLflow S3 endpoint: %s", os.environ["MLFLOW_S3_ENDPOINT_URL"])


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
    Mỗi model được thiết kế có tối thiểu khoảng 20 tổ hợp tham số.

    Lý do: nếu tổng số candidate của grid < 20 thì dù config
    max_evaluated_candidates=20, GridSearch cũng chỉ chạy được số candidate
    thực tế. Vì vậy grid bên dưới được mở rộng để mỗi model có >= 20 candidate
    ngay cả ở scenario before_preprocessing, nơi sampler chỉ là passthrough.
    """
    return {
        "Logistic_Regression": (
            LogisticRegression(max_iter=1000, solver="liblinear"),
            {
                # 10 * 2 = 20 candidates ở before_preprocessing.
                # Với after_preprocessing có thêm sampler nên tổng là 40, sau đó lấy tối đa 20.
                "model__C": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random_Forest": (
            RandomForestClassifier(random_state=42),
            {
                # 4 * 5 * 2 = 40 candidates ở before_preprocessing.
                "model__n_estimators": [50, 100, 200, 300],
                "model__max_depth": [None, 5, 10, 20, 30],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Naive_Bayes": (
            GaussianNB(),
            {
                # Đúng 20 candidates ở before_preprocessing.
                "model__var_smoothing": np.logspace(-12, -6, 20).tolist(),
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                # 8 * 2 * 2 = 32 candidates ở before_preprocessing.
                "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        "Decision_Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                # 5 * 4 * 3 * 2 = 120 candidates ở before_preprocessing.
                "model__max_depth": [None, 5, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 5],
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



def search_with_exhaustive_grid(
    *,
    pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str,
    cv_folds: int,
    grid_n_jobs: int,
    grid_verbose: int,
):
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_folds,
        n_jobs=grid_n_jobs,
        verbose=grid_verbose,
        error_score="raise",
        return_train_score=False,
    )
    grid.fit(X_train, y_train)

    return {
        "best_estimator": grid.best_estimator_,
        "best_score": float(grid.best_score_),
        "best_params": grid.best_params_,
        "total_candidates": len(list(ParameterGrid(param_grid))),
        "planned_candidates": len(list(ParameterGrid(param_grid))),
        "evaluated_candidates": len(list(ParameterGrid(param_grid))),
        "stopped_early": False,
        "stop_reason": "exhaustive_grid_completed",
        "search_history": [
            {
                "candidate_index": int(i + 1),
                "mean_cv_score": float(score),
                "params": to_jsonable_params(params),
            }
            for i, (score, params) in enumerate(
                zip(grid.cv_results_["mean_test_score"], grid.cv_results_["params"])
            )
        ],
    }


def search_with_early_stop_grid(
    *,
    pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str,
    cv_folds: int,
    grid_n_jobs: int,
    config: dict,
):
    """
    Custom grid search có early stopping theo số candidate không cải thiện liên tiếp.

    Ví dụ patience=10:
    - Nếu đã có best_score.
    - Sau đó thử 10 bộ tham số liên tiếp mà score không tăng quá min_delta.
    - Dừng search, fit lại best_estimator trên toàn bộ train data.

    Lưu ý: đây là best trong các candidate đã thử, không đảm bảo là global best
    nếu grid chưa chạy hết.
    """
    grid_cfg = config.get("grid_search", {})
    patience = int(grid_cfg.get("early_stop_patience", 10))
    min_delta = float(grid_cfg.get("early_stop_min_delta", 0.0))
    shuffle = bool(grid_cfg.get("param_shuffle", True))
    random_state = int(grid_cfg.get("random_state", 42))

    # Giới hạn số bộ tham số được thử cho mỗi model.
    # Ví dụ max_evaluated_candidates=20 nghĩa là mỗi model chỉ thử tối đa khoảng 20 candidate.
    # Nếu tổng số candidate nhỏ hơn 20 thì chạy hết.
    max_evaluated_candidates_raw = grid_cfg.get("max_evaluated_candidates", 20)
    max_evaluated_candidates = (
        None
        if max_evaluated_candidates_raw in [None, "", "all"]
        else int(max_evaluated_candidates_raw)
    )

    # Nếu muốn mỗi model phải chạy khoảng 20 bộ tham số, đặt min_evaluated_candidates=20.
    # Khi đó early stopping chỉ được kích hoạt sau khi đã thử ít nhất 20 candidate
    # hoặc sau khi đã thử hết toàn bộ candidate nếu grid nhỏ hơn 20.
    min_evaluated_candidates = int(grid_cfg.get("min_evaluated_candidates", 20))

    candidates = list(ParameterGrid(param_grid))
    total_candidates = len(candidates)

    if shuffle and total_candidates > 1:
        rng = np.random.default_rng(random_state)
        order = rng.permutation(total_candidates).tolist()
        candidates = [candidates[i] for i in order]

    if max_evaluated_candidates is not None and max_evaluated_candidates > 0:
        candidates = candidates[: min(max_evaluated_candidates, total_candidates)]

    planned_candidates = len(candidates)

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    best_score = -np.inf
    best_params = None
    no_improve_count = 0
    evaluated_candidates = 0
    history = []
    stopped_early = False
    stop_reason = "all_candidates_completed"

    for idx, params in enumerate(candidates, start=1):
        candidate = clone(pipeline)
        candidate.set_params(**params)

        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=grid_n_jobs,
            error_score="raise",
        )

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        evaluated_candidates += 1

        improved = mean_score > best_score + min_delta
        if improved:
            best_score = mean_score
            best_params = params
            no_improve_count = 0
        else:
            no_improve_count += 1

        history.append(
            {
                "candidate_index": idx,
                "mean_cv_score": mean_score,
                "std_cv_score": std_score,
                "is_new_best": bool(improved),
                "no_improve_count": int(no_improve_count),
                "params": to_jsonable_params(params),
            }
        )

        LOGGER.info(
            "candidate %s/%s | score=%.5f | best=%.5f | no_improve=%s/%s",
            idx,
            planned_candidates,
            mean_score,
            best_score,
            no_improve_count,
            patience,
        )

        can_stop_by_patience = evaluated_candidates >= min(min_evaluated_candidates, planned_candidates)

        if patience > 0 and no_improve_count >= patience and can_stop_by_patience:
            stopped_early = True
            stop_reason = f"no_improvement_for_{patience}_candidates_after_min_{min_evaluated_candidates}"
            break

    if best_params is None:
        raise RuntimeError("Early-stop grid search did not evaluate any valid candidate.")

    best_estimator = clone(pipeline)
    best_estimator.set_params(**best_params)
    best_estimator.fit(X_train, y_train)

    if (
        not stopped_early
        and max_evaluated_candidates is not None
        and max_evaluated_candidates > 0
        and total_candidates > planned_candidates
    ):
        stopped_early = True
        stop_reason = f"max_evaluated_candidates_{planned_candidates}_reached"

    return {
        "best_estimator": best_estimator,
        "best_score": float(best_score),
        "best_params": best_params,
        "total_candidates": int(total_candidates),
        "planned_candidates": int(planned_candidates),
        "min_evaluated_candidates": int(min_evaluated_candidates),
        "evaluated_candidates": int(evaluated_candidates),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "search_history": history,
    }


def run_hyperparameter_search(
    *,
    pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str,
    cv_folds: int,
    grid_n_jobs: int,
    grid_verbose: int,
    config: dict,
):
    method = str(config.get("grid_search", {}).get("method", "early_stop")).lower()

    if method in {"grid", "gridsearch", "grid_search", "exhaustive"}:
        return search_with_exhaustive_grid(
            pipeline=pipeline,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            scoring=scoring,
            cv_folds=cv_folds,
            grid_n_jobs=grid_n_jobs,
            grid_verbose=grid_verbose,
        )

    if method in {"early_stop", "early_stopping", "patience"}:
        return search_with_early_stop_grid(
            pipeline=pipeline,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            scoring=scoring,
            cv_folds=cv_folds,
            grid_n_jobs=grid_n_jobs,
            config=config,
        )

    raise ValueError(f"Unknown grid_search.method: {method}")


def train_one_candidate(
    *,
    config: dict,
    scenario: str,
    model_name: str,
    model: Any,
    model_grid: dict,
    thresholds: list[float],
    selection_threshold: float,
    selection_metric: str,
    experiment_name: str,
    cv_folds: int,
    scoring: str,
    grid_n_jobs: int,
    grid_verbose: int,
) -> tuple[dict, list[dict]]:
    started_at = time.perf_counter()
    run_name = f"{scenario}_{model_name}"

    X_train, X_test, y_train, y_test = load_dataset_pair(config, scenario)
    numeric_cols, categorical_cols = detect_columns(X_train)
    preprocessor = build_preprocessor(scenario, numeric_cols, categorical_cols)
    pipeline = build_pipeline(model=model, preprocessor=preprocessor)
    param_grid = build_grid(scenario=scenario, model_grid=model_grid, config=config)

    LOGGER.info("Start %s | grid_n_jobs=%s", run_name, grid_n_jobs)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        search_result = run_hyperparameter_search(
            pipeline=pipeline,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            scoring=scoring,
            cv_folds=cv_folds,
            grid_n_jobs=grid_n_jobs,
            grid_verbose=grid_verbose,
            config=config,
        )

        best_estimator = search_result["best_estimator"]
        best_score = float(search_result["best_score"])
        best_params = search_result["best_params"]

        flat_metrics, threshold_rows = evaluate_with_thresholds(
            model=best_estimator,
            X_test=X_test,
            y_test=y_test,
            thresholds=thresholds,
        )

        best_params_json = json.dumps(
            to_jsonable_params(best_params),
            ensure_ascii=False,
        )

        search_history_path = Path(config["staged_pipeline"]["artifact_dir"]) / "grid_search_history" / f"{run_name}.json"
        search_history_path.parent.mkdir(parents=True, exist_ok=True)
        search_history_path.write_text(
            json.dumps(search_result["search_history"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        params_to_log = {
            "scenario": scenario,
            "model_name": model_name,
            "grid_search_method": str(config.get("grid_search", {}).get("method", "early_stop")),
            "grid_scoring": scoring,
            "cv_folds": cv_folds,
            "grid_n_jobs": grid_n_jobs,
            "grid_total_candidates": search_result["total_candidates"],
            "grid_planned_candidates": search_result["planned_candidates"],
            "grid_min_evaluated_candidates": search_result.get("min_evaluated_candidates", search_result["planned_candidates"]),
            "grid_evaluated_candidates": search_result["evaluated_candidates"],
            "grid_stopped_early": search_result["stopped_early"],
            "grid_stop_reason": search_result["stop_reason"],
            "thresholds_compared": json.dumps(thresholds),
            "selection_threshold": selection_threshold,
            "selection_metric": selection_metric,
            "best_params": best_params_json,
        }

        mlflow.log_params(params_to_log)
        mlflow.log_metrics(
            {
                "cv_best_score": best_score,
                "grid_evaluated_candidates": float(search_result["evaluated_candidates"]),
                "grid_total_candidates": float(search_result["total_candidates"]),
                **flat_metrics,
            }
        )
        mlflow.log_artifact(str(search_history_path), artifact_path="grid_search_history")

        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            artifact_path="model_file",
        )

        wide_row = {
            "scenario": scenario,
            "model_name": model_name,
            "run_id": run.info.run_id,
            "mlflow_model_uri": f"runs:/{run.info.run_id}/model_file",
            "cv_best_score": best_score,
            "best_params": best_params_json,
            "grid_search_method": str(config.get("grid_search", {}).get("method", "early_stop")),
            "grid_total_candidates": search_result["total_candidates"],
            "grid_planned_candidates": search_result["planned_candidates"],
            "grid_min_evaluated_candidates": search_result.get("min_evaluated_candidates", search_result["planned_candidates"]),
            "grid_evaluated_candidates": search_result["evaluated_candidates"],
            "grid_stopped_early": search_result["stopped_early"],
            "grid_stop_reason": search_result["stop_reason"],
            **flat_metrics,
        }

        long_rows = []
        for tr in threshold_rows:
            long_rows.append(
                {
                    "scenario": scenario,
                    "model_name": model_name,
                    "run_id": run.info.run_id,
                    "mlflow_model_uri": f"runs:/{run.info.run_id}/model_file",
                    "cv_best_score": best_score,
                    "best_params": best_params_json,
                    "grid_search_method": str(config.get("grid_search", {}).get("method", "early_stop")),
                    "grid_total_candidates": search_result["total_candidates"],
                    "grid_evaluated_candidates": search_result["evaluated_candidates"],
                    "grid_stopped_early": search_result["stopped_early"],
                    "grid_stop_reason": search_result["stop_reason"],
                    **tr,
                }
            )

    elapsed = time.perf_counter() - started_at
    best_threshold_row = max(
        threshold_rows,
        key=lambda row: row["test_f1_churned_threshold"],
    )
    LOGGER.info(
        "Done %s | cv=%.4f | evaluated=%s/%s | stopped=%s | best_threshold=%.2f | f1=%.4f | %.1fs",
        run_name,
        best_score,
        search_result["evaluated_candidates"],
        search_result["total_candidates"],
        search_result["stopped_early"],
        best_threshold_row["threshold"],
        best_threshold_row["test_f1_churned_threshold"],
        elapsed,
    )

    return wide_row, long_rows


def resolve_parallel_settings(config: dict, total_jobs: int) -> tuple[int, int, int]:
    grid_cfg = config.get("grid_search", {})
    thread_cfg = config.get("thread_pool", {})

    max_workers = int(os.getenv("TRAIN_MAX_WORKERS", thread_cfg.get("max_workers", 1)))
    max_workers = max(1, min(max_workers, total_jobs))

    if "inner_n_jobs" in grid_cfg:
        grid_n_jobs = int(grid_cfg["inner_n_jobs"])
    elif max_workers > 1:
        grid_n_jobs = 1
    else:
        grid_n_jobs = int(grid_cfg.get("n_jobs", -1))

    grid_verbose = int(grid_cfg.get("verbose", 0))
    return max_workers, grid_n_jobs, grid_verbose


def main():
    config = load_config()
    configure_logging(config)

    artifact_dir = Path(config["staged_pipeline"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    grid_history_dir = artifact_dir / "grid_search_history"
    grid_history_dir.mkdir(parents=True, exist_ok=True)
    (grid_history_dir / ".gitkeep").touch()

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

    scenarios = config.get("comparison", {}).get(
        "scenarios",
        [
            "before_preprocessing",
            "after_preprocessing",
            "after_preprocessing_feature_selection",
        ],
    )

    models_and_grids = get_models_and_grids()
    jobs = []
    for scenario in scenarios:
        for model_name, (model, model_grid) in models_and_grids.items():
            jobs.append((scenario, model_name, model, model_grid))

    max_workers, grid_n_jobs, grid_verbose = resolve_parallel_settings(config, len(jobs))

    LOGGER.info("Total training jobs: %s", len(jobs))
    LOGGER.info("ThreadPoolExecutor max_workers: %s", max_workers)
    LOGGER.info("Inner CV n_jobs: %s", grid_n_jobs)
    LOGGER.info("Grid search method: %s", config.get("grid_search", {}).get("method", "early_stop"))
    LOGGER.info("Early stop patience: %s", config.get("grid_search", {}).get("early_stop_patience", 10))
    LOGGER.info("Max evaluated candidates per model: %s", config.get("grid_search", {}).get("max_evaluated_candidates", 20))
    LOGGER.info("Min evaluated candidates before early stop: %s", config.get("grid_search", {}).get("min_evaluated_candidates", 20))
    LOGGER.info("Thresholds: %s", thresholds)

    wide_results = []
    long_results = []

    if max_workers == 1:
        for scenario, model_name, model, model_grid in jobs:
            wide_row, long_rows = train_one_candidate(
                config=config,
                scenario=scenario,
                model_name=model_name,
                model=model,
                model_grid=model_grid,
                thresholds=thresholds,
                selection_threshold=selection_threshold,
                selection_metric=selection_metric,
                experiment_name=experiment_name,
                cv_folds=cv_folds,
                scoring=scoring,
                grid_n_jobs=grid_n_jobs,
                grid_verbose=grid_verbose,
            )
            wide_results.append(wide_row)
            long_results.extend(long_rows)
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="train-grid") as executor:
            future_to_name = {
                executor.submit(
                    train_one_candidate,
                    config=config,
                    scenario=scenario,
                    model_name=model_name,
                    model=model,
                    model_grid=model_grid,
                    thresholds=thresholds,
                    selection_threshold=selection_threshold,
                    selection_metric=selection_metric,
                    experiment_name=experiment_name,
                    cv_folds=cv_folds,
                    scoring=scoring,
                    grid_n_jobs=grid_n_jobs,
                    grid_verbose=grid_verbose,
                ): f"{scenario}_{model_name}"
                for scenario, model_name, model, model_grid in jobs
            }

            for future in as_completed(future_to_name):
                job_name = future_to_name[future]
                try:
                    wide_row, long_rows = future.result()
                except Exception:
                    LOGGER.exception("Training job failed: %s", job_name)
                    raise

                wide_results.append(wide_row)
                long_results.extend(long_rows)

    if not wide_results or not long_results:
        raise RuntimeError("No model results found.")

    scenario_order = {scenario: idx for idx, scenario in enumerate(scenarios)}
    model_order = {model_name: idx for idx, model_name in enumerate(models_and_grids.keys())}

    wide_results = sorted(
        wide_results,
        key=lambda row: (scenario_order.get(row["scenario"], 999), model_order.get(row["model_name"], 999)),
    )
    long_results = sorted(
        long_results,
        key=lambda row: (
            scenario_order.get(row["scenario"], 999),
            model_order.get(row["model_name"], 999),
            row["threshold"],
        ),
    )

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
        if abs(float(row["threshold"]) - selection_threshold) < 1e-12
    ]

    if not selection_candidates:
        LOGGER.warning(
            "selection_threshold=%s không nằm trong thresholds=%s. Sẽ chọn best trên toàn bộ threshold.",
            selection_threshold,
            thresholds,
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
        "thread_pool_max_workers": max_workers,
        "grid_search_inner_n_jobs": grid_n_jobs,
        "grid_search_method": best.get("grid_search_method"),
        "grid_total_candidates": best.get("grid_total_candidates"),
        "grid_evaluated_candidates": best.get("grid_evaluated_candidates"),
        "grid_stopped_early": best.get("grid_stopped_early"),
        "grid_stop_reason": best.get("grid_stop_reason"),
    }

    summary_path = config["staged_pipeline"]["best_summary_path"]
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(
        json.dumps(best_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    LOGGER.info("BEST MODEL")
    LOGGER.info(json.dumps(best_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
