import argparse
import os
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml


def load_config(params_path: str) -> dict:
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

    tracking_uri = normalize_url_for_container(
        tracking_uri,
        service_name="mlflow",
        port=5000,
    )

    s3_endpoint = normalize_url_for_container(
        s3_endpoint,
        service_name="minio",
        port=9000,
    )

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow S3 Endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")


def get_test_path_by_scenario(config: dict, scenario: str) -> str:
    if scenario == "before_preprocessing":
        return config["staged_pipeline"]["test_raw_path"]

    if scenario == "after_preprocessing":
        return config["staged_pipeline"]["test_clean_path"]

    if scenario == "after_preprocessing_feature_selection":
        return config["staged_pipeline"]["test_selected_path"]

    raise ValueError(f"Unknown scenario: {scenario}")


def percentile(values: list[float], q: int) -> float:
    if not values:
        return 0.0

    return float(np.percentile(values, q))


def measure_single_predict(model, X_sample: pd.DataFrame) -> dict:
    latencies_ms = []

    for i in range(len(X_sample)):
        row = X_sample.iloc[[i]]

        start = time.perf_counter()
        _ = model.predict(row)
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000)

    return {
        "predict_single_mean_ms": float(np.mean(latencies_ms)),
        "predict_single_p50_ms": percentile(latencies_ms, 50),
        "predict_single_p95_ms": percentile(latencies_ms, 95),
        "predict_single_p99_ms": percentile(latencies_ms, 99),
        "predict_single_min_ms": float(np.min(latencies_ms)),
        "predict_single_max_ms": float(np.max(latencies_ms)),
    }


def measure_single_predict_proba(model, X_sample: pd.DataFrame) -> dict:
    if not hasattr(model, "predict_proba"):
        return {
            "predict_proba_single_mean_ms": None,
            "predict_proba_single_p50_ms": None,
            "predict_proba_single_p95_ms": None,
            "predict_proba_single_p99_ms": None,
            "predict_proba_single_min_ms": None,
            "predict_proba_single_max_ms": None,
        }

    latencies_ms = []

    for i in range(len(X_sample)):
        row = X_sample.iloc[[i]]

        start = time.perf_counter()
        _ = model.predict_proba(row)
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000)

    return {
        "predict_proba_single_mean_ms": float(np.mean(latencies_ms)),
        "predict_proba_single_p50_ms": percentile(latencies_ms, 50),
        "predict_proba_single_p95_ms": percentile(latencies_ms, 95),
        "predict_proba_single_p99_ms": percentile(latencies_ms, 99),
        "predict_proba_single_min_ms": float(np.min(latencies_ms)),
        "predict_proba_single_max_ms": float(np.max(latencies_ms)),
    }


def measure_threshold_predict(
    model,
    X_sample: pd.DataFrame,
    thresholds: list[float],
) -> list[dict]:
    """
    Đo latency kiểu predict thực tế:

    predict_proba -> lấy xác suất class 1 -> so sánh với threshold.
    """

    if not hasattr(model, "predict_proba"):
        return []

    results = []

    for threshold in thresholds:
        latencies_ms = []

        for i in range(len(X_sample)):
            row = X_sample.iloc[[i]]

            start = time.perf_counter()
            probability = model.predict_proba(row)
            churn_probability = float(probability[0][1])
            _ = 1 if churn_probability >= threshold else 0
            end = time.perf_counter()

            latencies_ms.append((end - start) * 1000)

        results.append(
            {
                "threshold": float(threshold),
                "threshold_predict_single_mean_ms": float(np.mean(latencies_ms)),
                "threshold_predict_single_p50_ms": percentile(latencies_ms, 50),
                "threshold_predict_single_p95_ms": percentile(latencies_ms, 95),
                "threshold_predict_single_p99_ms": percentile(latencies_ms, 99),
                "threshold_predict_single_min_ms": float(np.min(latencies_ms)),
                "threshold_predict_single_max_ms": float(np.max(latencies_ms)),
            }
        )

    return results


def measure_batch_predict(
    model,
    X_sample: pd.DataFrame,
    repeats: int,
) -> dict:
    latencies_ms = []

    for _ in range(repeats):
        start = time.perf_counter()
        _ = model.predict(X_sample)
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000)

    mean_ms = float(np.mean(latencies_ms))
    p50_ms = percentile(latencies_ms, 50)
    p95_ms = percentile(latencies_ms, 95)

    return {
        "predict_batch_rows": int(len(X_sample)),
        "predict_batch_repeats": int(repeats),
        "predict_batch_mean_ms": mean_ms,
        "predict_batch_p50_ms": p50_ms,
        "predict_batch_p95_ms": p95_ms,
        "predict_batch_per_row_mean_ms": mean_ms / max(len(X_sample), 1),
        "predict_batch_per_row_p50_ms": p50_ms / max(len(X_sample), 1),
        "predict_batch_per_row_p95_ms": p95_ms / max(len(X_sample), 1),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--params",
        default="params_staged_pipeline.yaml",
        help="Path tới file params_staged_pipeline.yaml",
    )

    parser.add_argument(
        "--compare-results",
        default="artifacts/staged_pipeline/model_compare_results.csv",
        help="CSV chứa run_id và mlflow_model_uri của từng run",
    )

    parser.add_argument(
        "--output",
        default="artifacts/staged_pipeline/predict_latency_results.csv",
        help="File CSV output latency",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Số dòng test dùng để đo latency",
    )

    parser.add_argument(
        "--batch-repeats",
        type=int,
        default=10,
        help="Số lần lặp khi đo batch predict",
    )

    parser.add_argument(
        "--warmup-rows",
        type=int,
        default=5,
        help="Số dòng warm up trước khi đo latency",
    )

    parser.add_argument(
        "--thresholds",
        default="0.5,0.35",
        help="Danh sách threshold, ví dụ: 0.5,0.35",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.params)
    setup_mlflow(config)

    target = config["preprocess"]["target_column"]

    thresholds = [
        float(x.strip())
        for x in args.thresholds.split(",")
        if x.strip()
    ]

    compare_df = pd.read_csv(args.compare_results)

    required_cols = [
        "scenario",
        "model_name",
        "run_id",
        "mlflow_model_uri",
    ]

    for col in required_cols:
        if col not in compare_df.columns:
            raise ValueError(
                f"Missing column '{col}' in {args.compare_results}. "
                f"Available columns: {compare_df.columns.tolist()}"
            )

    rows = []

    for _, run_row in compare_df.iterrows():
        scenario = run_row["scenario"]
        model_name = run_row["model_name"]
        run_id = run_row["run_id"]
        model_uri = run_row["mlflow_model_uri"]

        print("=" * 100)
        print(f"Scenario : {scenario}")
        print(f"Model    : {model_name}")
        print(f"Run ID   : {run_id}")
        print(f"URI      : {model_uri}")

        test_path = get_test_path_by_scenario(config, scenario)
        test_df = pd.read_csv(test_path)

        if target not in test_df.columns:
            raise ValueError(
                f"Target column '{target}' not found in {test_path}"
            )

        X_test = test_df.drop(columns=[target])

        if len(X_test) == 0:
            print("Empty test set. Skip.")
            continue

        sample_size = min(args.sample_size, len(X_test))

        X_sample = X_test.sample(
            n=sample_size,
            random_state=args.random_state,
        ).reset_index(drop=True)

        load_start = time.perf_counter()
        model = mlflow.sklearn.load_model(model_uri)
        load_end = time.perf_counter()

        load_model_ms = (load_end - load_start) * 1000

        print(f"Load model latency: {load_model_ms:.2f} ms")

        warmup_n = min(args.warmup_rows, len(X_sample))

        if warmup_n > 0:
            _ = model.predict(X_sample.head(warmup_n))

            if hasattr(model, "predict_proba"):
                _ = model.predict_proba(X_sample.head(warmup_n))

        predict_single_metrics = measure_single_predict(
            model=model,
            X_sample=X_sample,
        )

        predict_proba_metrics = measure_single_predict_proba(
            model=model,
            X_sample=X_sample,
        )

        batch_metrics = measure_batch_predict(
            model=model,
            X_sample=X_sample,
            repeats=args.batch_repeats,
        )

        threshold_rows = measure_threshold_predict(
            model=model,
            X_sample=X_sample,
            thresholds=thresholds,
        )

        base_row = {
            "scenario": scenario,
            "model_name": model_name,
            "run_id": run_id,
            "mlflow_model_uri": model_uri,
            "test_path": test_path,
            "sample_size": int(sample_size),
            "load_model_ms": float(load_model_ms),
            **predict_single_metrics,
            **predict_proba_metrics,
            **batch_metrics,
        }

        if threshold_rows:
            for threshold_row in threshold_rows:
                rows.append(
                    {
                        **base_row,
                        **threshold_row,
                    }
                )
        else:
            rows.append(
                {
                    **base_row,
                    "threshold": None,
                    "threshold_predict_single_mean_ms": None,
                    "threshold_predict_single_p50_ms": None,
                    "threshold_predict_single_p95_ms": None,
                    "threshold_predict_single_p99_ms": None,
                    "threshold_predict_single_min_ms": None,
                    "threshold_predict_single_max_ms": None,
                }
            )

        print("Latency done.")

    result_df = pd.DataFrame(rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)

    print("=" * 100)
    print(f"Saved latency result to: {args.output}")

    if len(result_df) > 0:
        display_cols = [
            "scenario",
            "model_name",
            "threshold",
            "sample_size",
            "load_model_ms",
            "predict_single_p50_ms",
            "predict_single_p95_ms",
            "predict_proba_single_p50_ms",
            "predict_proba_single_p95_ms",
            "threshold_predict_single_p50_ms",
            "threshold_predict_single_p95_ms",
            "predict_batch_per_row_mean_ms",
        ]

        existing_cols = [
            col for col in display_cols
            if col in result_df.columns
        ]

        print("\nTop latency result:")
        print(
            result_df[existing_cols]
            .sort_values(
                by=[
                    "threshold_predict_single_p95_ms",
                    "threshold_predict_single_p50_ms",
                ],
                na_position="last",
            )
            .head(30)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()