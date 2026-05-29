import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient


def load_config() -> dict:
    params_path = os.getenv("PARAMS_PATH", "params_staged_pipeline.yaml")
    with open(params_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if os.getenv("MLFLOW_EXPERIMENT_NAME"):
        config.setdefault("mlflow", {})["experiment_name"] = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if os.getenv("MLFLOW_REGISTRY_NAME"):
        config.setdefault("mlflow", {})["registry_name"] = os.getenv("MLFLOW_REGISTRY_NAME")

    return config


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


def main():
    config = load_config()
    setup_mlflow(config)

    registry_name = config["mlflow"].get("registry_name", "churn_model_staged_best")

    summary_path = config["staged_pipeline"]["best_summary_path"]
    register_summary_path = config["staged_pipeline"]["register_summary_path"]

    best_summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

    threshold = float(best_summary["selected_threshold"])
    best_model_uri = best_summary["best_model_uri"]
    best_model = mlflow.sklearn.load_model(best_model_uri)

    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="register_best_staged_pipeline_model") as run:
        mlflow.log_params(
            {
                "best_model_uri": best_model_uri,
                "best_scenario": best_summary["best_scenario"],
                "best_model_name": best_summary["best_model_name"],
                "decision_threshold": threshold,
                "selection_metric": best_summary["selection_metric"],
            }
        )

        mlflow.log_metrics(
            {
                "cv_best_score": float(best_summary["cv_best_score"]),
                "test_f1_churned_threshold": float(best_summary["test_f1_churned_threshold"]),
                "test_precision_churned_threshold": float(best_summary["test_precision_churned_threshold"]),
                "test_recall_churned_threshold": float(best_summary["test_recall_churned_threshold"]),
                "test_accuracy_threshold": float(best_summary["test_accuracy_threshold"]),
            }
        )

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model_final",
            registered_model_name=registry_name,
        )

        client = MlflowClient()
        versions = list(client.search_model_versions(f"name='{registry_name}'"))
        matched_versions = [v for v in versions if v.run_id == run.info.run_id]

        registered_version = None
        if matched_versions:
            latest_version = max(matched_versions, key=lambda v: int(v.version))
            registered_version = latest_version.version

            client.set_model_version_tag(
                name=registry_name,
                version=registered_version,
                key="decision_threshold",
                value=str(threshold),
            )
            client.set_model_version_tag(
                name=registry_name,
                version=registered_version,
                key="best_scenario",
                value=best_summary["best_scenario"],
            )
            client.set_model_version_tag(
                name=registry_name,
                version=registered_version,
                key="best_model_name",
                value=best_summary["best_model_name"],
            )

            try:
                client.transition_model_version_stage(
                    name=registry_name,
                    version=registered_version,
                    stage="Production",
                )
            except Exception as e:
                print(f"⚠️ Cannot transition to Production: {e}")

        register_summary = {
            "registry_name": registry_name,
            "registered_version": registered_version,
            "run_id": run.info.run_id,
            "decision_threshold": threshold,
            "best_summary": best_summary,
        }

        Path(register_summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(register_summary_path).write_text(
            json.dumps(register_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("✅ Registered best model")
        print(json.dumps(register_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
