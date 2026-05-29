from __future__ import annotations

import hashlib
import os
from datetime import timedelta
from pathlib import Path

import pendulum
import yaml

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule


PROJECT_DIR = Variable.get(
    "CUSTOMER_CHURN_PROJECT_DIR",
    default_var=os.getenv("CUSTOMER_CHURN_PROJECT_DIR", "/opt/airflow/project"),
)

LOCAL_LOG_RETENTION_DAYS = Variable.get(
    "AIRFLOW_LOCAL_LOG_RETENTION_DAYS",
    default_var=os.getenv("AIRFLOW_LOCAL_LOG_RETENTION_DAYS", "3"),
)

BASE_ENV = {
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
    "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    "PARAMS_PATH": os.getenv("PARAMS_PATH", "params_staged_pipeline.yaml"),
    "PYTHONUNBUFFERED": os.getenv("PYTHONUNBUFFERED", "1"),
    "PYTHONPATH": os.getenv("PYTHONPATH", "/opt/airflow/project"),
    "DVC_NO_ANALYTICS": os.getenv("DVC_NO_ANALYTICS", "1"),
    "THREAD_POOL_MAX_WORKERS": os.getenv("THREAD_POOL_MAX_WORKERS", "2"),
    "GRID_SEARCH_METHOD": os.getenv("GRID_SEARCH_METHOD", "early_stop"),
    "GRID_SEARCH_SCORING": os.getenv("GRID_SEARCH_SCORING", "f1"),
    "GRID_SEARCH_CV_FOLDS": os.getenv("GRID_SEARCH_CV_FOLDS", "5"),
    "GRID_SEARCH_INNER_N_JOBS": os.getenv("GRID_SEARCH_INNER_N_JOBS", "1"),
    "GRID_SEARCH_VERBOSE": os.getenv("GRID_SEARCH_VERBOSE", "0"),
    "GRID_SEARCH_TRY_SMOTE": os.getenv("GRID_SEARCH_TRY_SMOTE", "true"),
    "GRID_SEARCH_MAX_EVALUATED_CANDIDATES": os.getenv("GRID_SEARCH_MAX_EVALUATED_CANDIDATES", "20"),
    "GRID_SEARCH_MIN_EVALUATED_CANDIDATES": os.getenv("GRID_SEARCH_MIN_EVALUATED_CANDIDATES", "20"),
    "GRID_SEARCH_EARLY_STOP_PATIENCE": os.getenv("GRID_SEARCH_EARLY_STOP_PATIENCE", "10"),
    "GRID_SEARCH_EARLY_STOP_MIN_DELTA": os.getenv("GRID_SEARCH_EARLY_STOP_MIN_DELTA", "0.0001"),
    "GRID_SEARCH_PARAM_SHUFFLE": os.getenv("GRID_SEARCH_PARAM_SHUFFLE", "true"),
    "GRID_SEARCH_RANDOM_STATE": os.getenv("GRID_SEARCH_RANDOM_STATE", "42"),
    "THRESHOLD_VALUES": os.getenv("THRESHOLD_VALUES", "0.5,0.35"),
    "MODEL_SELECTION_THRESHOLD": os.getenv("MODEL_SELECTION_THRESHOLD", "0.35"),
    "MODEL_SELECTION_METRIC": os.getenv("MODEL_SELECTION_METRIC", "test_f1_churned_threshold"),
}

COMMON_TASK_ARGS = {
    "env": BASE_ENV,
    "append_env": True,
    "do_xcom_push": False,
    "retries": 0,
}


def bash(command: str) -> str:
    # Giữ log task ngắn: không in version/check dài ở mọi task.
    return f"""
set -euo pipefail

export PATH="/opt/project-venv/bin:$PATH"
cd "{PROJECT_DIR}"

{command}
"""


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def should_run_pipeline(**context) -> bool:
    """
    Scheduled run: chỉ chạy khi raw data hoặc params thay đổi.
    Manual run: luôn chạy để dễ debug/chạy lại khi bạn chủ động trigger.
    """
    project_dir = Path(PROJECT_DIR)
    params_path = project_dir / "params_staged_pipeline.yaml"

    if not params_path.exists():
        raise FileNotFoundError(f"Missing params file: {params_path}")

    config = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    raw_path = project_dir / config["raw_data"]["path"]

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw data file: {raw_path}")

    raw_hash = _file_md5(raw_path)
    params_hash = _file_md5(params_path)
    current_signature = f"raw={raw_hash};params={params_hash}"

    dag_run = context.get("dag_run")
    run_type = getattr(dag_run, "run_type", "") if dag_run else ""

    # Manual trigger vẫn chạy, nhưng cập nhật signature để scheduled run kế tiếp không chạy lại vô ích.
    if run_type == "manual":
        Variable.set("CUSTOMER_CHURN_LAST_PIPELINE_SIGNATURE", current_signature)
        print("Manual run detected. Run pipeline and update latest data signature.")
        return True

    last_signature = Variable.get(
        "CUSTOMER_CHURN_LAST_PIPELINE_SIGNATURE",
        default_var="",
    )

    if current_signature == last_signature:
        print("Raw data and params unchanged. Skip heavy DVC pipeline.")
        return False

    Variable.set("CUSTOMER_CHURN_LAST_PIPELINE_SIGNATURE", current_signature)
    print("Raw data or params changed. Run DVC pipeline.")
    return True


with DAG(
    dag_id="v2_customer_churn_staged_preprocess_pipeline",
    description="Scheduled customer churn pipeline with short-circuit + optimized grid search",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule="0 2 * * *",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=6),
    tags=["customer-churn", "scheduled", "short-circuit", "grid20"],
) as dag:

    should_run = ShortCircuitOperator(
        task_id="check_should_run_pipeline",
        python_callable=should_run_pipeline,
        ignore_downstream_trigger_rules=False,
        do_xcom_push=False,
    )

    check_project = BashOperator(
        task_id="check_project",
        bash_command=bash(
            """
missing=0
for f in \
  params_staged_pipeline.yaml \
  dvc.yaml \
  src/pipeline_stages_v2/01_split_data.py \
  src/pipeline_stages_v2/02_clean_data.py \
  src/pipeline_stages_v2/03_feature_selection.py \
  src/pipeline_stages_v2/04_train_compare_grid.py \
  src/pipeline_stages_v2/05_register_best_model.py
do
  if [ ! -f "$f" ]; then
    echo "MISSING $f"
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  exit 1
fi

python - <<'PY_CHECK'
import imblearn, mlflow, pandas, sklearn
print("Project check passed")
PY_CHECK
"""
        ),
        **COMMON_TASK_ARGS,
    )
    run_staged_pipeline = BashOperator(
        task_id="dvc_repro_staged_pipeline",
        bash_command=bash(
            """
    mkdir -p artifacts/staged_pipeline/grid_search_history
    touch artifacts/staged_pipeline/grid_search_history/.gitkeep

    echo "Check DVC remote:"
    dvc remote list || true

    echo "Run DVC pipeline:"
    dvc -q repro

    echo "Push DVC cache to remote:"
    dvc -q push
    """
        ),
        env=BASE_ENV,
        append_env=True,
        do_xcom_push=False,
    )

    final_status = BashOperator(
        task_id="final_status",
        bash_command=bash(
            """
echo "Pipeline finished at $(date -Iseconds)"
echo "Artifacts generated:"
if [ -d artifacts/staged_pipeline ]; then
  find artifacts/staged_pipeline -maxdepth 1 -type f -printf "- %f\n" | sort || true
else
  echo "- artifacts/staged_pipeline not found"
fi
"""
        ),
        trigger_rule=TriggerRule.ALL_DONE,
        **COMMON_TASK_ARGS,
    )

    cleanup_local_airflow_logs = BashOperator(
        task_id="cleanup_local_airflow_logs",
        bash_command=bash(
            f"""
LOG_DIR="${{AIRFLOW_HOME:-/opt/airflow}}/logs"
RETENTION_DAYS="${{AIRFLOW_LOCAL_LOG_RETENTION_DAYS:-{LOCAL_LOG_RETENTION_DAYS}}}"

if [ -d "$LOG_DIR" ]; then
  echo "Cleaning local Airflow logs older than $RETENTION_DAYS day(s) in $LOG_DIR"
  find "$LOG_DIR" -type f -name "*.log" -mtime +"$RETENTION_DAYS" -delete || true
  find "$LOG_DIR" -type d -empty -delete || true
else
  echo "Local Airflow log dir not found: $LOG_DIR"
fi
"""
        ),
        trigger_rule=TriggerRule.ALL_DONE,
        **COMMON_TASK_ARGS,
    )

    should_run >> check_project >> run_staged_pipeline >> final_status >> cleanup_local_airflow_logs
