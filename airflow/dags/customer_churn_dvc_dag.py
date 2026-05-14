from __future__ import annotations

import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule


PROJECT_DIR = Variable.get(
    "CUSTOMER_CHURN_PROJECT_DIR",
    default_var="/opt/airflow/project",
)

DVC_BIN = Variable.get(
    "CUSTOMER_CHURN_DVC_BIN",
    default_var="dvc",
)

BASE_ENV = {
    # MLflow service trong docker-compose network
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",

    # MinIO service trong docker-compose network
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",

    # MinIO credentials
    "AWS_ACCESS_KEY_ID": "minioadmin",
    "AWS_SECRET_ACCESS_KEY": "minioadmin123",
    "AWS_DEFAULT_REGION": "us-east-1",

    # Log realtime hơn
    "PYTHONUNBUFFERED": "1",
}


def dvc_bash(command: str) -> str:
    return f"""
set -euo pipefail

export PATH="/opt/project-venv/bin:$PATH"

cd "{PROJECT_DIR}"

echo "========================================"
echo "Current directory:"
pwd

echo "========================================"
echo "Configure DVC for Docker bind mount..."
dvc config --local cache.type copy

echo "========================================"
echo "Check required files..."
test -f dvc.yaml
test -f params.yaml

echo "========================================"
echo "Python version:"
python --version

echo "========================================"
echo "DVC version:"
dvc version

echo "========================================"
echo "Command to run:"
cat <<'COMMAND_EOF'
{command}
COMMAND_EOF

echo "========================================"
{command}
"""


with DAG(
    dag_id="customer_churn_dvc_pipeline",
    description="Run customer churn pipeline using dvc.yaml and params.yaml",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["customer-churn", "dvc", "mlflow", "minio"],
) as dag:

    check_project = BashOperator(
        task_id="check_project",
        bash_command=dvc_bash(
            f"""
echo "Check project files..."
ls -lah

echo "Check source folder..."
ls -lah src || true

echo "Check data folder..."
ls -lah data || true

echo "Check MLflow client..."
python -c "import mlflow; print('mlflow client:', mlflow.__version__)"

echo "Check important ML packages..."
python -c "import pandas, sklearn; print('pandas/sklearn ok')"

echo "Check DVC status..."
{DVC_BIN} status || true
"""
        ),
        env=BASE_ENV,
        append_env=True,
    )

    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=dvc_bash(
            f"""
echo "Check DVC remotes..."
{DVC_BIN} remote list || true

if {DVC_BIN} remote list | grep -q .; then
  echo "DVC remote found. Pulling data..."
  {DVC_BIN} pull
else
  echo "No DVC remote configured. Skip dvc pull."
fi
"""
        ),
        env=BASE_ENV,
        append_env=True,
    )

    preprocess = BashOperator(
        task_id="dvc_repro_preprocess",
        bash_command=dvc_bash(
            f"""
echo "Run preprocess stage..."
{DVC_BIN} repro -f preprocess
"""
        ),
        env=BASE_ENV,
        append_env=True,
    )

    train = BashOperator(
        task_id="dvc_repro_train",
        bash_command=dvc_bash(
            f"""
echo "Run train stage..."
{DVC_BIN} repro -f train
"""
        ),
        env=BASE_ENV,
        append_env=True,
    )

    dvc_push = BashOperator(
        task_id="dvc_push",
        bash_command=dvc_bash(
            f"""
echo "Push DVC-tracked outputs to remote if configured..."

if {DVC_BIN} remote list | grep -q .; then
  echo "DVC remote found. Pushing data..."
  {DVC_BIN} push
else
  echo "No DVC remote configured. Skip dvc push."
fi
"""
        ),
        env=BASE_ENV,
        append_env=True,
    )

    final_status = BashOperator(
        task_id="final_dvc_status",
        bash_command=dvc_bash(
            f"""
echo "Final DVC status..."
{DVC_BIN} status || true

echo "Final Git status..."
git status --short || true
"""
        ),
        env=BASE_ENV,
        append_env=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    check_project >> dvc_pull >> preprocess >> train >> dvc_push >> final_status