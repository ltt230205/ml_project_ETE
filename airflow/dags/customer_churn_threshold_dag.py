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

BASE_ENV = {
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",
    "AWS_ACCESS_KEY_ID": "minioadmin",
    "AWS_SECRET_ACCESS_KEY": "minioadmin123",
    "AWS_DEFAULT_REGION": "us-east-1",
    "PYTHONUNBUFFERED": "1",
}


def bash(command: str) -> str:
    return f"""
set -euo pipefail

export PATH="/opt/project-venv/bin:$PATH"

cd "{PROJECT_DIR}"

echo "========================================"
echo "Current directory:"
pwd

echo "========================================"
echo "Python version:"
python --version

echo "========================================"
echo "DVC version:"
dvc version

echo "========================================"
echo "Configure DVC for Docker bind mount:"
dvc config --local cache.type copy

echo "========================================"
echo "Command:"
cat <<'COMMAND_EOF'
{command}
COMMAND_EOF

echo "========================================"
{command}
"""


with DAG(
    dag_id="customer_churn_threshold_pipeline",
    description="Run independent threshold-aware customer churn pipeline",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["customer-churn", "threshold", "mlflow", "dvc"],
) as dag:

    check_project = BashOperator(
        task_id="check_project",
        bash_command=bash(
            """
test -f params_threshold.yaml
test -f src/preprocess_threshold.py
test -f src/train_v2_threshold.py
test -f dvc.yaml

python -c "import mlflow, pandas, sklearn; print('packages ok')"
"""
        ),
        env=BASE_ENV,
        append_env=True,
    )

    threshold_pipeline = BashOperator(
        task_id="dvc_repro_threshold_pipeline",
        bash_command=bash(
            """
dvc repro dvc.yaml:threshold_pipeline
"""
        ),
        env=BASE_ENV,
        append_env=True,
    )

    final_status = BashOperator(
        task_id="final_status",
        bash_command=bash(
            """
echo "Final DVC status:"
dvc status || true

echo "Artifacts:"
ls -lah artifacts || true
"""
        ),
        env=BASE_ENV,
        append_env=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    check_project >> threshold_pipeline >> final_status