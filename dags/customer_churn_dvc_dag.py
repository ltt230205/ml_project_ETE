from __future__ import annotations

import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule


# Nếu chạy Airflow bằng docker-compose, nên mount project vào /opt/airflow/project
PROJECT_DIR = Variable.get(
    "CUSTOMER_CHURN_PROJECT_DIR",
    default_var="/opt/airflow/project",
)

# Vì dvc.yaml của bạn đang dùng ./venv/bin/python,
# nên DAG cũng dùng DVC trong venv của project.
DVC_BIN = Variable.get(
    "CUSTOMER_CHURN_DVC_BIN",
    default_var="./venv/bin/dvc",
)

BASE_ENV = {
    # Khi chạy trong Docker network, dùng service name thay vì localhost
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
    "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000",

    # MinIO credentials
    "AWS_ACCESS_KEY_ID": "minioadmin",
    "AWS_SECRET_ACCESS_KEY": "minioadmin123",
    "AWS_DEFAULT_REGION": "us-east-1",

    # Tránh Python buffer log, giúp Airflow log hiện realtime hơn
    "PYTHONUNBUFFERED": "1",
}


def dvc_bash(command: str) -> str:
    return f"""
set -euo pipefail

cd "{PROJECT_DIR}"

echo "Current directory:"
pwd

echo "Check required files..."
test -f dvc.yaml
test -f params.yaml

echo "DVC version:"
{DVC_BIN} version

echo "Running command:"
echo "{command}"

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
            """
echo "Check Python in project venv..."
./venv/bin/python --version

echo "Check MLflow client version..."
./venv/bin/python -c "import mlflow; print('mlflow client:', mlflow.__version__)"

echo "Check DVC status..."
./venv/bin/dvc status || true
"""
        ),
        env=BASE_ENV,
    )

    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=dvc_bash(
            """
echo "Pull data from DVC remote if configured..."

if ./venv/bin/dvc remote list | grep -q .; then
  ./venv/bin/dvc pull
else
  echo "No DVC remote configured. Skip dvc pull."
fi
"""
        ),
        env=BASE_ENV,
    )

    preprocess = BashOperator(
        task_id="dvc_repro_preprocess",
        bash_command=dvc_bash(
            """
./venv/bin/dvc repro -f preprocess
"""
        ),
        env=BASE_ENV,
    )

    train = BashOperator(
        task_id="dvc_repro_train",
        bash_command=dvc_bash(
            """
./venv/bin/dvc repro -f train
"""
        ),
        env=BASE_ENV,
    )

    dvc_push = BashOperator(
        task_id="dvc_push",
        bash_command=dvc_bash(
            """
echo "Push DVC-tracked outputs to remote if configured..."

if ./venv/bin/dvc remote list | grep -q .; then
  ./venv/bin/dvc push
else
  echo "No DVC remote configured. Skip dvc push."
fi
"""
        ),
        env=BASE_ENV,
    )

    final_status = BashOperator(
        task_id="final_dvc_status",
        bash_command=dvc_bash(
            """
./venv/bin/dvc status || true
"""
        ),
        env=BASE_ENV,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    check_project >> dvc_pull >> preprocess >> train >> dvc_push >> final_status