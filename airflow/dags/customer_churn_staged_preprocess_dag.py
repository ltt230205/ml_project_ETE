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
    dag_id="_customer_churn_staged_preprocess_pipeline",
    description="Staged preprocessing + feature selection + grid search pipeline",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["customer-churn", "staged-preprocess", "grid-search", "feature-selection"],
) as dag:

    check_project = BashOperator(
        task_id="check_project",
        bash_command=bash(
            """
for f in \
  params_staged_pipeline.yaml \
  dvc_staged_pipeline.yaml \
  src/pipeline_stages/01_split_data.py \
  src/pipeline_stages/02_clean_data.py \
  src/pipeline_stages/03_feature_selection.py \
  src/pipeline_stages/04_train_compare_grid.py \
  src/pipeline_stages/05_register_best_model.py
do
  if [ -f "$f" ]; then
    echo "OK      $f"
  else
    echo "MISSING $f"
    exit 1
  fi
done

python -c "import mlflow, pandas, sklearn, imblearn; print('packages ok')"
"""
        ),
        env=BASE_ENV,
        append_env=True,
    )

    run_staged_pipeline = BashOperator(
        task_id="dvc_repro_staged_pipeline",
        bash_command=bash(
            """
dvc repro 
"""
        ),
        env=BASE_ENV,
        append_env=True,
    )

    final_status = BashOperator(
        task_id="final_status",
        bash_command=bash(
            """
echo "Artifacts:"
ls -lah artifacts/staged_pipeline || true

echo "DVC status:"
dvc status || true
"""
        ),
        env=BASE_ENV,
        append_env=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    check_project >> run_staged_pipeline >> final_status
