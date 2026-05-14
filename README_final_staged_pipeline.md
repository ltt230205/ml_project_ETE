# Final staged MLOps pipeline

Pipeline này chia từng giai đoạn thành file riêng, dùng DVC để chạy tuần tự.

## Models

Chỉ dùng 4 model:

- Logistic_Regression
- Random_Forest
- Naive_Bayes
- SVM

## Threshold comparison

So sánh cả:

- threshold = 0.5
- threshold = 0.35

Mặc định chọn model tốt nhất theo F1 tại threshold 0.35.

## Stages

- split_data
- clean_data
- feature_selection
- train_compare_grid
- register_best_model

## Run

```bash
docker compose exec airflow-worker bash -lc "
cd /opt/airflow/project &&
export PATH=/opt/project-venv/bin:\$PATH &&
export MLFLOW_TRACKING_URI=http://mlflow:5000 &&
export MLFLOW_S3_ENDPOINT_URL=http://minio:9000 &&
dvc repro dvc_staged_pipeline.yaml
"
```
