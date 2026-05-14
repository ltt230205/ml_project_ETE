FROM apache/airflow:2.8.0-python3.10

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/project-venv \
    && chown -R airflow:0 /opt/project-venv

USER airflow

# Quan trọng: tắt pip --user mặc định của Airflow image
ENV PIP_USER=false
ENV PATH="/opt/project-venv/bin:${PATH}"

RUN /opt/project-venv/bin/python -m pip install --upgrade "pip<26" "setuptools<82" wheel \
    && /opt/project-venv/bin/python -m pip install --no-cache-dir \
        "dvc[s3]==3.67.1" \
        "mlflow==2.12.2" \
        "pandas" \
        "numpy" \
        "scikit-learn" \
        "imbalanced-learn" \
        "pyyaml" \
        "matplotlib" \
        "seaborn" \
        "joblib" \
        "xgboost" \
        "lightgbm"