# from logging import config

# import pandas as pd
# import yaml
# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import cross_validate
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
# from imblearn.pipeline import Pipeline as ImbPipeline 
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import (classification_report, accuracy_score, f1_score, 
#                              recall_score, precision_score, make_scorer)
# from sklearn.base import BaseEstimator, TransformerMixin
# import os
# import numpy as np

# # Import các model
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
#                               AdaBoostClassifier, ExtraTreesClassifier)
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier


# # 🔹 CUSTOM TRANSFORMER CHO LABEL ENCODING
# class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
#     """
#     Label Encoder cho nhiều cột categorical.
#     Tự động fit trên train data và transform trên test/inference data.
#     """
#     def __init__(self, columns=None):
#         """
#         columns: list các tên cột categorical cần encode
#         """
#         self.columns = columns
#         self.encoders = {}
    
#     def fit(self, X, y=None):
#         # X là pandas DataFrame
#         if self.columns is None:
#             self.columns = X.select_dtypes(include=['object']).columns.tolist()
        
#         for col in self.columns:
#             if col in X.columns:
#                 le = LabelEncoder()
#                 # Chuyển về string để đảm bảo consistency
#                 le.fit(X[col].astype(str))
#                 self.encoders[col] = le
#             else:
#                 print(f"⚠️ Column '{col}' not found in data")
        
#         return self
    
#     def transform(self, X):
#         X_copy = X.copy()
        
#         for col, le in self.encoders.items():
#             if col in X_copy.columns:
#                 # Xử lý category mới chưa thấy lúc train
#                 def safe_transform(x):
#                     x_str = str(x)
#                     if x_str in le.classes_:
#                         return le.transform([x_str])[0]
#                     else:
#                         # Gán -1 cho category lạ (hoặc có thể dùng 0)
#                         return -1
                
#                 X_copy[col] = X_copy[col].apply(safe_transform)
#             else:
#                 print(f"⚠️ Column '{col}' not found during transform")
        
#         return X_copy


# def train():
#     with open("params.yaml", "r") as f:
#         config = yaml.safe_load(f)

#     # 1. Đọc dữ liệu (DATA NGUYÊN BẢN, CHƯA ENCODE)
#     train_df = pd.read_csv(config['processed_data']['train_path'])
#     test_df = pd.read_csv(config['processed_data']['test_path'])
#     target = config['preprocess']['target_column']

#     X_train = train_df.drop(columns=[target])
#     y_train = train_df[target]
#     X_test = test_df.drop(columns=[target])
#     y_test = test_df[target]

#     # 2. Xác định categorical columns (tự động)
#     categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
#     print(f"📋 Categorical columns sẽ được encode: {categorical_cols}")

#     # 3. Định nghĩa metrics
#     scoring_metrics = {
#         'accuracy': 'accuracy',
#         'f1_c1': make_scorer(f1_score, pos_label=1),
#         'f1_c0': make_scorer(f1_score, pos_label=0),
#         'recall_c1': make_scorer(recall_score, pos_label=1),
#         'recall_c0': make_scorer(recall_score, pos_label=0),
#         'precision_c1': make_scorer(precision_score, pos_label=1),
#         'precision_c0': make_scorer(precision_score, pos_label=0)
#     }

#     scalers = {"MinMax": MinMaxScaler(), "Standard": StandardScaler()}
#     sampling_methods = config['preprocess'].get('sampling_methods', ['none', 'smote'])

#     models = {
#         "Logistic_Regression": LogisticRegression(max_iter=1000),
#         "Decision_Tree": DecisionTreeClassifier(),
#         "Random_Forest": RandomForestClassifier(n_estimators=200),
#         "Gradient_Boosting": GradientBoostingClassifier(),
#         "SVM": SVC(probability=True),
#         "KNN": KNeighborsClassifier(),
#         "Naive_Bayes": GaussianNB(),
#         "AdaBoost": AdaBoostClassifier(),
#         "Extra_Trees": ExtraTreesClassifier(n_estimators=200),
#         "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
#     }

#     mlflow.set_experiment(config['mlflow']['experiment_name'])

#     # --- VÒNG LẶP HUẤN LUYỆN ---
#     for s_name, scaler in scalers.items():
#         for sampling in sampling_methods:
#             for m_name, model in models.items():
#                 run_name = f"{m_name}_{s_name}_SMOTE_{sampling}"
                
#                 with mlflow.start_run(run_name=run_name):
#                     # 🔹 PIPELINE HOÀN CHỈNH:
#                     # LabelEncoder → Scaler → SMOTE → Model
#                     steps = [
#                         ('label_encoder', MultiColumnLabelEncoder(columns=categorical_cols)),
#                         ('scaler', scaler)
#                     ]
                    
#                     if sampling == "smote":
#                         steps.append(('smote', SMOTE(random_state=42)))
                    
#                     steps.append(('model', model))
                    
#                     pipeline = ImbPipeline(steps)
                    
#                     # Cross-validation
#                     cv_results = cross_validate(
#                         pipeline, X_train, y_train, 
#                         cv=config['preprocess']['cv_folds'],
#                         scoring=scoring_metrics,
#                         n_jobs=-1 
#                     )

#                     # Log params
#                     mlflow.log_params({
#                         "model_type": m_name, 
#                         "scaler_type": s_name, 
#                         "sampling_type": sampling,
#                         "categorical_columns": str(categorical_cols)
#                     })
                    
#                     # Log metrics
#                     mlflow.log_metrics({
#                         "cv_accuracy": cv_results['test_accuracy'].mean(),
#                         "cv_f1_churned": cv_results['test_f1_c1'].mean(),
#                         "cv_f1_stayed": cv_results['test_f1_c0'].mean(),
#                         "cv_recall_churned": cv_results['test_recall_c1'].mean(),
#                         "cv_recall_stayed": cv_results['test_recall_c0'].mean(),
#                         "cv_precision_churned": cv_results['test_precision_c1'].mean(),
#                         "cv_precision_stayed": cv_results['test_precision_c0'].mean()
#                     })

#                     # Fit pipeline trên toàn bộ train set
#                     pipeline.fit(X_train, y_train)
                    
#                     # Log model vào MLflow
#                     mlflow.sklearn.log_model(
#                         sk_model=pipeline, 
#                         artifact_path="model_file",
#                         input_example=X_train[:5].to_dict(orient="records")  # Schema input
#                     )
                    
#                     print(f"✅ Done CV: {run_name} | Acc: {cv_results['test_accuracy'].mean():.4f}")

#     # --- TÌM MODEL TỐT NHẤT VÀ REGISTER ---
#     print("\n--- Tìm model tốt nhất để register ---")
    
#     experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
#     runs = mlflow.search_runs(
#         experiment_ids=[experiment.experiment_id], 
#         order_by=["metrics.cv_accuracy DESC"]
#     )
    
#     if len(runs) == 0:
#         print("❌ Không tìm thấy run nào!")
#         return
    
#     best_run = runs.iloc[0]
#     best_run_id = best_run.run_id
#     best_model_architecture = best_run["tags.mlflow.runName"]
#     registry_name = "churn_model_final"
    
#     print(f"🏆 Model tốt nhất: {best_model_architecture}")
#     print(f"📊 CV Accuracy: {best_run['metrics.cv_accuracy']:.4f}")

#     # Load best pipeline
#     best_pipeline = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model_file")
    
#     # Evaluate trên test set
#     y_pred = best_pipeline.predict(X_test)
#     report = classification_report(y_test, y_pred, output_dict=True)
    
#     # Register vào Model Registry
#     with mlflow.start_run(run_name="register_best_model"):
#         mlflow.sklearn.log_model(
#             sk_model=best_pipeline,
#             artifact_path="model_final",
#             registered_model_name=registry_name,
#             input_example=X_test[:5].to_dict(orient="records")
#         )
        
#         mlflow.log_metrics({
#             "test_accuracy": accuracy_score(y_test, y_pred),
#             "test_f1_churned": report['1']['f1-score'],
#             "test_f1_stayed": report['0']['f1-score'],
#             "test_recall_churned": report['1']['recall'],
#             "test_precision_churned": report['1']['precision']
#         })
        
#         print(f"✅ Đã register '{registry_name}' vào Model Registry")
#         print(classification_report(y_test, y_pred))

#     # Tự động promote lên Production
#     try:
#         from mlflow.tracking import MlflowClient
#         client = MlflowClient()
        
#         # Lấy version mới nhất
#         model_versions = client.get_latest_versions(registry_name, stages=["None"])
#         if len(model_versions) > 0:
#             latest_version = model_versions[0].version
#             client.transition_model_version_stage(
#                 name=registry_name,
#                 version=latest_version,
#                 stage="Production"
#             )
#             print(f"🚀 Đã chuyển {registry_name} v{latest_version} sang Production")
#     except Exception as e:
#         print(f"⚠️ Không thể auto-promote: {e}")


# if __name__ == "__main__":
#     # Cấu hình MLflow
#     # os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
#     os.environ['AWS_ACCESS_KEY_ID'] = "minioadmin"
#     os.environ['AWS_SECRET_ACCESS_KEY'] = "minioadmin123"
#     tracking_uri = os.getenv(
#     "MLFLOW_TRACKING_URI",
#     config["mlflow"].get("tracking_uri", "http://localhost:5000"),
# )
#     s3_endpoint = os.getenv(
#     "MLFLOW_S3_ENDPOINT_URL",
#     config["mlflow"].get("s3_endpoint", "http://localhost:9000"),
#     )

#     os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint

#     mlflow.set_tracking_uri(tracking_uri)
    
#     train()

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def load_config(path: str = "params.yaml") -> dict:
    """Load params.yaml as a Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_running_inside_airflow_container() -> bool:
    """
    Detect whether the script is running inside the Airflow Docker container.
    This helps convert localhost URLs to Docker Compose service names.
    """
    cwd = str(Path.cwd())
    return (
        os.getenv("AIRFLOW_HOME") is not None
        or cwd.startswith("/opt/airflow")
        or Path("/opt/airflow").exists()
    )


def normalize_url_for_container(url: str | None, service_name: str, port: int) -> str:
    """
    If the script is running inside Docker/Airflow and URL points to localhost,
    convert it to Docker Compose service name.
    """
    if not url:
        return f"http://localhost:{port}"

    if is_running_inside_airflow_container():
        url = url.replace(f"http://localhost:{port}", f"http://{service_name}:{port}")
        url = url.replace(f"http://127.0.0.1:{port}", f"http://{service_name}:{port}")

    return url


def setup_mlflow(config: dict) -> None:
    """
    Configure MLflow tracking server and MinIO/S3 endpoint.

    Priority:
    1. Environment variables from Airflow DAG
    2. params.yaml
    3. Local defaults
    """
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

    print(f"🔗 MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"🪣 MLflow S3 endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Label Encoder cho nhiều cột categorical.
    Tự động fit trên train data và transform trên test/inference data.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=["object"]).columns.tolist()

        for col in self.columns:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders[col] = le
            else:
                print(f"⚠️ Column '{col}' not found in data")

        return self

    def transform(self, X):
        X_copy = X.copy()

        for col, le in self.encoders.items():
            if col in X_copy.columns:

                def safe_transform(x):
                    x_str = str(x)
                    if x_str in le.classes_:
                        return le.transform([x_str])[0]
                    return -1

                X_copy[col] = X_copy[col].apply(safe_transform)
            else:
                print(f"⚠️ Column '{col}' not found during transform")

        return X_copy


def get_class_metric(report: dict, label: str, metric: str, default: float = 0.0) -> float:
    """
    Safely get classification_report metric.
    classification_report output_dict usually uses string labels like '0' and '1'.
    """
    return float(report.get(label, {}).get(metric, default))


def train():
    config = load_config("params.yaml")
    setup_mlflow(config)

    train_df = pd.read_csv(config["processed_data"]["train_path"])
    test_df = pd.read_csv(config["processed_data"]["test_path"])
    target = config["preprocess"]["target_column"]

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    print(f"📋 Categorical columns sẽ được encode: {categorical_cols}")

    scoring_metrics = {
        "accuracy": "accuracy",
        "f1_c1": make_scorer(f1_score, pos_label=1),
        "f1_c0": make_scorer(f1_score, pos_label=0),
        "recall_c1": make_scorer(recall_score, pos_label=1),
        "recall_c0": make_scorer(recall_score, pos_label=0),
        "precision_c1": make_scorer(precision_score, pos_label=1),
        "precision_c0": make_scorer(precision_score, pos_label=0),
    }

    scalers = {
        "MinMax": MinMaxScaler(),
        "Standard": StandardScaler(),
    }

    sampling_methods = config["preprocess"].get("sampling_methods", ["none", "smote"])

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Decision_Tree": DecisionTreeClassifier(),
        "Random_Forest": RandomForestClassifier(n_estimators=200),
        "Gradient_Boosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Naive_Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(),
        "Extra_Trees": ExtraTreesClassifier(n_estimators=200),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
    }

    experiment_name = config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    for s_name, scaler in scalers.items():
        for sampling in sampling_methods:
            for m_name, model in models.items():
                run_name = f"{m_name}_{s_name}_SMOTE_{sampling}"

                with mlflow.start_run(run_name=run_name):
                    steps = [
                        ("label_encoder", MultiColumnLabelEncoder(columns=categorical_cols)),
                        ("scaler", scaler),
                    ]

                    if sampling == "smote":
                        steps.append(("smote", SMOTE(random_state=42)))

                    steps.append(("model", model))
                    pipeline = ImbPipeline(steps)

                    cv_results = cross_validate(
                        pipeline,
                        X_train,
                        y_train,
                        cv=config["preprocess"]["cv_folds"],
                        scoring=scoring_metrics,
                        n_jobs=-1,
                    )

                    mlflow.log_params(
                        {
                            "model_type": m_name,
                            "scaler_type": s_name,
                            "sampling_type": sampling,
                            "categorical_columns": str(categorical_cols),
                        }
                    )

                    mlflow.log_metrics(
                        {
                            "cv_accuracy": float(cv_results["test_accuracy"].mean()),
                            "cv_f1_churned": float(cv_results["test_f1_c1"].mean()),
                            "cv_f1_stayed": float(cv_results["test_f1_c0"].mean()),
                            "cv_recall_churned": float(cv_results["test_recall_c1"].mean()),
                            "cv_recall_stayed": float(cv_results["test_recall_c0"].mean()),
                            "cv_precision_churned": float(cv_results["test_precision_c1"].mean()),
                            "cv_precision_stayed": float(cv_results["test_precision_c0"].mean()),
                        }
                    )

                    pipeline.fit(X_train, y_train)

                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        artifact_path="model_file",
                        input_example=X_train.head(5),
                    )

                    print(
                        f"✅ Done CV: {run_name} | "
                        f"Acc: {cv_results['test_accuracy'].mean():.4f}"
                    )

    print("\n--- Tìm model tốt nhất để register ---")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"❌ Không tìm thấy experiment: {experiment_name}")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.cv_accuracy DESC"],
    )

    if len(runs) == 0:
        print("❌ Không tìm thấy run nào!")
        return

    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    best_model_architecture = best_run["tags.mlflow.runName"]
    registry_name = "churn_model_final"

    print(f"🏆 Model tốt nhất: {best_model_architecture}")
    print(f"📊 CV Accuracy: {best_run['metrics.cv_accuracy']:.4f}")

    best_pipeline = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model_file")

    y_pred = best_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    with mlflow.start_run(run_name="register_best_model"):
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model_final",
            registered_model_name=registry_name,
            input_example=X_test.head(5),
        )

        mlflow.log_metrics(
            {
                "test_accuracy": float(accuracy_score(y_test, y_pred)),
                "test_f1_churned": get_class_metric(report, "1", "f1-score"),
                "test_f1_stayed": get_class_metric(report, "0", "f1-score"),
                "test_recall_churned": get_class_metric(report, "1", "recall"),
                "test_precision_churned": get_class_metric(report, "1", "precision"),
            }
        )

        print(f"✅ Đã register '{registry_name}' vào Model Registry")
        print(classification_report(y_test, y_pred))

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        model_versions = client.get_latest_versions(registry_name, stages=["None"])

        if len(model_versions) > 0:
            latest_version = model_versions[0].version
            client.transition_model_version_stage(
                name=registry_name,
                version=latest_version,
                stage="Production",
            )
            print(f"🚀 Đã chuyển {registry_name} v{latest_version} sang Production")

    except Exception as e:
        print(f"⚠️ Không thể auto-promote: {e}")


if __name__ == "__main__":
    train()
