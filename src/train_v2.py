import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from imblearn.pipeline import Pipeline as ImbPipeline 
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (classification_report, accuracy_score, f1_score, 
                             recall_score, precision_score, make_scorer)
from sklearn.base import BaseEstimator, TransformerMixin
import os
import numpy as np

# Import các model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


# 🔹 CUSTOM TRANSFORMER CHO LABEL ENCODING
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Label Encoder cho nhiều cột categorical.
    Tự động fit trên train data và transform trên test/inference data.
    """
    def __init__(self, columns=None):
        """
        columns: list các tên cột categorical cần encode
        """
        self.columns = columns
        self.encoders = {}
    
    def fit(self, X, y=None):
        # X là pandas DataFrame
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in self.columns:
            if col in X.columns:
                le = LabelEncoder()
                # Chuyển về string để đảm bảo consistency
                le.fit(X[col].astype(str))
                self.encoders[col] = le
            else:
                print(f"⚠️ Column '{col}' not found in data")
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col, le in self.encoders.items():
            if col in X_copy.columns:
                # Xử lý category mới chưa thấy lúc train
                def safe_transform(x):
                    x_str = str(x)
                    if x_str in le.classes_:
                        return le.transform([x_str])[0]
                    else:
                        # Gán -1 cho category lạ (hoặc có thể dùng 0)
                        return -1
                
                X_copy[col] = X_copy[col].apply(safe_transform)
            else:
                print(f"⚠️ Column '{col}' not found during transform")
        
        return X_copy


def train():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Đọc dữ liệu (DATA NGUYÊN BẢN, CHƯA ENCODE)
    train_df = pd.read_csv(config['processed_data']['train_path'])
    test_df = pd.read_csv(config['processed_data']['test_path'])
    target = config['preprocess']['target_column']

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # 2. Xác định categorical columns (tự động)
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"📋 Categorical columns sẽ được encode: {categorical_cols}")

    # 3. Định nghĩa metrics
    scoring_metrics = {
        'accuracy': 'accuracy',
        'f1_c1': make_scorer(f1_score, pos_label=1),
        'f1_c0': make_scorer(f1_score, pos_label=0),
        'recall_c1': make_scorer(recall_score, pos_label=1),
        'recall_c0': make_scorer(recall_score, pos_label=0),
        'precision_c1': make_scorer(precision_score, pos_label=1),
        'precision_c0': make_scorer(precision_score, pos_label=0)
    }

    scalers = {"MinMax": MinMaxScaler(), "Standard": StandardScaler()}
    sampling_methods = config['preprocess'].get('sampling_methods', ['none', 'smote'])

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
        "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    }

    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # --- VÒNG LẶP HUẤN LUYỆN ---
    for s_name, scaler in scalers.items():
        for sampling in sampling_methods:
            for m_name, model in models.items():
                run_name = f"{m_name}_{s_name}_SMOTE_{sampling}"
                
                with mlflow.start_run(run_name=run_name):
                    # 🔹 PIPELINE HOÀN CHỈNH:
                    # LabelEncoder → Scaler → SMOTE → Model
                    steps = [
                        ('label_encoder', MultiColumnLabelEncoder(columns=categorical_cols)),
                        ('scaler', scaler)
                    ]
                    
                    if sampling == "smote":
                        steps.append(('smote', SMOTE(random_state=42)))
                    
                    steps.append(('model', model))
                    
                    pipeline = ImbPipeline(steps)
                    
                    # Cross-validation
                    cv_results = cross_validate(
                        pipeline, X_train, y_train, 
                        cv=config['preprocess']['cv_folds'],
                        scoring=scoring_metrics,
                        n_jobs=-1 
                    )

                    # Log params
                    mlflow.log_params({
                        "model_type": m_name, 
                        "scaler_type": s_name, 
                        "sampling_type": sampling,
                        "categorical_columns": str(categorical_cols)
                    })
                    
                    # Log metrics
                    mlflow.log_metrics({
                        "cv_accuracy": cv_results['test_accuracy'].mean(),
                        "cv_f1_churned": cv_results['test_f1_c1'].mean(),
                        "cv_f1_stayed": cv_results['test_f1_c0'].mean(),
                        "cv_recall_churned": cv_results['test_recall_c1'].mean(),
                        "cv_recall_stayed": cv_results['test_recall_c0'].mean(),
                        "cv_precision_churned": cv_results['test_precision_c1'].mean(),
                        "cv_precision_stayed": cv_results['test_precision_c0'].mean()
                    })

                    # Fit pipeline trên toàn bộ train set
                    pipeline.fit(X_train, y_train)
                    
                    # Log model vào MLflow
                    mlflow.sklearn.log_model(
                        sk_model=pipeline, 
                        artifact_path="model_file",
                        input_example=X_train[:5].to_dict(orient="records")  # Schema input
                    )
                    
                    print(f"✅ Done CV: {run_name} | Acc: {cv_results['test_accuracy'].mean():.4f}")

    # --- TÌM MODEL TỐT NHẤT VÀ REGISTER ---
    print("\n--- Tìm model tốt nhất để register ---")
    
    experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["metrics.cv_accuracy DESC"]
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

    # Load best pipeline
    best_pipeline = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model_file")
    
    # Evaluate trên test set
    y_pred = best_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Register vào Model Registry
    with mlflow.start_run(run_name="register_best_model"):
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model_final",
            registered_model_name=registry_name,
            input_example=X_test[:5].to_dict(orient="records")
        )
        
        mlflow.log_metrics({
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1_churned": report['1']['f1-score'],
            "test_f1_stayed": report['0']['f1-score'],
            "test_recall_churned": report['1']['recall'],
            "test_precision_churned": report['1']['precision']
        })
        
        print(f"✅ Đã register '{registry_name}' vào Model Registry")
        print(classification_report(y_test, y_pred))

    # Tự động promote lên Production
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Lấy version mới nhất
        model_versions = client.get_latest_versions(registry_name, stages=["None"])
        if len(model_versions) > 0:
            latest_version = model_versions[0].version
            client.transition_model_version_stage(
                name=registry_name,
                version=latest_version,
                stage="Production"
            )
            print(f"🚀 Đã chuyển {registry_name} v{latest_version} sang Production")
    except Exception as e:
        print(f"⚠️ Không thể auto-promote: {e}")


if __name__ == "__main__":
    # Cấu hình MLflow
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = "minioadmin"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "minioadmin123"
    mlflow.set_tracking_uri("http://localhost:5000") 
    
    train()