# import pandas as pd
# import yaml
# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import cross_validate
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from imblearn.pipeline import Pipeline as ImbPipeline 
# from imblearn.over_sampling import SMOTE
# import os

# # Import 10 models
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
#                               AdaBoostClassifier, ExtraTreesClassifier)
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier

# def train():
#     with open("params.yaml", "r") as f:
#         config = yaml.safe_load(f)

#     df = pd.read_csv(config['processed_data']['path'])
#     X = df.drop(columns=[config['preprocess']['target_column']])
#     y = df[config['preprocess']['target_column']]

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
#         "XGBoost": XGBClassifier(eval_metric="logloss")
#     }

#     mlflow.set_experiment(config['mlflow']['experiment_name'])

#     for s_name, scaler in scalers.items():
#         for sampling in sampling_methods:
#             for m_name, model in models.items():
                
#                 # Run Name này sẽ hiển thị trên UI giúp bạn phân biệt
#                 run_name = f"{m_name}_{s_name}_SMOTE_{sampling}"
                
#                 with mlflow.start_run(run_name=run_name):
#                     # 1. Pipeline
#                     steps = [('scaler', scaler)]
#                     if sampling == "smote":
#                         steps.append(('smote', SMOTE(random_state=42)))
#                     steps.append(('model', model))
                    
#                     pipeline = ImbPipeline(steps)
                    
#                     # 2. Cross-Validation
#                     cv_results = cross_validate(
#                         pipeline, X, y, 
#                         cv=config['preprocess']['cv_folds'],
#                         scoring=['f1', 'accuracy', 'recall'],
#                         n_jobs=-1 
#                     )

#                     # 3. Log Params & Metrics
#                     mlflow.log_params({
#                         "model_type": m_name, 
#                         "scaler_type": s_name, 
#                         "sampling_type": sampling
#                     })
                    
#                     metrics = {
#                         "f1_mean": cv_results['test_f1'].mean(),
#                         "recall_mean": cv_results['test_recall'].mean(),
#                         "accuracy_mean": cv_results['test_accuracy'].mean()
#                     }
#                     mlflow.log_metrics(metrics)

#                     # 4. SỬA LỖI TẠI ĐÂY: Lưu Artifacts vào MinIO
#                     pipeline.fit(X, y)
                    
#                     # CHỈ đặt tên thư mục đơn giản, KHÔNG dùng dấu "/" trong tham số artifact_path
#                     # MLflow sẽ tự tạo cấu trúc: mlruns/<run_id>/artifacts/model_file
#                     mlflow.sklearn.log_model(
#                         sk_model=pipeline, 
#                         artifact_path="model_file" 
#                     )
                    
#                     print(f"Finished: {run_name} | F1: {metrics['f1_mean']:.4f}")

# if __name__ == "__main__":
#     # Cấu hình MinIO (S3)
#     os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
#     os.environ['AWS_ACCESS_KEY_ID'] = "minioadmin"
#     os.environ['AWS_SECRET_ACCESS_KEY'] = "minioadmin123"

#     # Trỏ về MLflow Server trong Docker
#     mlflow.set_tracking_uri("http://localhost:5000") 
    
#     train()


import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline 
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
import os

# Import các model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def train():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Đọc dữ liệu Train và Test riêng biệt
    train_df = pd.read_csv(config['processed_data']['train_path'])
    test_df = pd.read_csv(config['processed_data']['test_path'])
    target = config['preprocess']['target_column']

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

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
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # --- VÒNG LẶP HUẤN LUYỆN 40 KỊCH BẢN ---
    for s_name, scaler in scalers.items():
        for sampling in sampling_methods:
            for m_name, model in models.items():
                run_name = f"{m_name}_{s_name}_SMOTE_{sampling}"
                
                with mlflow.start_run(run_name=run_name):
                    steps = [('scaler', scaler)]
                    if sampling == "smote":
                        steps.append(('smote', SMOTE(random_state=42)))
                    steps.append(('model', model))
                    
                    pipeline = ImbPipeline(steps)
                    
                    # Cross-Validation để xem độ ổn định
                    cv_results = cross_validate(
                        pipeline, X_train, y_train, 
                        cv=config['preprocess']['cv_folds'],
                        scoring=['f1', 'accuracy', 'recall', 'precision'],
                        n_jobs=-1 
                    )

                    # Log Params
                    mlflow.log_params({
                        "model_type": m_name, 
                        "scaler_type": s_name, 
                        "sampling_type": sampling
                    })
                    
                    # Log các giá trị trung bình từ CV
                    mlflow.log_metrics({
                        "cv_f1": cv_results['test_f1'].mean(),
                        "cv_recall": cv_results['test_recall'].mean(),
                        "cv_accuracy": cv_results['test_accuracy'].mean(),
                        "cv_precision": cv_results['test_precision'].mean()
                    })

                    # Fit model trên toàn bộ tập Train để lưu artifact
                    pipeline.fit(X_train, y_train)
                    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model_file")
                    print(f"CV Done: {run_name} | CV Acc: {cv_results['test_accuracy'].mean():.4f}")

    # --- BƯỚC CUỐI: TÌM MODEL TỐT NHẤT VÀ KIỂM TRA TRÊN TẬP TEST ---
    print("\n--- Đang tìm model có Accuracy cao nhất để đánh giá trên tập Test ---")
    
    # Truy vấn kết quả từ MLflow
    experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.cv_accuracy DESC"])
    best_run = runs.iloc[0] # Lấy dòng đầu tiên (Accuracy cao nhất)
    
    best_run_id = best_run.run_id
    best_model_name = best_run["tags.mlflow.runName"]
    
    print(f"Model tốt nhất là: {best_model_name} (ID: {best_run_id})")

    # Load model tốt nhất về để test
    best_pipeline = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model_file")
    
    # Dự đoán trên tập Test sạch
    y_pred = best_pipeline.predict(X_test)

    # Đánh giá chi tiết (Classification Report cho từng class Churned 0 và 1)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Tạo một Run mới đặc biệt để lưu kết quả Test cuối cùng
    with mlflow.start_run(run_name=f"FINAL_TEST_EVALUATION"):
        mlflow.log_param("original_best_run_id", best_run_id)
        mlflow.log_param("best_model_architecture", best_model_name)
        
        # Log Metrics chi tiết cho từng trường hợp Churned (0 và 1)
        # Class 1 thường là Churned (Khách rời đi)
        mlflow.log_metrics({
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1_class_1": report['1']['f1-score'],
            "test_recall_class_1": report['1']['recall'],
            "test_precision_class_1": report['1']['precision'],
            "test_f1_class_0": report['0']['f1-score']
        })
        
        print(f"--- KẾT QUẢ TRÊN TẬP TEST SẠCH ---")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = "minioadmin"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "minioadmin123"
    mlflow.set_tracking_uri("http://localhost:5000") 
    
    train()