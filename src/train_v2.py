import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline 
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (classification_report, accuracy_score, f1_score, 
                             recall_score, precision_score, make_scorer)
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

    # 1. Đọc dữ liệu
    train_df = pd.read_csv(config['processed_data']['train_path'])
    test_df = pd.read_csv(config['processed_data']['test_path'])
    target = config['preprocess']['target_column']

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # 2. Định nghĩa bộ đo lường chi tiết cho cả Class 0 và Class 1
    # pos_label=1 cho Churned, pos_label=0 cho Stayed
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
                    
                    # Chạy CV với bộ scoring_metrics đã định nghĩa
                    cv_results = cross_validate(
                        pipeline, X_train, y_train, 
                        cv=config['preprocess']['cv_folds'],
                        scoring=scoring_metrics,
                        n_jobs=-1 
                    )

                    # Log Params
                    mlflow.log_params({
                        "model_type": m_name, 
                        "scaler_type": s_name, 
                        "sampling_type": sampling
                    })
                    
                    # Log đầy đủ metrics trung bình của CV cho cả 2 Class
                    mlflow.log_metrics({
                        "cv_accuracy": cv_results['test_accuracy'].mean(),
                        "cv_f1_churned": cv_results['test_f1_c1'].mean(),
                        "cv_f1_stayed": cv_results['test_f1_c0'].mean(),
                        "cv_recall_churned": cv_results['test_recall_c1'].mean(),
                        "cv_recall_stayed": cv_results['test_recall_c0'].mean(),
                        "cv_precision_churned": cv_results['test_precision_c1'].mean(),
                        "cv_precision_stayed": cv_results['test_precision_c0'].mean()
                    })

                    # Huấn luyện trên toàn bộ tập Train để lưu model
                    pipeline.fit(X_train, y_train)
                    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model_file")
                    print(f"Done CV: {run_name} | Acc: {cv_results['test_accuracy'].mean():.4f}")

    # --- BƯỚC CUỐI: ĐÁNH GIÁ MODEL TỐT NHẤT TRÊN TẬP TEST SẠCH ---
    print("\n--- Tìm model có Accuracy cao nhất để test cuối cùng ---")
    
    experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.cv_accuracy DESC"])
    best_run = runs.iloc[0]
    
    best_run_id = best_run.run_id
    best_model_name = best_run["tags.mlflow.runName"]
    
    print(f"Model vô địch: {best_model_name}")

    best_pipeline = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model_file")
    y_pred = best_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    with mlflow.start_run(run_name="FINAL_TEST_EVALUATION"):
        mlflow.log_param("best_model_architecture", best_model_name)
        mlflow.log_metrics({
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1_churned": report['1']['f1-score'],
            "test_recall_churned": report['1']['recall'],
            "test_precision_churned": report['1']['precision'],
            "test_f1_stayed": report['0']['f1-score'],
            "test_recall_stayed": report['0']['recall'],
            "test_precision_stayed": report['0']['precision']
        })
        print(f"--- KẾT QUẢ CUỐI CÙNG ---")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = "minioadmin"
    os.environ['AWS_SECRET_ACCESS_KEY'] = "minioadmin123"
    mlflow.set_tracking_uri("http://localhost:5000") 
    train()