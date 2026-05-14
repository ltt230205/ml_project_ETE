# import os
# import time
# from typing import Optional

# import mlflow
# import mlflow.sklearn
# import pandas as pd
# import streamlit as st
# from mlflow.tracking import MlflowClient


# # =========================
# # MLflow / MinIO config
# # =========================

# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

# os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
# os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
# os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")
# os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# client = MlflowClient()


# # =========================
# # MLflow helpers
# # =========================

# def list_model_versions(model_name: str):
#     versions = list(client.search_model_versions(f"name='{model_name}'"))
#     versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
#     return versions


# def get_latest_model_version(model_name: str):
#     versions = list_model_versions(model_name)

#     if not versions:
#         return None

#     return versions[0]


# def get_production_model_version(model_name: str):
#     versions = list_model_versions(model_name)

#     for version in versions:
#         if getattr(version, "current_stage", None) == "Production":
#             return version

#     return None


# def get_threshold_from_model_version(model_version, default: float = 0.35) -> float:
#     if model_version is None:
#         return default

#     if model_version.tags and "decision_threshold" in model_version.tags:
#         return float(model_version.tags["decision_threshold"])

#     if model_version.run_id:
#         run = client.get_run(model_version.run_id)

#         if "decision_threshold" in run.data.params:
#             return float(run.data.params["decision_threshold"])

#     return default


# @st.cache_resource
# def load_model(model_name: str, version: str):
#     model_uri = f"models:/{model_name}/{version}"

#     start = time.perf_counter()
#     model = mlflow.sklearn.load_model(model_uri)
#     end = time.perf_counter()

#     load_latency_ms = (end - start) * 1000

#     return model, model_uri, load_latency_ms


# def get_expected_columns(model) -> list[str]:
#     """
#     Lấy danh sách input columns mà model pipeline mong đợi.

#     Ưu tiên:
#     1. model.feature_names_in_
#     2. model.steps[0][1].feature_names_in_
#     3. ColumnTransformer columns
#     """

#     if hasattr(model, "feature_names_in_"):
#         return list(model.feature_names_in_)

#     if hasattr(model, "steps"):
#         for _, step in model.steps:
#             if hasattr(step, "feature_names_in_"):
#                 return list(step.feature_names_in_)

#     try:
#         preprocessor = model.named_steps.get("preprocess")

#         cols = []
#         for _, _, col_list in preprocessor.transformers_:
#             if isinstance(col_list, list):
#                 cols.extend(col_list)

#         if cols:
#             return cols
#     except Exception:
#         pass

#     return []


# def align_input_columns(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
#     if not expected_columns:
#         return df

#     missing_cols = [col for col in expected_columns if col not in df.columns]

#     if missing_cols:
#         raise ValueError(
#             "Input thiếu các cột mà model cần: "
#             + ", ".join(missing_cols)
#         )

#     return df[expected_columns].copy()


# def predict_with_threshold(model, input_df: pd.DataFrame, threshold: float):
#     start = time.perf_counter()

#     if hasattr(model, "predict_proba"):
#         proba = model.predict_proba(input_df)
#         churn_prob = float(proba[0][1])
#         stay_prob = float(proba[0][0])
#         prediction = 1 if churn_prob >= threshold else 0
#     else:
#         prediction = int(model.predict(input_df)[0])
#         churn_prob = None
#         stay_prob = None

#     end = time.perf_counter()
#     latency_ms = (end - start) * 1000

#     return prediction, stay_prob, churn_prob, latency_ms


# def predict_batch_with_threshold(model, input_df: pd.DataFrame, threshold: float):
#     start = time.perf_counter()

#     result_df = input_df.copy()

#     if hasattr(model, "predict_proba"):
#         proba = model.predict_proba(input_df)

#         result_df["stay_probability"] = proba[:, 0]
#         result_df["churn_probability"] = proba[:, 1]
#         result_df["prediction"] = (result_df["churn_probability"] >= threshold).astype(int)
#     else:
#         result_df["prediction"] = model.predict(input_df)

#     result_df["prediction_label"] = result_df["prediction"].map({
#         0: "STAY",
#         1: "CHURN",
#     })

#     end = time.perf_counter()
#     latency_ms = (end - start) * 1000

#     return result_df, latency_ms


# # =========================
# # Streamlit UI
# # =========================

# st.set_page_config(
#     page_title="Customer Churn Prediction",
#     page_icon="📊",
#     layout="wide",
# )

# st.title("📊 Customer Churn Prediction")
# st.caption("Streamlit app dùng MLflow Model Registry, threshold tuning và batch prediction.")


# with st.sidebar:
#     st.header("⚙️ Model Config")

#     st.write("MLflow Tracking URI")
#     st.code(MLFLOW_TRACKING_URI)

#     st.write("MinIO S3 Endpoint")
#     st.code(MLFLOW_S3_ENDPOINT_URL)

#     model_name = st.text_input(
#         "Registered Model Name",
#         value=os.getenv("MODEL_NAME", "churn_model_staged_best"),
#     )

#     version_mode = st.radio(
#         "Chọn model version",
#         ["Production", "Latest", "Manual"],
#         index=1,
#     )

#     selected_model_version = None

#     if version_mode == "Production":
#         selected_model_version = get_production_model_version(model_name)

#         if selected_model_version is None:
#             st.warning("Không tìm thấy Production version. Tự động dùng Latest.")
#             selected_model_version = get_latest_model_version(model_name)

#     elif version_mode == "Latest":
#         selected_model_version = get_latest_model_version(model_name)

#     else:
#         manual_version = st.text_input("Nhập version", value="1")

#         try:
#             selected_model_version = client.get_model_version(
#                 name=model_name,
#                 version=manual_version,
#             )
#         except Exception as e:
#             st.error(f"Không tìm thấy model version: {e}")
#             st.stop()

#     if selected_model_version is None:
#         st.error(f"Không tìm thấy model nào với tên `{model_name}`")
#         st.stop()

#     model_version = selected_model_version.version
#     default_threshold = get_threshold_from_model_version(
#         selected_model_version,
#         default=0.35,
#     )

#     threshold = st.slider(
#         "Decision threshold",
#         min_value=0.0,
#         max_value=1.0,
#         value=float(default_threshold),
#         step=0.01,
#     )

#     st.info(f"Model version: {model_version}")
#     st.info(f"Default threshold từ MLflow: {default_threshold}")

#     if st.button("Reload model"):
#         load_model.clear()


# try:
#     model, model_uri, load_latency_ms = load_model(model_name, model_version)
# except Exception as e:
#     st.error(f"Không load được model từ MLflow: {e}")
#     st.stop()


# expected_columns = get_expected_columns(model)

# st.success(f"✅ Loaded model: `{model_uri}`")
# st.caption(f"Load model latency: `{load_latency_ms:.2f} ms`")

# if expected_columns:
#     with st.expander("Xem input columns model cần"):
#         st.write(expected_columns)


# tab_single, tab_batch, tab_info = st.tabs([
#     "👤 Single Prediction",
#     "📁 Batch Prediction",
#     "ℹ️ Model Info",
# ])


# # =========================
# # Single prediction
# # =========================

# with tab_single:
#     st.subheader("Nhập thông tin khách hàng")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         age = st.number_input("Age", value=43.0)
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         country = st.text_input("Country", value="France")
#         city = st.text_input("City", value="Marseille")
#         membership_years = st.number_input("Membership_Years", value=2.9)
#         login_frequency = st.number_input("Login_Frequency", value=14.0)
#         session_duration_avg = st.number_input("Session_Duration_Avg", value=27.4)
#         pages_per_session = st.number_input("Pages_Per_Session", value=6.0)

#     with col2:
#         cart_abandonment_rate = st.number_input("Cart_Abandonment_Rate", value=50.6)
#         wishlist_items = st.number_input("Wishlist_Items", value=3.0)
#         total_purchases = st.number_input("Total_Purchases", value=9.0)
#         average_order_value = st.number_input("Average_Order_Value", value=94.72)
#         days_since_last_purchase = st.number_input("Days_Since_Last_Purchase", value=34.0)
#         discount_usage_rate = st.number_input("Discount_Usage_Rate", value=46.4)
#         returns_rate = st.number_input("Returns_Rate", value=2.0)
#         email_open_rate = st.number_input("Email_Open_Rate", value=17.9)

#     with col3:
#         customer_service_calls = st.number_input("Customer_Service_Calls", value=9.0)
#         product_reviews_written = st.number_input("Product_Reviews_Written", value=4.0)
#         social_media_score = st.number_input("Social_Media_Engagement_Score", value=16.3)
#         mobile_app_usage = st.number_input("Mobile_App_Usage", value=20.8)
#         payment_method_diversity = st.number_input("Payment_Method_Diversity", value=1.0)
#         lifetime_value = st.number_input("Lifetime_Value", value=953.33)
#         credit_balance = st.number_input("Credit_Balance", value=2278.0)
#         signup_quarter = st.selectbox("Signup_Quarter", ["Q1", "Q2", "Q3", "Q4"])

#     input_data = {
#         "Age": age,
#         "Gender": gender,
#         "Country": country,
#         "City": city,
#         "Membership_Years": membership_years,
#         "Login_Frequency": login_frequency,
#         "Session_Duration_Avg": session_duration_avg,
#         "Pages_Per_Session": pages_per_session,
#         "Cart_Abandonment_Rate": cart_abandonment_rate,
#         "Wishlist_Items": wishlist_items,
#         "Total_Purchases": total_purchases,
#         "Average_Order_Value": average_order_value,
#         "Days_Since_Last_Purchase": days_since_last_purchase,
#         "Discount_Usage_Rate": discount_usage_rate,
#         "Returns_Rate": returns_rate,
#         "Email_Open_Rate": email_open_rate,
#         "Customer_Service_Calls": customer_service_calls,
#         "Product_Reviews_Written": product_reviews_written,
#         "Social_Media_Engagement_Score": social_media_score,
#         "Mobile_App_Usage": mobile_app_usage,
#         "Payment_Method_Diversity": payment_method_diversity,
#         "Lifetime_Value": lifetime_value,
#         "Credit_Balance": credit_balance,
#         "Signup_Quarter": signup_quarter,
#     }

#     input_df = pd.DataFrame([input_data])

#     try:
#         aligned_input_df = align_input_columns(input_df, expected_columns)
#     except Exception as e:
#         st.error(str(e))
#         st.stop()

#     st.write("Input data dùng cho model")
#     st.dataframe(aligned_input_df, use_container_width=True)

#     if st.button("🚀 Predict single"):
#         try:
#             prediction, stay_prob, churn_prob, latency_ms = predict_with_threshold(
#                 model=model,
#                 input_df=aligned_input_df,
#                 threshold=threshold,
#             )

#             st.divider()

#             if prediction == 1:
#                 st.error("Kết quả: CHURN")
#             else:
#                 st.success("Kết quả: STAY")

#             c1, c2, c3 = st.columns(3)

#             c1.metric("Threshold", f"{threshold:.2f}")
#             c2.metric("Predict latency", f"{latency_ms:.2f} ms")

#             if churn_prob is not None:
#                 c3.metric("Churn probability", f"{churn_prob:.2%}")

#                 c4, c5 = st.columns(2)
#                 c4.metric("Xác suất ở lại", f"{stay_prob:.2%}")
#                 c5.metric("Xác suất rời bỏ", f"{churn_prob:.2%}")

#                 st.progress(churn_prob)
#             else:
#                 c3.metric("Probability", "N/A")

#         except Exception as e:
#             st.error(f"Lỗi khi predict: {e}")


# # =========================
# # Batch prediction
# # =========================

# with tab_batch:
#     st.subheader("Upload CSV để predict hàng loạt")

#     uploaded_file = st.file_uploader(
#         "Upload file CSV",
#         type=["csv"],
#     )

#     if uploaded_file is not None:
#         batch_df = pd.read_csv(uploaded_file)

#         st.write("Preview input")
#         st.dataframe(batch_df.head(20), use_container_width=True)

#         try:
#             aligned_batch_df = align_input_columns(batch_df, expected_columns)
#         except Exception as e:
#             st.error(str(e))
#             st.stop()

#         st.write("Preview data sau khi align columns")
#         st.dataframe(aligned_batch_df.head(20), use_container_width=True)

#         if st.button("🚀 Predict batch"):
#             try:
#                 result_df, batch_latency_ms = predict_batch_with_threshold(
#                     model=model,
#                     input_df=aligned_batch_df,
#                     threshold=threshold,
#                 )

#                 st.success("✅ Predict batch thành công")

#                 c1, c2, c3 = st.columns(3)
#                 c1.metric("Rows", len(result_df))
#                 c2.metric("Batch latency", f"{batch_latency_ms:.2f} ms")
#                 c3.metric(
#                     "Latency / row",
#                     f"{batch_latency_ms / max(len(result_df), 1):.4f} ms",
#                 )

#                 st.dataframe(result_df.head(100), use_container_width=True)

#                 csv_bytes = result_df.to_csv(index=False).encode("utf-8")

#                 st.download_button(
#                     label="⬇️ Download prediction result CSV",
#                     data=csv_bytes,
#                     file_name="churn_prediction_result.csv",
#                     mime="text/csv",
#                 )

#             except Exception as e:
#                 st.error(f"Lỗi khi predict batch: {e}")


# # =========================
# # Model info
# # =========================

# with tab_info:
#     st.subheader("Thông tin model")

#     info_df = pd.DataFrame(
#         [
#             {
#                 "model_name": model_name,
#                 "version": model_version,
#                 "model_uri": model_uri,
#                 "run_id": selected_model_version.run_id,
#                 "stage": getattr(selected_model_version, "current_stage", None),
#                 "default_threshold": default_threshold,
#                 "current_threshold": threshold,
#                 "load_latency_ms": load_latency_ms,
#             }
#         ]
#     )

#     st.dataframe(info_df, use_container_width=True)

#     st.subheader("Model version tags")

#     if selected_model_version.tags:
#         tags_df = pd.DataFrame(
#             [
#                 {"key": key, "value": value}
#                 for key, value in selected_model_version.tags.items()
#             ]
#         )
#         st.dataframe(tags_df, use_container_width=True)
#     else:
#         st.info("Model version không có tags.")

#     st.subheader("Expected input columns")

#     if expected_columns:
#         st.dataframe(
#             pd.DataFrame({"column": expected_columns}),
#             use_container_width=True,
#         )
#     else:
#         st.warning("Không lấy được expected columns từ model.")
import json
import os
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient


# =========================================================
# Config
# =========================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


# =========================================================
# Default values
# =========================================================

FALLBACK_DEFAULT_VALUES = {
    "Age": 35.0,
    "Gender": "Male",
    "Country": "France",
    "City": "Paris",
    "Membership_Years": 3.0,
    "Login_Frequency": 10.0,
    "Session_Duration_Avg": 20.0,
    "Pages_Per_Session": 5.0,
    "Cart_Abandonment_Rate": 30.0,
    "Wishlist_Items": 2.0,
    "Total_Purchases": 5.0,
    "Average_Order_Value": 80.0,
    "Days_Since_Last_Purchase": 15.0,
    "Discount_Usage_Rate": 20.0,
    "Returns_Rate": 1.0,
    "Email_Open_Rate": 30.0,
    "Customer_Service_Calls": 2.0,
    "Product_Reviews_Written": 1.0,
    "Social_Media_Engagement_Score": 20.0,
    "Mobile_App_Usage": 20.0,
    "Payment_Method_Diversity": 1.0,
    "Lifetime_Value": 1000.0,
    "Credit_Balance": 1000.0,
    "Signup_Quarter": "Q1",
}


MAIN_INPUT_FEATURES = [
    "Customer_Service_Calls",
    "Cart_Abandonment_Rate",
    "Pages_Per_Session",
    "Session_Duration_Avg",
    "Email_Open_Rate",
    "Mobile_App_Usage",
    "Login_Frequency",
    "Wishlist_Items",
    "Total_Purchases",
    "Days_Since_Last_Purchase",
]


ADVANCED_INPUT_FEATURES = [
    "Age",
    "Gender",
    "Country",
    "City",
    "Membership_Years",
    "Average_Order_Value",
    "Discount_Usage_Rate",
    "Returns_Rate",
    "Product_Reviews_Written",
    "Social_Media_Engagement_Score",
    "Payment_Method_Diversity",
    "Lifetime_Value",
    "Credit_Balance",
    "Signup_Quarter",
]


NUMERIC_FEATURES = [
    "Age",
    "Membership_Years",
    "Login_Frequency",
    "Session_Duration_Avg",
    "Pages_Per_Session",
    "Cart_Abandonment_Rate",
    "Wishlist_Items",
    "Total_Purchases",
    "Average_Order_Value",
    "Days_Since_Last_Purchase",
    "Discount_Usage_Rate",
    "Returns_Rate",
    "Email_Open_Rate",
    "Customer_Service_Calls",
    "Product_Reviews_Written",
    "Social_Media_Engagement_Score",
    "Mobile_App_Usage",
    "Payment_Method_Diversity",
    "Lifetime_Value",
    "Credit_Balance",
]


CATEGORICAL_FEATURES = [
    "Gender",
    "Country",
    "City",
    "Signup_Quarter",
]


# =========================================================
# MLflow helpers
# =========================================================

def list_model_versions(model_name: str):
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
    return versions


def get_latest_model_version(model_name: str):
    versions = list_model_versions(model_name)

    if not versions:
        return None

    return versions[0]


def get_production_model_version(model_name: str):
    versions = list_model_versions(model_name)

    for version in versions:
        if getattr(version, "current_stage", None) == "Production":
            return version

    return None


def get_threshold_from_model_version(model_version, default: float = 0.35) -> float:
    if model_version is None:
        return default

    if model_version.tags and "decision_threshold" in model_version.tags:
        return float(model_version.tags["decision_threshold"])

    if model_version.run_id:
        run = client.get_run(model_version.run_id)

        if "decision_threshold" in run.data.params:
            return float(run.data.params["decision_threshold"])

    return default


@st.cache_resource
def load_model(model_name: str, version: str):
    model_uri = f"models:/{model_name}/{version}"

    start = time.perf_counter()
    model = mlflow.sklearn.load_model(model_uri)
    end = time.perf_counter()

    load_latency_ms = (end - start) * 1000

    return model, model_uri, load_latency_ms


def get_expected_columns(model) -> list[str]:
    """
    Lấy danh sách input columns mà model pipeline cần.

    Ưu tiên:
    1. model.feature_names_in_
    2. step.feature_names_in_ trong pipeline
    3. ColumnTransformer transformers_
    """

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "steps"):
        for _, step in model.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    try:
        preprocessor = model.named_steps.get("preprocess")

        cols = []

        for _, _, col_list in preprocessor.transformers_:
            if isinstance(col_list, list):
                cols.extend(col_list)

        if cols:
            return cols

    except Exception:
        pass

    return []


# =========================================================
# Default values helpers
# =========================================================

def load_default_values_from_cleaning_stats(
    stats_path: str = "artifacts/staged_pipeline/cleaning_stats.json",
) -> dict:
    """
    Lấy default từ cleaning_stats.json.

    Numeric:
        median từ train

    Categorical:
        mode/fill_value từ train

    Nếu không có file cleaning_stats.json thì dùng fallback.
    """

    defaults = FALLBACK_DEFAULT_VALUES.copy()

    if not os.path.exists(stats_path):
        return defaults

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        for col, col_stats in stats.get("numeric", {}).items():
            if "median" in col_stats:
                defaults[col] = float(col_stats["median"])

        for col, col_stats in stats.get("categorical", {}).items():
            if "fill_value" in col_stats:
                defaults[col] = str(col_stats["fill_value"])

        return defaults

    except Exception:
        return defaults


def complete_input_df(
    input_df: pd.DataFrame,
    default_values: dict,
    expected_columns: list[str],
    target_column: str = "Churned",
) -> pd.DataFrame:
    """
    Đảm bảo input DataFrame có đủ cột model cần.

    Nếu thiếu cột nào:
        lấy default value.

    Nếu thừa cột target:
        drop target.

    Sau đó align đúng thứ tự expected_columns.
    """

    df = input_df.copy()

    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    required_columns = expected_columns if expected_columns else list(default_values.keys())

    for col in required_columns:
        if col not in df.columns:
            df[col] = default_values.get(col, None)

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(float(default_values.get(col, 0.0)))
            df[col] = df[col].astype("float64")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(str(default_values.get(col, "unknown")))
            df[col] = df[col].astype(str)

    return df[required_columns].copy()


def build_single_input_df(
    user_values: dict,
    default_values: dict,
    expected_columns: list[str],
) -> pd.DataFrame:
    full_input = default_values.copy()
    full_input.update(user_values)

    df = pd.DataFrame([full_input])

    return complete_input_df(
        input_df=df,
        default_values=default_values,
        expected_columns=expected_columns,
    )


# =========================================================
# Prediction helpers
# =========================================================

def predict_with_threshold(
    model,
    input_df: pd.DataFrame,
    threshold: float,
):
    start = time.perf_counter()

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)
        stay_prob = float(proba[0][0])
        churn_prob = float(proba[0][1])
        prediction = 1 if churn_prob >= threshold else 0
    else:
        prediction = int(model.predict(input_df)[0])
        stay_prob = None
        churn_prob = None

    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    return prediction, stay_prob, churn_prob, latency_ms


def predict_batch_with_threshold(
    model,
    input_df: pd.DataFrame,
    threshold: float,
):
    start = time.perf_counter()

    result_df = input_df.copy()

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)

        result_df["stay_probability"] = proba[:, 0]
        result_df["churn_probability"] = proba[:, 1]
        result_df["prediction"] = (
            result_df["churn_probability"] >= threshold
        ).astype(int)
    else:
        result_df["prediction"] = model.predict(input_df)

    result_df["prediction_label"] = result_df["prediction"].map(
        {
            0: "STAY",
            1: "CHURN",
        }
    )

    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    return result_df, latency_ms


# =========================================================
# UI input render helpers
# =========================================================

def render_main_inputs(default_values: dict) -> dict:
    st.markdown("### Các thông tin chính cần nhập")

    col1, col2 = st.columns(2)

    values = {}

    with col1:
        values["Customer_Service_Calls"] = st.number_input(
            "Customer_Service_Calls",
            value=float(default_values.get("Customer_Service_Calls", 2.0)),
        )

        values["Cart_Abandonment_Rate"] = st.number_input(
            "Cart_Abandonment_Rate",
            value=float(default_values.get("Cart_Abandonment_Rate", 30.0)),
        )

        values["Pages_Per_Session"] = st.number_input(
            "Pages_Per_Session",
            value=float(default_values.get("Pages_Per_Session", 5.0)),
        )

        values["Session_Duration_Avg"] = st.number_input(
            "Session_Duration_Avg",
            value=float(default_values.get("Session_Duration_Avg", 20.0)),
        )

        values["Email_Open_Rate"] = st.number_input(
            "Email_Open_Rate",
            value=float(default_values.get("Email_Open_Rate", 30.0)),
        )

    with col2:
        values["Mobile_App_Usage"] = st.number_input(
            "Mobile_App_Usage",
            value=float(default_values.get("Mobile_App_Usage", 20.0)),
        )

        values["Login_Frequency"] = st.number_input(
            "Login_Frequency",
            value=float(default_values.get("Login_Frequency", 10.0)),
        )

        values["Wishlist_Items"] = st.number_input(
            "Wishlist_Items",
            value=float(default_values.get("Wishlist_Items", 2.0)),
        )

        values["Total_Purchases"] = st.number_input(
            "Total_Purchases",
            value=float(default_values.get("Total_Purchases", 5.0)),
        )

        values["Days_Since_Last_Purchase"] = st.number_input(
            "Days_Since_Last_Purchase",
            value=float(default_values.get("Days_Since_Last_Purchase", 15.0)),
        )

    return values


def render_advanced_inputs(default_values: dict) -> dict:
    values = {}

    with st.expander("Tuỳ chỉnh thêm các cột còn lại", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            values["Age"] = st.number_input(
                "Age",
                value=float(default_values.get("Age", 35.0)),
            )

            gender_default = str(default_values.get("Gender", "Male"))
            gender_options = ["Male", "Female"]
            gender_index = (
                gender_options.index(gender_default)
                if gender_default in gender_options
                else 0
            )

            values["Gender"] = st.selectbox(
                "Gender",
                gender_options,
                index=gender_index,
            )

            values["Country"] = st.text_input(
                "Country",
                value=str(default_values.get("Country", "France")),
            )

            values["City"] = st.text_input(
                "City",
                value=str(default_values.get("City", "Paris")),
            )

            values["Membership_Years"] = st.number_input(
                "Membership_Years",
                value=float(default_values.get("Membership_Years", 3.0)),
            )

        with col2:
            values["Average_Order_Value"] = st.number_input(
                "Average_Order_Value",
                value=float(default_values.get("Average_Order_Value", 80.0)),
            )

            values["Discount_Usage_Rate"] = st.number_input(
                "Discount_Usage_Rate",
                value=float(default_values.get("Discount_Usage_Rate", 20.0)),
            )

            values["Returns_Rate"] = st.number_input(
                "Returns_Rate",
                value=float(default_values.get("Returns_Rate", 1.0)),
            )

            values["Product_Reviews_Written"] = st.number_input(
                "Product_Reviews_Written",
                value=float(default_values.get("Product_Reviews_Written", 1.0)),
            )

            values["Social_Media_Engagement_Score"] = st.number_input(
                "Social_Media_Engagement_Score",
                value=float(default_values.get("Social_Media_Engagement_Score", 20.0)),
            )

        with col3:
            values["Payment_Method_Diversity"] = st.number_input(
                "Payment_Method_Diversity",
                value=float(default_values.get("Payment_Method_Diversity", 1.0)),
            )

            values["Lifetime_Value"] = st.number_input(
                "Lifetime_Value",
                value=float(default_values.get("Lifetime_Value", 1000.0)),
            )

            values["Credit_Balance"] = st.number_input(
                "Credit_Balance",
                value=float(default_values.get("Credit_Balance", 1000.0)),
            )

            signup_default = str(default_values.get("Signup_Quarter", "Q1"))
            signup_options = ["Q1", "Q2", "Q3", "Q4"]
            signup_index = (
                signup_options.index(signup_default)
                if signup_default in signup_options
                else 0
            )

            values["Signup_Quarter"] = st.selectbox(
                "Signup_Quarter",
                signup_options,
                index=signup_index,
            )

    return values


# =========================================================
# Streamlit app
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction")
st.caption(
    "Streamlit app dùng MLflow Model Registry. "
    "Người dùng chỉ cần nhập một số feature chính, các feature còn lại lấy default từ train data."
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.header("⚙️ Model Config")

    st.write("MLflow Tracking URI")
    st.code(MLFLOW_TRACKING_URI)

    st.write("MinIO S3 Endpoint")
    st.code(MLFLOW_S3_ENDPOINT_URL)

    model_name = st.text_input(
        "Registered Model Name",
        value=os.getenv("MODEL_NAME", "churn_model_staged_best"),
    )

    version_mode = st.radio(
        "Chọn model version",
        ["Production", "Latest", "Manual"],
        index=1,
    )

    selected_model_version = None

    if version_mode == "Production":
        selected_model_version = get_production_model_version(model_name)

        if selected_model_version is None:
            st.warning("Không tìm thấy Production version. Tự động dùng Latest.")
            selected_model_version = get_latest_model_version(model_name)

    elif version_mode == "Latest":
        selected_model_version = get_latest_model_version(model_name)

    else:
        manual_version = st.text_input("Nhập version", value="1")

        try:
            selected_model_version = client.get_model_version(
                name=model_name,
                version=manual_version,
            )
        except Exception as e:
            st.error(f"Không tìm thấy model version: {e}")
            st.stop()

    if selected_model_version is None:
        st.error(f"Không tìm thấy model nào với tên `{model_name}`")
        st.stop()

    model_version = selected_model_version.version

    default_threshold = get_threshold_from_model_version(
        selected_model_version,
        default=0.35,
    )

    threshold_mode = st.radio(
        "Threshold mode",
        ["From MLflow", "0.5", "0.35", "Custom"],
        index=0,
    )

    if threshold_mode == "From MLflow":
        threshold = float(default_threshold)
    elif threshold_mode == "0.5":
        threshold = 0.5
    elif threshold_mode == "0.35":
        threshold = 0.35
    else:
        threshold = st.slider(
            "Custom threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(default_threshold),
            step=0.01,
        )

    st.info(f"Model version: {model_version}")
    st.info(f"Current threshold: {threshold}")

    if st.button("Reload model"):
        load_model.clear()


# =========================================================
# Load model
# =========================================================

try:
    model, model_uri, load_latency_ms = load_model(model_name, model_version)
except Exception as e:
    st.error(f"Không load được model từ MLflow: {e}")
    st.stop()


expected_columns = get_expected_columns(model)

default_values = load_default_values_from_cleaning_stats(
    "artifacts/staged_pipeline/cleaning_stats.json"
)

st.success(f"✅ Loaded model: `{model_uri}`")
st.caption(f"Load model latency: `{load_latency_ms:.2f} ms`")

if expected_columns:
    with st.expander("Xem input columns model cần", expanded=False):
        st.write(expected_columns)


tab_single, tab_batch, tab_info = st.tabs(
    [
        "👤 Single Prediction",
        "📁 Batch Prediction",
        "ℹ️ Model Info",
    ]
)


# =========================================================
# Single Prediction
# =========================================================

with tab_single:
    st.subheader("Nhập thông tin khách hàng")

    st.info(
        "Bạn chỉ cần nhập các thông tin chính. "
        "Các cột còn lại sẽ lấy default từ dữ liệu train. "
        "Nếu muốn chỉnh, mở phần 'Tuỳ chỉnh thêm các cột còn lại'."
    )

    main_values = render_main_inputs(default_values)
    advanced_values = render_advanced_inputs(default_values)

    user_values = {
        **main_values,
        **advanced_values,
    }

    input_df = build_single_input_df(
        user_values=user_values,
        default_values=default_values,
        expected_columns=expected_columns,
    )

    st.write("Input đầy đủ sau khi ghép default values")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🚀 Predict single"):
        try:
            prediction, stay_prob, churn_prob, latency_ms = predict_with_threshold(
                model=model,
                input_df=input_df,
                threshold=threshold,
            )

            st.divider()

            if prediction == 1:
                st.error("Kết quả: CHURN")
            else:
                st.success("Kết quả: STAY")

            c1, c2, c3 = st.columns(3)

            c1.metric("Threshold", f"{threshold:.2f}")
            c2.metric("Predict latency", f"{latency_ms:.2f} ms")

            if churn_prob is not None:
                c3.metric("Churn probability", f"{churn_prob:.2%}")

                c4, c5 = st.columns(2)
                c4.metric("Xác suất ở lại", f"{stay_prob:.2%}")
                c5.metric("Xác suất rời bỏ", f"{churn_prob:.2%}")

                st.progress(churn_prob)
            else:
                c3.metric("Probability", "N/A")

        except Exception as e:
            st.error(f"Lỗi khi predict: {e}")


# =========================================================
# Batch Prediction
# =========================================================

with tab_batch:
    st.subheader("Upload CSV để predict hàng loạt")

    st.info(
        "CSV không bắt buộc phải có đủ toàn bộ cột. "
        "Cột nào thiếu sẽ được tự động fill bằng default từ train data."
    )

    uploaded_file = st.file_uploader(
        "Upload file CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        raw_batch_df = pd.read_csv(uploaded_file)

        st.write("Preview input gốc")
        st.dataframe(raw_batch_df.head(20), use_container_width=True)

        try:
            batch_df = complete_input_df(
                input_df=raw_batch_df,
                default_values=default_values,
                expected_columns=expected_columns,
            )
        except Exception as e:
            st.error(f"Lỗi khi chuẩn hoá input batch: {e}")
            st.stop()

        st.write("Preview input sau khi fill default và align columns")
        st.dataframe(batch_df.head(20), use_container_width=True)

        if st.button("🚀 Predict batch"):
            try:
                result_df, batch_latency_ms = predict_batch_with_threshold(
                    model=model,
                    input_df=batch_df,
                    threshold=threshold,
                )

                output_df = raw_batch_df.copy()

                for col in result_df.columns:
                    if col not in output_df.columns:
                        output_df[col] = result_df[col].values

                st.success("✅ Predict batch thành công")

                c1, c2, c3 = st.columns(3)
                c1.metric("Rows", len(result_df))
                c2.metric("Batch latency", f"{batch_latency_ms:.2f} ms")
                c3.metric(
                    "Latency / row",
                    f"{batch_latency_ms / max(len(result_df), 1):.4f} ms",
                )

                st.dataframe(output_df.head(100), use_container_width=True)

                csv_bytes = output_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download prediction result CSV",
                    data=csv_bytes,
                    file_name="churn_prediction_result.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Lỗi khi predict batch: {e}")


# =========================================================
# Model Info
# =========================================================

with tab_info:
    st.subheader("Thông tin model")

    info_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "version": model_version,
                "model_uri": model_uri,
                "run_id": selected_model_version.run_id,
                "stage": getattr(selected_model_version, "current_stage", None),
                "default_threshold_from_mlflow": default_threshold,
                "current_threshold": threshold,
                "load_latency_ms": load_latency_ms,
            }
        ]
    )

    st.dataframe(info_df, use_container_width=True)

    st.subheader("Model version tags")

    if selected_model_version.tags:
        tags_df = pd.DataFrame(
            [
                {
                    "key": key,
                    "value": value,
                }
                for key, value in selected_model_version.tags.items()
            ]
        )
        st.dataframe(tags_df, use_container_width=True)
    else:
        st.info("Model version không có tags.")

    st.subheader("Expected input columns")

    if expected_columns:
        st.dataframe(
            pd.DataFrame({"column": expected_columns}),
            use_container_width=True,
        )
    else:
        st.warning("Không lấy được expected columns từ model.")

    st.subheader("Default values đang dùng")

    default_df = pd.DataFrame(
        [
            {
                "feature": key,
                "default_value": value,
            }
            for key, value in default_values.items()
        ]
    )

    st.dataframe(default_df, use_container_width=True)