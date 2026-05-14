import os
from typing import Optional

import mlflow
import mlflow.sklearn
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient


# =========================
# MLflow config for LOCAL HOST
# =========================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def get_latest_model_version(model_name: str) -> Optional[str]:
    versions = list(client.search_model_versions(f"name='{model_name}'"))

    if not versions:
        return None

    latest_version = max(versions, key=lambda v: int(v.version))
    return latest_version.version


@st.cache_resource
def load_model(model_name: str, model_version: str):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model, model_uri


def build_input_df(values: dict) -> pd.DataFrame:
    df = pd.DataFrame([values])

    numeric_cols = [
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

    for col in numeric_cols:
        df[col] = df[col].astype("float64")

    return df


def predict(model, input_df: pd.DataFrame):
    pred = model.predict(input_df)

    stay_prob = None
    churn_prob = None

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)
        stay_prob = float(prob[0][0])
        churn_prob = float(prob[0][1])

    return int(pred[0]), stay_prob, churn_prob


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction")
st.caption("Ứng dụng Streamlit local dùng MLflow Model Registry để dự đoán churn.")

with st.sidebar:
    st.header("⚙️ Model Config")

    st.write("MLflow Tracking URI")
    st.code(MLFLOW_TRACKING_URI)

    st.write("MinIO S3 Endpoint")
    st.code(MLFLOW_S3_ENDPOINT_URL)

    model_name = st.text_input("Registered Model Name", value="churn_model_final")

    version_mode = st.radio(
        "Model version",
        ["Latest", "Manual"],
        index=0,
    )

    if version_mode == "Latest":
        latest_version = get_latest_model_version(model_name)

        if latest_version is None:
            st.error(f"Không tìm thấy version nào cho model `{model_name}`")
            st.stop()

        model_version = latest_version
        st.success(f"Latest version: {model_version}")
    else:
        model_version = st.text_input("Version", value="1")

    if st.button("Reload model"):
        load_model.clear()


try:
    model, model_uri = load_model(model_name, model_version)
    st.success(f"✅ Loaded model: `{model_uri}`")
except Exception as e:
    st.error(f"❌ Không load được model: {e}")
    st.stop()


tab_single, tab_batch = st.tabs(["👤 Single Prediction", "📁 Batch Prediction"])


with tab_single:
    st.subheader("Nhập thông tin khách hàng")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", value=43.0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        country = st.text_input("Country", value="France")
        city = st.text_input("City", value="Marseille")
        membership_years = st.number_input("Membership_Years", value=2.9)
        login_frequency = st.number_input("Login_Frequency", value=14.0)
        session_duration_avg = st.number_input("Session_Duration_Avg", value=27.4)
        pages_per_session = st.number_input("Pages_Per_Session", value=6.0)

    with col2:
        cart_abandonment_rate = st.number_input("Cart_Abandonment_Rate", value=50.6)
        wishlist_items = st.number_input("Wishlist_Items", value=3.0)
        total_purchases = st.number_input("Total_Purchases", value=9.0)
        average_order_value = st.number_input("Average_Order_Value", value=94.72)
        days_since_last_purchase = st.number_input("Days_Since_Last_Purchase", value=34.0)
        discount_usage_rate = st.number_input("Discount_Usage_Rate", value=46.4)
        returns_rate = st.number_input("Returns_Rate", value=2.0)
        email_open_rate = st.number_input("Email_Open_Rate", value=17.9)

    with col3:
        customer_service_calls = st.number_input("Customer_Service_Calls", value=9.0)
        product_reviews_written = st.number_input("Product_Reviews_Written", value=4.0)
        social_media_score = st.number_input("Social_Media_Engagement_Score", value=16.3)
        mobile_app_usage = st.number_input("Mobile_App_Usage", value=20.8)
        payment_method_diversity = st.number_input("Payment_Method_Diversity", value=1.0)
        lifetime_value = st.number_input("Lifetime_Value", value=953.33)
        credit_balance = st.number_input("Credit_Balance", value=2278.0)
        signup_quarter = st.selectbox("Signup_Quarter", ["Q1", "Q2", "Q3", "Q4"])

    values = {
        "Age": age,
        "Gender": gender,
        "Country": country,
        "City": city,
        "Membership_Years": membership_years,
        "Login_Frequency": login_frequency,
        "Session_Duration_Avg": session_duration_avg,
        "Pages_Per_Session": pages_per_session,
        "Cart_Abandonment_Rate": cart_abandonment_rate,
        "Wishlist_Items": wishlist_items,
        "Total_Purchases": total_purchases,
        "Average_Order_Value": average_order_value,
        "Days_Since_Last_Purchase": days_since_last_purchase,
        "Discount_Usage_Rate": discount_usage_rate,
        "Returns_Rate": returns_rate,
        "Email_Open_Rate": email_open_rate,
        "Customer_Service_Calls": customer_service_calls,
        "Product_Reviews_Written": product_reviews_written,
        "Social_Media_Engagement_Score": social_media_score,
        "Mobile_App_Usage": mobile_app_usage,
        "Payment_Method_Diversity": payment_method_diversity,
        "Lifetime_Value": lifetime_value,
        "Credit_Balance": credit_balance,
        "Signup_Quarter": signup_quarter,
    }

    input_df = build_input_df(values)

    st.write("Input data")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🚀 Predict"):
        try:
            pred, stay_prob, churn_prob = predict(model, input_df)

            st.divider()

            if pred == 1:
                st.error("Kết quả: CHURN")
            else:
                st.success("Kết quả: STAY")

            if churn_prob is not None:
                c1, c2 = st.columns(2)
                c1.metric("Xác suất rời bỏ", f"{churn_prob:.2%}")
                c2.metric("Xác suất ở lại", f"{stay_prob:.2%}")

                st.progress(churn_prob)

        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")


with tab_batch:
    st.subheader("Upload CSV để predict hàng loạt")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.write("Preview")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("🚀 Predict batch"):
            try:
                preds = model.predict(batch_df)

                result_df = batch_df.copy()
                result_df["prediction"] = preds
                result_df["prediction_label"] = result_df["prediction"].map({
                    0: "STAY",
                    1: "CHURN",
                })

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(batch_df)
                    result_df["stay_probability"] = probs[:, 0]
                    result_df["churn_probability"] = probs[:, 1]

                st.success("✅ Predict batch thành công")
                st.dataframe(result_df, use_container_width=True)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download result CSV",
                    data=csv,
                    file_name="churn_prediction_result.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"❌ Lỗi khi predict batch: {e}")