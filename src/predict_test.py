import mlflow.sklearn  # Thay đổi từ pyfunc sang sklearn
import pandas as pd

# 1. Cấu hình MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# 2. Load model dưới dạng sklearn để dùng được hàm predict_proba
model_uri = "models:/test/1"
try:
    model = mlflow.sklearn.load_model(model_uri)
    print(f"🚀 Đã tải mô hình từ: {model_uri}")
except Exception as e:
    print(f"❌ Lỗi khi tải mô hình: {e}")
    exit(1)

# 3. Tạo dữ liệu test (Đảm bảo các cột số là float64 để khớp với Schema)
test_data = pd.DataFrame([{
    "Age": 43.0,
    "Gender": "Male",
    "Country": "France",
    "City": "Marseille",
    "Membership_Years": 2.9,
    "Login_Frequency": 14.0,
    "Session_Duration_Avg": 27.4,
    "Pages_Per_Session": 6.0,
    "Cart_Abandonment_Rate": 50.6,
    "Wishlist_Items": 3.0,
    "Total_Purchases": 9.0,
    "Average_Order_Value": 94.72,
    "Days_Since_Last_Purchase": 34.0,
    "Discount_Usage_Rate": 46.4,
    "Returns_Rate": 2.0,
    "Email_Open_Rate": 17.9,
    "Customer_Service_Calls": 9.0,
    "Product_Reviews_Written": 4.0,
    "Social_Media_Engagement_Score": 16.3,
    "Mobile_App_Usage": 20.8,
    "Payment_Method_Diversity": 1.0,
    "Lifetime_Value": 953.33,
    "Credit_Balance": 2278.0,
    "Signup_Quarter": "Q1"
}])

# Đảm bảo ép kiểu float cho tất cả các cột số để tránh lỗi Schema mismatch
numeric_cols = test_data.select_dtypes(include=['int64', 'float64']).columns
test_data[numeric_cols] = test_data[numeric_cols].astype('float64')

# 4. Dự đoán
try:
    # Lấy class dự đoán (0 hoặc 1)
    prediction = model.predict(test_data)
    
    # Lấy xác suất dự đoán
    probability = model.predict_proba(test_data)

    print("-" * 30)
    print(f"✅ Kết quả dự đoán: {'CHURN' if prediction[0] == 1 else 'STAY'}")
    print(f"📊 Xác suất rời bỏ: {probability[0][1]:.2%}")
    print(f"📊 Xác suất ở lại: {probability[0][0]:.2%}")
    print("-" * 30)

except Exception as e:
    print(f"❌ Lỗi khi dự đoán: {e}"