# import pandas as pd
# import numpy as np
# import yaml
# import os
# from sklearn.preprocessing import LabelEncoder

# def preprocess():
#     with open("params.yaml", "r") as f:
#         config = yaml.safe_load(f)

#     # Đọc dữ liệu thô
#     df = pd.read_csv(config['raw_data']['path'])
    
#     id_cols = [col for col in df.columns if 'id' in col.lower()]
#     df = df.drop(columns=id_cols)

#     num_cols = [
#     "Age",
#     "Session_Duration_Avg",
#     "Pages_Per_Session",
#     "Wishlist_Items",
#     "Days_Since_Last_Purchase",
#     "Discount_Usage_Rate",
#     "Returns_Rate",
#     "Email_Open_Rate",
#     "Customer_Service_Calls",
#     "Product_Reviews_Written",
#     "Social_Media_Engagement_Score",
#     "Mobile_App_Usage",
#     "Payment_Method_Diversity",
#     "Credit_Balance"
#     ]

#     for col in num_cols:
#         df[col].fillna(df[col].median(), inplace=True)

#     num_cols = df.select_dtypes(include=np.number).columns.drop("Churned")

#     for col in num_cols:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
        
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR
        
#         df[col] = np.where(df[col] < lower, lower,
#                         np.where(df[col] > upper, upper, df[col]))
        
#     cat_cols = df.select_dtypes(include=["object"]).columns

#     le = LabelEncoder()
#     for col in cat_cols:
#         df[col] = le.fit_transform(df[col])

#     os.makedirs(os.path.dirname(config['processed_data']['path']), exist_ok=True)
#     df.to_csv(config['processed_data']['path'], index=False)
#     print(f"--- Đã chuẩn bị xong dữ liệu tại: {config['processed_data']['path']} ---")

# if __name__ == "__main__":
#     preprocess()
import pandas as pd
import numpy as np
import yaml
import os
from sklearn.preprocessing import LabelEncoder

def preprocess():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Đọc dữ liệu thô
    df = pd.read_csv(config['raw_data']['path'])
    
    # Loại bỏ ID ngay từ đầu
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    df = df.drop(columns=id_cols)

    # 2. TÁCH TẬP TEST SẠCH (20%)
    # Lọc các dòng không có bất kỳ giá trị NaN nào
    clean_df = df.dropna()
    test_size_count = int(len(df) * 0.2)

    if len(clean_df) < test_size_count:
        print(f"Cảnh báo: Chỉ tìm thấy {len(clean_df)} bản ghi sạch. Lấy toàn bộ làm tập Test.")
        test_df = clean_df.copy()
    else:
        test_df = clean_df.sample(n=test_size_count, random_state=42).copy()

    # Tập Train là phần còn lại (bao gồm cả các bản ghi có NaN)
    train_df = df.drop(test_df.index).copy()

    # 3. XỬ LÝ DỮ LIỆU (Chỉ thực hiện biến đổi trên Train, Test giữ nguyên để LabelEncode sau)
    num_cols = [
    "Age",
    "Session_Duration_Avg",
    "Pages_Per_Session",
    "Wishlist_Items",
    "Days_Since_Last_Purchase",
    "Discount_Usage_Rate",
    "Returns_Rate",
    "Email_Open_Rate",
    "Customer_Service_Calls",
    "Product_Reviews_Written",
    "Social_Media_Engagement_Score",
    "Mobile_App_Usage",
    "Payment_Method_Diversity",
    "Credit_Balance"
    ]

    # --- Xử lý trên tập TRAIN ---
    for col in num_cols:
        # Điền median của chính tập Train vào chỗ trống
        train_df[col] = train_df[col].fillna(train_df[col].median())

        # Xử lý Outliers bằng IQR (Chỉ áp dụng cho Train)
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        train_df[col] = np.where(train_df[col] < lower, lower,
                                np.where(train_df[col] > upper, upper, train_df[col]))

    # 4. LABEL ENCODING (Cho cả 2 tập)
    cat_cols = train_df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    
    for col in cat_cols:
        # Train fit và transform
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        # Test chỉ transform (dựa trên các nhãn đã thấy ở Train)
        test_df[col] = le.transform(test_df[col].astype(str))

    # 5. LƯU FILE
    os.makedirs(os.path.dirname(config['processed_data']['train_path']), exist_ok=True)
    
    train_df.to_csv(config['processed_data']['train_path'], index=False)
    test_df.to_csv(config['processed_data']['test_path'], index=False)
    
    print(f"--- Đã chia dữ liệu thành công ---")
    print(f"Train path (80%): {config['processed_data']['train_path']} | NaN đã được xử lý")
    print(f"Test path (20% sạch): {config['processed_data']['test_path']}")

if __name__ == "__main__":
    preprocess()