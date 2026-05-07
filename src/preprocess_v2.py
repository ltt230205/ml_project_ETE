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
    clean_df = df.dropna()
    test_size_count = int(len(df) * 0.2)

    if len(clean_df) < test_size_count:
        print(f"Cảnh báo: Chỉ tìm thấy {len(clean_df)} bản ghi sạch. Lấy toàn bộ làm tập Test.")
        test_df = clean_df.copy()
    else:
        test_df = clean_df.sample(n=test_size_count, random_state=42).copy()

    # Tập Train là phần còn lại
    train_df = df.drop(test_df.index).copy()

    # 3. XỬ LÝ MISSING VALUES & OUTLIERS (Chỉ trên Train)
    num_cols = [
        "Age", "Session_Duration_Avg", "Pages_Per_Session", "Wishlist_Items",
        "Days_Since_Last_Purchase", "Discount_Usage_Rate", "Returns_Rate",
        "Email_Open_Rate", "Customer_Service_Calls", "Product_Reviews_Written",
        "Social_Media_Engagement_Score", "Mobile_App_Usage",
        "Payment_Method_Diversity", "Credit_Balance"
    ]

    for col in num_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(train_df[col].median())
            
            # IQR Outlier handling
            Q1 = train_df[col].quantile(0.25)
            Q3 = train_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            train_df[col] = np.where(train_df[col] < lower, lower,
                                    np.where(train_df[col] > upper, upper, train_df[col]))

    # 4. LƯU DATA ĐÃ XỬ LÝ (CHƯA ENCODE - vì sẽ encode trong Pipeline)
    os.makedirs(os.path.dirname(config['processed_data']['train_path']), exist_ok=True)
    
    train_df.to_csv(config['processed_data']['train_path'], index=False)
    test_df.to_csv(config['processed_data']['test_path'], index=False)
    
    print(f"✅ Đã chia dữ liệu thành công")
    print(f"Train: {config['processed_data']['train_path']}")
    print(f"Test: {config['processed_data']['test_path']}")

if __name__ == "__main__":
    preprocess()