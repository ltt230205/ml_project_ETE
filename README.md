
# import dữ liệu từ nguồn bên ngoài 
dvc import-url s3://datasets/ecommerce_customer_churn_dataset.csv data/raw/



# start api streamlit

export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
export AWS_DEFAULT_REGION=us-east-1

streamlit run src/streamlit_app.py --server.address 0.0.0.0 --server.port 8501