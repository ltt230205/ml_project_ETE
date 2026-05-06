#!/bin/bash

# Tên thư mục môi trường ảo
VENV_NAME="venv"

echo "--- Đang bắt đầu quá trình thiết lập dự án ---"

# 1. Cập nhật hệ thống và cài đặt python3-venv nếu chưa có
echo "Cập nhật và kiểm tra python3-venv..."
sudo apt update && sudo apt install -y python3-venv

# 2. Tạo môi trường ảo nếu chưa tồn tại
if [ ! -d "$VENV_NAME" ]; then
    echo "Đang tạo môi trường ảo: $VENV_NAME..."
    python3 -m venv $VENV_NAME
else
    echo "Môi trường ảo '$VENV_NAME' đã tồn tại."
fi

# 3. Kích hoạt môi trường ảo
echo "Đang kích hoạt môi trường ảo..."
source $VENV_NAME/bin/activate

# 4. Nâng cấp pip lên bản mới nhất trong venv
echo "Đang nâng cấp pip..."
pip install --upgrade pip

# 5. Cài đặt các thư viện cần thiết
echo "Đang cài đặt boto3 và s3fs..."
pip install boto3 s3fs

# 6. Kiểm tra kết quả
if [ $? -eq 0 ]; then
    echo "--- Cài đặt thành công! ---"
    echo "Để bắt đầu làm việc, hãy gõ lệnh: source $VENV_NAME/bin/activate"
else
    echo "--- Có lỗi xảy ra trong quá trình cài đặt ---"
fi