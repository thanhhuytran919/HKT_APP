# HKT_app

## Giới thiệu
HKT_app là một ứng dụng Python sử dụng YOLOv5 để phát hiện đối tượng trong ảnh và sau đó áp dụng thuật toán Apriori để khai thác luật kết hợp từ các đối tượng được phát hiện.

## Cài đặt
1. Clone repository từ GitHub:
    ```bash
    git clone https://github.com/thanhhuytran919/HKT_app.git
    ```
2. Di chuyển vào thư mục dự án:
    ```bash
    cd HKT_app
    ```
3. Cài đặt các dependencies từ file `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Sử dụng
1. Chạy file `app.py` để khởi động ứng dụng:
    ```bash
    python app.py
    ```
2. Sử dụng nút "Upload Image" hoặc "Upload Folder" để tải lên ảnh hoặc thư mục chứa các ảnh.
3. Sau khi tải lên, ứng dụng sẽ hiển thị ảnh gốc, ảnh đã phát hiện và kết quả thuật toán Apriori trong giao diện người dùng.

## Yêu cầu
Để chạy được ứng dụng, bạn cần có các dependencies sau:
- `efficient_apriori`
- `torch`
- `tk`
- `Pillow`

## Đóng góp
Mọi đóng góp và ý kiến đóng góp đều được hoan nghênh. Nếu bạn muốn đóng góp vào dự án, vui lòng tạo pull request trên GitHub.

## Tác giả
- [Thanh Huy Tran](https://github.com/thanhhuytran919)

## Giấy phép
Dự án này được phân phối dưới giấy phép [MIT License](https://opensource.org/licenses/MIT).
