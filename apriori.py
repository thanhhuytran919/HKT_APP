import os
import torch
from efficient_apriori import apriori
import csv

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/best.pt')

# Đường dẫn đến thư mục chứa ảnh
folder_path = './test/'

# Danh sách các tuples của các tên lớp từ tất cả các ảnh
all_transactions = []

# Duyệt qua từng tệp trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
        # Đường dẫn đến ảnh
        image_path = os.path.join(folder_path, filename)

        # Inference
        results = model(image_path)

        # Extract class names
        class_names_series = results.pandas().xyxy[0].name

        # Chuyển Series thành tuple và kiểm tra nếu có nhiều hơn một phần tử
        transaction = tuple(class_names_series.tolist())
        if len(set(transaction)) > 1:
            all_transactions.append(set(transaction))

print(all_transactions)

# Apriori
itemsets, rules = apriori(all_transactions, min_support=0.5, min_confidence=1)
print(rules)

# Lưu kết quả suy luận vào file CSV
with open('inference_results.csv', 'w', newline='') as csvfile:
    # Định dạng các header
    fieldnames = ['A', 'B']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Viết header vào file CSV
    writer.writeheader()

    # Viết kết quả suy luận vào file CSV
    for rule in rules:
        A, B = rule.lhs, rule.rhs
        writer.writerow({'A': ','.join(A), 'B': ','.join(B)})
