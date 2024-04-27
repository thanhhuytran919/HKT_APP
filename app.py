import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import torch
from efficient_apriori import apriori
import csv


class YOLOv5App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IMG2RULE (HKT APP)")
        self.geometry("1000x600")
        self.create_widgets()

        # Load model YOLOv5
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path='./models/best.pt').to(self.device)
        self.model.eval()

    def create_widgets(self):
        self.title_label = tk.Label(self, text="IMG2RULE (HKT APP)", font=(
            "Helvetica", 20, "bold"), fg="red")
        self.title_label.pack(pady=10)

        # Thêm tiêu đề cho từng frame
        self.original_image_title = tk.Label(
            text="Original Image", font=("Helvetica", 8, "bold"))
        self.detected_image_title = tk.Label(
            text="Detected Image", font=("Helvetica", 8, "bold"))
        self.transaction_title = tk.Label(
            text="Transaction", font=("Helvetica", 8, "bold"))
        self.rule_title = tk.Label(
            text="Rule", font=("Helvetica", 8, "bold"))

        # Đặt các tiêu đề bên ngoài frame tương ứng
        self.original_image_title.place(x=120, y=40)
        self.detected_image_title.place(x=420, y=40)
        self.transaction_title.place(x=800, y=40)
        self.rule_title.place(x=1180, y=40)

        # Tạo cửa sổ cuộn cho ảnh gốc và đặt vị trí bên trái
        self.original_canvas = tk.Canvas(
            self, borderwidth=0, background="#ffffff", width=300, height=500)
        self.original_canvas.pack(side="left", fill="both", expand=False)
        self.original_frame = tk.Frame(
            self.original_canvas, background="#ffffff")
        self.original_frame.pack(side="left", fill="both", expand=False)

        # Tạo thanh cuộn cho ảnh gốc và đặt vị trí bên trái
        self.original_vsb = tk.Scrollbar(
            self, orient="vertical", command=self.original_canvas.yview)
        self.original_vsb.pack(side="left", fill="y")
        self.original_canvas.configure(
            yscrollcommand=self.original_vsb.set)
        self.original_canvas.create_window(
            (4, 4), window=self.original_frame, anchor="nw")

        self.original_frame.bind(
            "<Configure>", self.onOriginalFrameConfigure)

        self.original_image_title = tk.Label(
            self.original_frame, text="Original Image")
        self.original_image_title.pack()

        # Tạo cửa sổ cuộn cho ảnh đã phát hiện và đặt vị trí bên trái
        self.detected_canvas = tk.Canvas(
            self, borderwidth=0, background="#ffffff", width=300, height=500)
        self.detected_canvas.pack(side="left", fill="both", expand=False)
        self.detected_frame = tk.Frame(
            self.detected_canvas, background="#ffffff")
        self.detected_frame.pack(side="left", fill="both", expand=False)

        # Tạo thanh cuộn cho ảnh đã phát hiện và đặt vị trí bên trái
        self.detected_vsb = tk.Scrollbar(
            self, orient="vertical", command=self.detected_canvas.yview)
        self.detected_vsb.pack(side="left", fill="y")
        self.detected_canvas.configure(
            yscrollcommand=self.detected_vsb.set)
        self.detected_canvas.create_window(
            (4, 4), window=self.detected_frame, anchor="nw")

        self.detected_frame.bind(
            "<Configure>", self.onDetectedFrameConfigure)

        self.detected_image_title = tk.Label(
            self.detected_frame, text="Detected Image")
        self.detected_image_title.pack()

        # Tạo cửa sổ cuộn cho transaction và đặt vị trí bên phải
        self.transaction_canvas = tk.Canvas(
            self, borderwidth=0, background="#ffffff", width=350, height=500)
        self.transaction_canvas.pack(side="left", fill="both", expand=False)
        self.transaction_frame = tk.Frame(
            self.transaction_canvas, background="#ffffff")
        self.transaction_frame.pack(side="left", fill="both", expand=False)

        # Tạo thanh cuộn cho transaction và đặt vị trí bên phải
        self.transaction_vsb = tk.Scrollbar(
            self, orient="vertical", command=self.transaction_canvas.yview)
        self.transaction_vsb.pack(side="left", fill="y")
        self.transaction_canvas.configure(
            yscrollcommand=self.transaction_vsb.set)
        self.transaction_canvas.create_window(
            (4, 4), window=self.transaction_frame, anchor="nw")

        self.transaction_frame.bind(
            "<Configure>", self.onTransactionFrameConfigure)

        self.transaction_title = tk.Label(
            self.transaction_frame, text="Transaction")
        self.transaction_title.pack()

        # Tạo cửa sổ cuộn cho luật và đặt vị trí bên phải
        self.rule_canvas = tk.Canvas(
            self, borderwidth=0, background="#ffffff", width=350, height=500)
        self.rule_canvas.pack(side="left", fill="both", expand=False)
        self.rule_frame = tk.Frame(
            self.rule_canvas, background="#ffffff")
        self.rule_frame.pack(side="left", fill="both", expand=False)

        # Tạo thanh cuộn cho luật và đặt vị trí bên phải
        self.rule_vsb = tk.Scrollbar(
            self, orient="vertical", command=self.rule_canvas.yview)
        self.rule_vsb.pack(side="left", fill="y")
        self.rule_canvas.configure(
            yscrollcommand=self.rule_vsb.set)
        self.rule_canvas.create_window(
            (4, 4), window=self.rule_frame, anchor="nw")

        self.rule_frame.bind(
            "<Configure>", self.onRuleFrameConfigure)

        self.rule_title = tk.Label(
            self.rule_frame, text="Rule")
        self.rule_title.pack()

        # Nút để tải ảnh lên và đặt vị trí bên dưới
        self.upload_button = tk.Button(
            self, text="Upload Image", command=self.upload_images)
        self.upload_button.pack(side="top", pady=10)

        # Nút để tải thư mục lên và đặt vị trí bên dưới
        self.load_folder_button = tk.Button(
            self, text="Upload Folder", command=self.load_images_from_folder)
        self.load_folder_button.pack(side="top", pady=10)

        # Add a button to save results
        self.save_results_button = tk.Button(
            self, text="Save Results", command=self.save_results_to_csv)
        self.save_results_button.pack(side="top", pady=10)

        # Initialize rules variable to store inference results
        self.rules = []
        self.transactions = []

    def onOriginalFrameConfigure(self, event):
        '''Cập nhật thanh cuộn cho ảnh gốc'''
        self.original_canvas.configure(
            scrollregion=self.original_canvas.bbox("all"))

    def onDetectedFrameConfigure(self, event):
        '''Cập nhật thanh cuộn cho ảnh đã phát hiện'''
        self.detected_canvas.configure(
            scrollregion=self.detected_canvas.bbox("all"))

    def onTransactionFrameConfigure(self, event):
        '''Cập nhật thanh cuộn cho transaction'''
        self.transaction_canvas.configure(
            scrollregion=self.transaction_canvas.bbox("all"))

    def onRuleFrameConfigure(self, event):
        '''Cập nhật thanh cuộn cho rule'''
        self.rule_canvas.configure(
            scrollregion=self.rule_canvas.bbox("all"))

    def upload_images(self):
        self.rules = []
        self.transactions = []
        # Xóa tất cả các widget trong original_frame, detected_frame, transaction_frame và rule_frame
        for widget in self.original_frame.winfo_children():
            widget.destroy()
        for widget in self.detected_frame.winfo_children():
            widget.destroy()
        for widget in self.transaction_frame.winfo_children():
            widget.destroy()
        for widget in self.rule_frame.winfo_children():
            widget.destroy()

        # Mở hộp thoại để chọn tệp
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

        # Kiểm tra xem người dùng đã chọn tệp hay chưa
        if file_paths:
            for file_path in file_paths:
                # Mở và hiển thị ảnh gốc
                original_image = Image.open(file_path)
                original_image = original_image.resize(
                    (300, 300), resample=Image.BILINEAR)
                original_image = ImageTk.PhotoImage(original_image)
                original_label = tk.Label(
                    self.original_frame, image=original_image)
                original_label.image = original_image
                original_label.pack()

                # Phát hiện các đối tượng trong ảnh và hiển thị ảnh đã phát hiện
                detected_image = self.detect_objects(file_path)
                detected_label = tk.Label(
                    self.detected_frame, image=detected_image)
                detected_label.image = detected_image
                detected_label.pack()

                # Xử lý transaction và hiển thị
                self.process_image(file_path)
        self.display_transaction(self.transactions)
        # Apriori
        if self.transactions:
            itemsets, rules = apriori(
                self.transactions, min_support=0.5, min_confidence=1)
            self.rules = rules  # Gán lại biến self.rules với các luật mới
        self.display_rules(self.rules)

    def load_images_from_folder(self):
        self.rules = []
        self.transactions = []
        # Xóa tất cả các widget trong original_frame, detected_frame, transaction_frame và rule_frame
        for widget in self.original_frame.winfo_children():
            widget.destroy()
        for widget in self.detected_frame.winfo_children():
            widget.destroy()
        for widget in self.transaction_frame.winfo_children():
            widget.destroy()
        for widget in self.rule_frame.winfo_children():
            widget.destroy()

        # Mở hộp thoại để chọn thư mục
        folder_path = filedialog.askdirectory()

        # Kiểm tra xem người dùng đã chọn thư mục hay chưa
        if folder_path:
            # Đọc tất cả các tệp hình ảnh trong thư mục
            image_files = [f for f in os.listdir(
                folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

            # Kiểm tra xem có ít nhất một tệp hình ảnh hay không
            if image_files:
                for image_file in image_files:
                    # Xử lý mỗi tệp hình ảnh
                    image_path = os.path.join(folder_path, image_file)
                    # Mở và hiển thị ảnh gốc
                    original_image = Image.open(image_path)
                    original_image = original_image.resize(
                        (300, 300), resample=Image.BILINEAR)
                    original_image = ImageTk.PhotoImage(original_image)
                    original_label = tk.Label(
                        self.original_frame, image=original_image)
                    original_label.image = original_image
                    original_label.pack()

                    # Phát hiện các đối tượng trong ảnh và hiển thị ảnh đã phát hiện
                    detected_image = self.detect_objects(image_path)
                    detected_label = tk.Label(
                        self.detected_frame, image=detected_image)
                    detected_label.image = detected_image
                    detected_label.pack()

                    # Xử lý transaction và hiển thị
                    self.process_image(image_path)
            self.display_transaction(self.transactions)
            # Apriori
            if self.transactions:
                itemsets, rules = apriori(
                    self.transactions, min_support=0.5, min_confidence=1)
                self.rules = rules  # Gán lại biến self.rules với các luật mới
            self.display_rules(self.rules)

    def process_image(self, image_path):
        # Phát hiện đối tượng trong ảnh
        results = self.model(image_path)

        # Extract class names
        class_names_series = results.pandas().xyxy[0].name

        # Chuyển Series thành tuple và kiểm tra nếu có nhiều hơn một phần tử
        transaction = tuple(class_names_series.tolist())
        if len(set(transaction)) > 1:
            self.transactions.append(set(transaction))

    def detect_objects(self, image_path):
        # Phát hiện đối tượng trong ảnh
        results = self.model(image_path)

        # Vẽ bounding box và hiển thị ảnh
        annotated_image = results.render()[0]
        annotated_image = Image.fromarray(annotated_image)
        annotated_image = annotated_image.resize(
            (300, 300), resample=Image.BILINEAR)
        detected_image = ImageTk.PhotoImage(annotated_image)

        return detected_image

    def display_transaction(self, transactions):
        if transactions:
            transaction_text = "\n\n".join([str(transaction)
                                            for transaction in transactions])
            transaction_label = tk.Label(
                self.transaction_frame, text=transaction_text)
            transaction_label.pack()

    def display_rules(self, rules):
        if rules:
            rule_text = "\n\n".join(
                [f"{rule.lhs} => {rule.rhs}" for rule in rules])
            rule_label = tk.Label(self.rule_frame, text=rule_text)
            rule_label.pack()

    def save_results_to_csv(self):
        # Mở hộp thoại để chọn vị trí để lưu file CSV
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

        # Viết kết quả ra file CSV
        if file_path:
            with open(file_path, mode='w', newline='') as csvfile:
                fieldnames = ['A', 'B']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Viết header vào file CSV
                writer.writeheader()

                # Viết kết quả suy luận vào file CSV từ các luật
                if self.rules:
                    for rule in self.rules:
                        # Lấy danh sách các phần tử trong lhs và rhs
                        A, B = rule.lhs, rule.rhs
                        # Ghi mỗi cặp phần tử của lhs và rhs vào file CSV
                        for item_A in A:
                            for item_B in B:
                                writer.writerow({'A': item_A, 'B': item_B})

if __name__ == "__main__":
    app = YOLOv5App()
    app.mainloop()
