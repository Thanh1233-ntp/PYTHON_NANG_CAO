# Báo cáo tiến trình: Thành viên 1 — xử lý ảnh và nhận diện rác

## Phạm vi và thay đổi

- Đã đọc `HƯỚNG DẪN.md`, kiểm tra model, notebook YOLO cũ, cấu trúc module/test và các thư viện được import trong file Python/notebook của dự án.
- `modules/image_processor.py`: kiểm tra ảnh là `numpy.ndarray` có 3 chiều và kích thước khác 0; xử lý ảnh về khung vuông 640×640 bằng nearest-neighbor, giữ nguyên tỷ lệ và thêm viền màu 114 (letterbox). Cải tiến này tránh kéo méo ảnh khi đưa ảnh ngang/dọc về đầu vào vuông.
- `modules/trash_detector.py`: báo lỗi rõ ràng nếu model không tồn tại; tải model YOLO được truyền vào; duyệt mọi kết quả và mọi bounding box; trả `class_name`, confidence dạng `float` và bbox bốn tọa độ dạng `float`. Kết quả không có box trả về `[]`; tắt log lặp của YOLO khi dự đoán.
- `tests/test_detector.py`: kiểm tra đầu vào sai/đúng, letterbox ảnh nhỏ và lớn, nhiều box, không có detection và thiếu model.
- Tạo `visualize_detections.ipynb`: chọn đều tối đa 10 ảnh, chạy model, hiển thị ảnh gốc cạnh ảnh đã letterbox/vẽ box, lưu ảnh minh họa và CSV.
- Tạo `requirements.txt` với dependency trực tiếp cần cho module, notebook và test; các thư viện chuẩn Python không cần liệt kê.

## Chạy thử thực tế

- Model: `models/best_model.pt`; YOLO load thành công. Model khai báo 5 nhãn: `Glass`, `Metal`, `Paper`, `Plastic`, `Waste`.
- Trong thư mục thử nghiệm có **3.266 ảnh** với phần mở rộng ảnh được hỗ trợ. Notebook chạy một mẫu gồm 10 ảnh lấy đều từ đầu đến cuối danh sách, trên CPU (CUDA không khả dụng).
- Kết quả mẫu:

| Ảnh | Số box | Nhãn model trả về |
|---|---:|---|
| `image_0001.jpg` | 0 | Không phát hiện |
| `image_0363.jpg` | 1 | Metal |
| `image_0726.jpg` | 2 | Plastic, Plastic |
| `image_1089.jpg` | 1 | Metal |
| `image_1452.jpg` | 1 | Paper |
| `image_1814.jpg` | 1 | Paper |
| `image_2177.jpeg` | 2 | Paper, Waste |
| `image_2540.jpg` | 1 | Plastic |
| `image_2903.jpg` | 5 | Plastic, Plastic, Metal, Plastic, Plastic |
| `image_3266.jpg` | 2 | Plastic, Plastic |

- Tổng cộng notebook đã ghi 16 detection. Sau lần chạy đầu làm nóng model (khoảng 3,7 giây), các mẫu còn lại mất khoảng 0,15–0,28 giây mỗi ảnh trên máy hiện tại. Đây là thời gian đo cho mẫu nhỏ trên CPU, không phải benchmark toàn bộ tập.
- Tạo được 10 ảnh có bbox, `results/visualizations/detection_grid.png` và `results/visualizations/detection_summary.csv`.
- Kết quả dự đoán chưa thể dùng để tính độ chính xác hoặc kết luận ảnh nào bị nhận sai/bỏ sót vì thư mục hiện không có nhãn chuẩn đối chiếu. Ảnh `image_0001.jpg` không có box cũng chưa đủ căn cứ kết luận model bỏ sót.

## Thư viện và phiên bản

Các import bên ngoài trong source/notebook/test và phiên bản có trong môi trường đã kiểm tra:

| Package (pip) | Phiên bản | Vai trò |
|---|---:|---|
| `numpy` | 2.1.3 | Mảng ảnh và xử lý letterbox |
| `ultralytics` | 8.4.183 | Nạp và chạy YOLO |
| `opencv-python` | 5.0.0.93 | Đọc ảnh/vẽ bbox trong notebook (`cv2.__version__` hiển thị 5.0.0) |
| `matplotlib` | 3.10.0 | Hiển thị và lưu biểu đồ ảnh |
| `ipykernel` | 7.3.0 | Kernel chạy notebook trong VS Code/Jupyter |
| `pytest` | 9.1.1 | Chạy lệnh test theo hướng dẫn |

Model YOLO hiện dùng `torch 2.14.0+cpu` và `torchvision 0.29.0+cpu` như dependency bắc cầu; CUDA không khả dụng trên môi trường chạy hiện tại. Các package bắc cầu khác được pip cài theo metadata của `ultralytics`. `requirements.txt` ghim các dependency trực tiếp của dự án, không sao chép toàn bộ package không liên quan trong Python environment.

## Vướng mắc và giới hạn

1. Ban đầu thiếu `pytest`; đã cài bằng pip, sau đó cài/kiểm tra thành công bằng `python -m pip install -r requirements.txt`.
2. Môi trường Python 3.13.7 dùng chung còn các xung đột không thuộc dependency dự án; `pip check` vẫn báo `alembic` và `optuna` thiếu `sqlalchemy`, và `autoviz` yêu cầu `xgboost <1.7` trong khi môi trường có `xgboost 3.1.2`. Không tự cài/sửa các gói ngoài phạm vi.
3. Công cụ chạy cell trực tiếp trong VS Code báo không tìm thấy đường dẫn notebook dù file đã có trên đĩa. Đã xác minh JSON và thực thi toàn bộ source code cell bằng Python; inference, CSV và ảnh grid được tạo thành công. Lần thực thi kiểm tra dùng backend Matplotlib không tương tác nên cảnh báo `plt.show()` không hiển thị inline trong lần đó; ảnh grid đã lưu để xem trực tiếp. Khi mở trong VS Code/Jupyter, có thể chạy lần lượt các cell để hiện biểu đồ tương tác.
4. Phạm vi dự án mô tả bốn lớp, nhưng model trả cả `Waste`. Hiện detector và notebook giữ nguyên nhãn model, không âm thầm loại lớp; cần nhóm quyết định có tính `Waste` vào đầu ra hay không.
5. `models/README.md` ghi `best.pt` nhưng file thật là `models/best_model.pt`; phần triển khai dùng đường dẫn file đang tồn tại.
6. Worktree có một số thay đổi khác ngoài phạm vi xuất hiện trong lần kiểm tra cuối (`Data/README.md`, `Data/test_images/automatic_rename.py`, `data/test_images/auto_rename.py`, `abcxyz.md`). Tôi không chỉnh sửa hay khôi phục các đường dẫn này.

## Lệnh kiểm tra và cách chạy lại

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/test_detector.py
```

Mở `visualize_detections.ipynb` trong VS Code, chạy các cell từ trên xuống. Có thể chỉnh `MAX_IMAGES` để tăng/giảm số ảnh mẫu. Hiện notebook giới hạn 10 ảnh nhằm tạo kết quả nhanh; chạy nhiều ảnh hơn sẽ tăng thời gian xử lý tương ứng.

## Đề xuất bước tiếp theo

1. Xác nhận quy tắc xử lý nhãn `Waste` và cập nhật README model cho đúng tên file.
2. Đánh dấu/chuẩn bị nhãn chuẩn cho một tập kiểm thử, rồi ghi lại false positive/false negative; không suy luận lỗi chỉ từ confidence hoặc kết quả rỗng.
3. Sau khi kiểm tra mẫu 10 ảnh, tăng số lượng mẫu hoặc chạy đủ 3.266 ảnh nếu cần thống kê; ghi nhận phần cứng và thời gian để kết quả có thể so sánh.
4. Nếu cần sử dụng model trong ứng dụng, bổ sung bước chuyển bbox từ khung letterbox về tọa độ ảnh gốc; notebook hiện vẽ bbox trên khung 640×640 đã xử lý để đảm bảo tọa độ hiển thị đúng.
