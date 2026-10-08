# HƯỚNG DẪN THỰC HÀNH 


---

## Thành viên 1: Xử lý ảnh và nhận diện rác

**File phụ trách:** `modules/image_processor.py`, `modules/trash_detector.py`, `tests/test_detector.py`

### Bước 1. Viết `ImageProcessor`

1. `__init__(target_size: int = 640)`: lưu kích thước đích.
2. `validate(image) -> bool`: kiểm tra ảnh không phải `None`, đúng kiểu `numpy.ndarray`, có 3 chiều, kích thước lớn hơn 0. Nếu sai thì báo `ValueError` rõ ràng.
3. `process(image) -> np.ndarray`: gọi `validate`, resize về `target_size`, trả về ảnh đã xử lý.
4. Chạy thử với: ảnh bình thường, ảnh rất nhỏ, ảnh rất lớn, ảnh lỗi hoặc `None`.

### Bước 2. Viết `TrashDetector`

1. `__init__(model_path: str)`: load YOLO từ `models/best_model.pt`. Nếu không thấy file thì báo lỗi dễ hiểu.
2. `detect(image) -> list[dict]`: chạy model và duyệt **toàn bộ** box trong kết quả (không chỉ lấy box đầu tiên) để nhận được nhiều loại rác trong một ảnh.
3. Mỗi vật thể trả về đúng định dạng đã thống nhất:

   ```python
   {"class_name": "Plastic", "confidence": 0.92, "bbox": [x1, y1, x2, y2]}
   ```

4. Lưu ý: `confidence` là số thực, `bbox` là list 4 số, `class_name` lấy từ `model.names`.
5. Không phát hiện được gì thì trả về `[]`, không để chương trình lỗi.

### Bước 3. Viết `tests/test_detector.py`

Tối thiểu các test sau:

- `validate` từ chối ảnh `None` hoặc sai kiểu.
- `process` trả ra ảnh đúng kích thước đích.
- `detect` trả về `list`, mỗi phần tử có đủ 3 khóa `class_name`, `confidence`, `bbox`.
- `detect` trên ảnh không có rác thì trả về `[]`.

Lệnh chạy:

```bash
python -m pytest tests/test_detector.py
```

### Bước 4. Chạy thử thực tế

Chạy qua nhiều ảnh khác nhau và kiểm tra:

- Có nhận diện được **nhiều loại rác trong cùng một ảnh** hay không?
- Kết quả có đủ **tên loại, độ tin cậy, vị trí** hay không?
- Ghi lại các trường hợp **nhận diện sai** hoặc **bỏ sót**: tên ảnh, mô tả lỗi, nguyên nhân nghi ngờ (ảnh mờ, vật bị che, nền phức tạp...).
- Vẽ bbox lên ảnh (làm ở script phụ, không đặt trong module) và lưu 5–10 ảnh kết quả tiêu biểu.

### Bước 5. Bàn giao cho trưởng nhóm

- [ ] Code hoàn chỉnh (2 module + file test)
- [ ] Kết quả chạy thử (log hoặc bảng)
- [ ] Một số ảnh kết quả có vẽ bbox
- [ ] Danh sách lỗi còn tồn tại
- [ ] Mô tả ngắn 5–7 dòng về phần đã làm (dùng cho báo cáo)

---

## 3. Thành viên 2: Đánh giá và chạy thử

**File phụ trách:** `modules/trash_evaluator.py`, `modules/data_streamer.py`, `tests/test_evaluator.py`

### Bước 1. Viết `TrashEvaluator`

1. `__init__(confidence_threshold: float = 0.5)`: lưu ngưỡng (0.5 là ngưỡng do nhóm chọn, không bắt buộc theo đề).
2. `evaluate(detections: list[dict]) -> dict`: nhận danh sách từ `TrashDetector` và trả về:

   ```python
   {"total_objects": 2, "average_confidence": 0.82,
    "high_confidence": 2, "low_confidence": 0}
   ```

3. Xử lý danh sách **rỗng**: `total_objects = 0`, `average_confidence = 0.0` (tránh chia cho 0).
4. Quy ước rõ: `confidence >= threshold` là cao, còn lại là thấp.

### Bước 2. Viết `DataStreamer`

1. `__init__(image_dir: str)`: lưu đường dẫn thư mục.
2. `stream()`: **bắt buộc dùng generator (`yield`)**, lần lượt trả về đường dẫn từng ảnh.
3. Chỉ lấy file ảnh (`.jpg`, `.jpeg`, `.png`...), sắp xếp theo tên để thứ tự cố định.
4. Thư mục không tồn tại hoặc trống thì báo lỗi hoặc xử lý rõ ràng, không làm chương trình sập.

### Bước 3. Viết `tests/test_evaluator.py`

- Evaluator: danh sách rỗng, toàn confidence cao, toàn thấp, lẫn lộn, giá trị đúng bằng ngưỡng.
- Kiểm tra `average_confidence` tính đúng.
- Streamer: `stream()` là generator (`inspect.isgenerator`), trả đúng số ảnh, bỏ qua file không phải ảnh.

Lệnh chạy:

```bash
python -m pytest tests/test_evaluator.py
```

### Bước 4. Thử nghiệm

Bỏ tất cả vào `data/test_images/`, **không chia thư mục con** theo loại rác (`plastic`, `paper`, `metal`...). Nên có sự đa dạng:

- Ảnh chứa 1 loại rác và ảnh chứa nhiều loại rác trong cùng một ảnh.
- Ảnh đủ sáng, thiếu sáng, hơi mờ.
- Nền đơn giản và nền lộn xộn.
- Vật thể to, nhỏ, bị che một phần.
- Vài ảnh không có rác (để kiểm tra trường hợp rỗng).

### Bước 5. Chạy thử và ghi kết quả

Chạy chương trình qua toàn bộ ảnh theo chuỗi `DataStreamer` → `ImageProcessor` → `TrashDetector` → `TrashEvaluator`. Trong lúc Thành viên 1 chưa xong, có thể dùng dữ liệu giả cùng định dạng để kiểm tra phần của mình.

Lưu kết quả vào thư mục `results/` dưới dạng `.csv` theo bảng sau (đo thời gian bằng `time.perf_counter()`):

| Tên ảnh | Số vật thể | Các loại rác phát hiện | Độ tin cậy | Thời gian xử lý (s) |
|---|---|---|---|---|
| img_001.jpg | 2 | Plastic, Paper | 0.92; 0.81 | 0.35 |
| ... | ... | ... | ... | ... |

### Bước 6. Bàn giao cho trưởng nhóm

- [ ] Code hoàn chỉnh (2 module + file test)
- [ ] Bảng kết quả thử nghiệm trong `results/`
- [ ] Một số ảnh kết quả
- [ ] Danh sách lỗi còn tồn tại
- [ ] Mô tả ngắn 5–7 dòng về phần đã làm (dùng cho báo cáo)

---

