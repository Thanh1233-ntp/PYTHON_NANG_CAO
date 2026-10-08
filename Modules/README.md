```
modules/
│
├── image_processor.py
├── trash_detector.py
├── trash_evaluator.py
└── data_streamer.py
```
image_processor.py ( Xử lý ảnh  )       
```                                     
Input image
    ↓
Validate
    ↓
Resize
    ↓
Preprocess
```
trash_detector.py ( YOLO )
```
Image
 ↓
YOLO
 ↓
Detection
 ↓
Class
Confidence
Bounding box
```
trash_evaluator.py ( Đánh giá detection )
```
Detection
 ↓
Confidence
 ↓
Status
```
data_streamer.py ( Generator ) 
```
Image 1
Image 2
Image 3
```

# PHÂN CÔNG MODULE

---

## 👤 Thành viên 1 — ImageProcessor + TrashDetector

### 1. `modules/image_processor.py`
- **Class:** `ImageProcessor`
- **Nhiệm vụ:**
  - Kiểm tra ảnh đầu vào.
  - Chuẩn hóa / resize ảnh.
  - Trả về ảnh đã xử lý.
- **Method:**
  ```python
  __init__(target_size=640)
  validate(image)
  process(image)
  ```

### 2. `modules/trash_detector.py`
- **Class:** `TrashDetector`
- **Nhiệm vụ:**
  - Load YOLO model `models/best_model.pt`.
  - Nhận ảnh và thực hiện detection.
  - Lấy `class_name`, `confidence`, `bbox`.
- **Method:**
  ```python
  __init__(model_path)
  detect(image)
  ```
- **Output bắt buộc:**
  ```json
  [
      {
          "class_name": "Plastic",
          "confidence": 0.92,
          "bbox": [x1, y1, x2, y2]
      }
  ]
  ```

---

## 👤 Thành viên 2 — TrashEvaluator + DataStreamer

### 3. `modules/trash_evaluator.py`
- **Class:** `TrashEvaluator`
- **Nhiệm vụ:**
  - Nhận kết quả từ `TrashDetector`.
  - Tính số lượng object.
  - Tính confidence trung bình.
  - Phân loại confidence cao/thấp.
- **Method:**
  ```python
  __init__(confidence_threshold=0.5)
  evaluate(detections)
  ```
- **Output:**
  ```json
  {
      "total_objects": 2,
      "average_confidence": 0.82,
      "high_confidence": 2,
      "low_confidence": 0
  }
  ```
  *(Lưu ý: `0.5` là ngưỡng do nhóm lựa chọn, không phải yêu cầu bắt buộc trong đề).*

### 4. `modules/data_streamer.py`
- **Class:** `DataStreamer`
- **Nhiệm vụ:**
  - Đọc các ảnh trong `data/test_images/`.
  - Duyệt ảnh theo thứ tự.
  - Bắt buộc sử dụng Generator (`yield`).
- **Method:**
  ```python
  __init__(image_dir)
  stream()
  ```
- **Cách sử dụng:**
  ```python
  streamer = DataStreamer("data/test_images")

  for image_path in streamer.stream():
      print(image_path)
  ```
