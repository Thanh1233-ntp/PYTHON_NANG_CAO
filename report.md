# Báo cáo hiện trạng: Nhận diện rác bằng YOLO

## Tình trạng hiện tại

- Notebook `results/visualize_detections.ipynb` đã chạy được; lỗi môi trường và import `cv2` trước đây không còn là vấn đề theo xác nhận hiện tại.
- Model đang dùng: `models/best_model.pt`, có 5 nhãn: `Glass`, `Metal`, `Paper`, `Plastic`, `Waste`.
- Lỗi còn lại là độ chính xác phân loại/phát hiện của model. Trọng số model chưa được huấn luyện lại.
- Đã bổ sung cấu hình suy luận cho `TrashDetector`: ngưỡng confidence mặc định `0.25`, kích thước ảnh `640`; có thể đổi khi khởi tạo detector.
- Khi suy luận, nên đưa ảnh gốc trực tiếp vào YOLO để model tự resize/letterbox, thay vì giảm ảnh trước bằng phép nearest-neighbor của `ImageProcessor`.

## Kiểm tra trên 10 ảnh có nhãn tham chiếu

Nhãn tham chiếu dưới đây do người dùng cung cấp. Đây là mười ảnh đã được gán nhãn, không phải mẫu ngẫu nhiên từ toàn bộ tập ảnh. Đánh giá ở mức “có phát hiện đúng lớp mong đợi trong ảnh”; hiện chưa có bounding-box chuẩn để tính mAP hoặc đánh giá vị trí khung.

| Ảnh | Nhãn chuẩn | Kết quả YOLO (`imgsz=640`, `conf=0.25`) | Đúng lớp |
|---|---|---|---|
| `image_0001.jpg` | Glass | Không phát hiện | Không |
| `image_0363.jpg` | Glass | Metal | Không |
| `image_0726.jpg` | Plastic | Plastic, Plastic | Có |
| `image_1089.jpg` | Metal | Metal | Có |
| `image_1452.jpg` | Paper | Paper | Có |
| `image_1814.jpg` | Paper | Paper | Có |
| `image_2177.jpeg` | Paper | Paper, Paper | Có |
| `image_2540.jpg` | Plastic | Plastic | Có |
| `image_2903.jpg` | Plastic | Plastic, Plastic, Metal, Plastic, Plastic | Có, kèm một lớp phụ sai |
| `image_3266.jpg` | Plastic | Plastic | Có |

- Có lớp đúng trong **8/10 ảnh (80%)**.
- **7/10 ảnh** chỉ có detection thuộc lớp mong đợi; ảnh `image_2903.jpg` có thêm detection `Metal`.
- Hai trường hợp sai chính: bỏ sót ảnh Glass `image_0001.jpg`, và phân loại ảnh Glass `image_0363.jpg` thành Metal.

### So sánh cấu hình suy luận

Đã chạy trực tiếp YOLO trên ảnh gốc với các cấu hình dưới đây, không thay đổi trọng số:

| `imgsz` | `conf` | Ảnh có lớp chuẩn được phát hiện | Chỉ có lớp chuẩn |
|---:|---:|---:|---:|
| 640 | 0.25 | 8/10 | 7/10 |
| 960 | 0.25 | 7/10 | 4/10 |
| 960 | 0.15 | 8/10 | 4/10 |

Trên mẫu nhỏ này, `640/0.25` cho ít detection sai lớp phụ hơn; vì vậy đây là cấu hình mặc định hiện tại. Kết quả 10 ảnh chưa đủ để khẳng định cấu hình tối ưu cho dữ liệu nói chung.

## Chạy tự động từ Windows CMD

Từ CMD, vào thư mục dự án rồi chạy:

```bat
run.bat
```

Lệnh mặc định chọn ngẫu nhiên tối đa 10 ảnh với seed mới mỗi lần chạy, chạy YOLO trên ảnh gốc, lưu CSV và ảnh đã vẽ bounding box vào một thư mục có thời gian chạy riêng trong `results/automatic_run/`. Chương trình in seed đã dùng ở cuối để bạn có thể chạy lại đúng mẫu đó.

Các tùy chọn:

```bat
run.bat --reference
run.bat --limit 25 --seed 123
run.bat --all
```

- `--reference`: chạy đúng 10 ảnh có nhãn người dùng cung cấp để kiểm tra tỷ lệ đúng lớp.
- `--limit N --seed S`: chạy một mẫu ngẫu nhiên có thể lặp lại. Nếu bỏ `--seed`, mỗi lần chạy sẽ chọn mẫu mới.
- `--all`: chạy toàn bộ ảnh trong `data/test_images/`; không lưu ảnh đã vẽ cho toàn bộ tập để tránh tạo hàng nghìn file. CSV vẫn được lưu.
- Có thể thêm `--save-images` để lưu ảnh đã vẽ trong chế độ `--all`.

Kết quả của lần chạy kiểm tra tham chiếu gần nhất:

- CSV: `results/automatic_run/20261022_230837/detection_summary.csv`
- Ảnh kết quả: `results/automatic_run/20261022_230837/annotated_images/`
- Thời gian inference: `6.61 giây` cho 10 ảnh, gồm khoảng `4.76 giây` khởi động model ở ảnh đầu; các ảnh sau khoảng `0.15–0.26 giây/ảnh` trên môi trường hiện tại.

Nếu chưa có môi trường `.venv` trong thư mục dự án, cài dependency trước bằng `python -m pip install -r requirements.txt`. Launcher tự ưu tiên `.venv\Scripts\python.exe`, nếu chưa có thì gọi Python Launcher `py`.

## Kiểm thử phần mềm

```bat
python -m pytest tests\test_detector.py
```

Kết quả sau khi cập nhật detector: **9 tests passed**. Các test bao gồm định dạng detection, tham số `conf`/`imgsz`, giá trị tham số không hợp lệ, ảnh không có detection và thiếu model.

## Giới hạn và hướng cải thiện

- Không có bộ dữ liệu huấn luyện/validation kèm nhãn bounding box trong dự án. Vì vậy chưa thể huấn luyện lại hoặc đánh giá mAP một cách đáng tin cậy; không nên sửa/loại detection theo tên file hay nhãn của 10 ảnh mẫu.
- Kết quả hiện tại chỉ đo khả năng tìm thấy nhãn vật liệu mong đợi ở cấp ảnh; không đánh giá đầy đủ số lượng vật thể, vị trí khung hoặc detection thừa.
- Muốn tăng độ chính xác bền vững, cần bổ sung ảnh đại diện và annotation bounding box đúng cho bốn lớp mục tiêu, thống nhất cách xử lý lớp `Waste`, sau đó chia tập train/validation/test độc lập và fine-tune YOLO. Tạm thời giữ nguyên `best_model.pt`; chưa có bằng chứng để thay trọng số hay chọn ngưỡng khác.
