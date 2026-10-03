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
...
```
