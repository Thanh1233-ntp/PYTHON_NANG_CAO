MỤC TIÊU
  ``` 
                   USER
                     │
                     ▼
              Upload hình ảnh
                     │
                     ▼
             ImageProcessor
                     │
                     ▼
              TrashDetector
                     │
              ┌──────┴──────┐
              │             │
         class/label     confidence
              │             │
              └──────┬──────┘
                     ▼
              TrashEvaluator
                     │
                     ▼
              Kết quả đánh giá
                     │
                     ▼
                 Streamlit
                     │
                     ▼
              Hiển thị kết quả
```
