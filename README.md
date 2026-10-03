MỤC TIÊU
  ``` 
                   USER
                     │
                     │ Upload ảnh
                     ▼
              ┌──────────────┐
              │  Streamlit   │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │ImageProcessor│
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │TrashDetector │
              │     YOLO     │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │TrashEvaluator│
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │    Result    │
              └──────┬───────┘
                     │
                     ▼
               Streamlit
```
