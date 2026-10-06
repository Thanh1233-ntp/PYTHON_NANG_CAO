Kiểm tra xem từng module trong modules/ có hoạt động đúng hay không.

```
tests/
│
├── __init__.py
├── test_image_processor.py
├── test_detector.py
└── test_evaluator.py
```
```
                TESTS
                   │
       ┌───────────┼───────────┐
       ↓           ↓           ↓
 ImageProcessor   YOLO      Evaluator
       ↓           ↓           ↓
    Test 1       Test 2      Test 3
       │           │           │
       └───────────┼───────────┘
                   ↓
            Kiểm tra hệ thống
```
