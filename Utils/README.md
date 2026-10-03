Chứa các thành phần dùng chung cho toàn bộ hệ thống

```
utils/
│
├── __init__.py
├── decorators.py
└── README.md
```
decorators.py ( theo dõi và hỗ trợ quá trình xử lý của hệ thống )

```
Function
   ↓
Decorator
   ↓
Monitoring / Timing / Logging
   ↓
Function Result
```
