"""
DataStreamer: lần lượt trả về đường dẫn từng ảnh trong một thư mục.

Ví dụ:
    streamer = DataStreamer("data/test_images")
    for duong_dan in streamer.stream():
        print(duong_dan)          # Path tới từng ảnh, theo thứ tự tên
"""
from pathlib import Path
from typing import Iterator


class DataStreamer:
    # Chỉ lấy các tệp ảnh có phần mở rộng này (không phân biệt hoa/thường)
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, image_dir: str) -> None:
        # Chỉ lưu đường dẫn thư mục; việc kiểm tra được làm khi gọi stream().
        self.image_dir = Path(image_dir).expanduser()

    def stream(self) -> Iterator[Path]:
        """Trình tạo (generator): lần lượt yield đường dẫn từng ảnh, sắp xếp theo tên.

        Lỗi được báo rõ ràng ngay khi bắt đầu lặp:
          - FileNotFoundError: thư mục không tồn tại, hoặc không có ảnh hợp lệ nào.
          - NotADirectoryError: đường dẫn trỏ tới một tệp chứ không phải thư mục.
        """
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.image_dir}")
        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"Đường dẫn không phải thư mục: {self.image_dir}")

        image_paths = sorted(
            (
                path
                for path in self.image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ),
            key=lambda path: (path.name.lower(), path.name),
        )
        if not image_paths:
            raise FileNotFoundError(
                f"Thư mục không có ảnh hợp lệ ({', '.join(sorted(self.SUPPORTED_EXTENSIONS))}): "
                f"{self.image_dir}"
            )

        for path in image_paths:
            yield path
