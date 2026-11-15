"""Chạy nhận diện rác hàng loạt trên thư mục ảnh thử nghiệm và lưu kết quả CSV/ảnh.

Mô hình phát hiện đối tượng được truy cập qua Roboflow Workflow (xem
modules/trash_detector.py). Cần đặt biến môi trường ROBOFLOW_API_KEY trước khi chạy.

Ví dụ:
    python run_detection.py --all
    python run_detection.py --limit 5 --conf 0.25
    python run_detection.py --reference
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from modules.data_streamer import DataStreamer
from modules.trash_detector import TrashDetector


ROOT = Path(__file__).resolve().parent


def _find_image_dir() -> Path:
    """Tìm thư mục ảnh thử nghiệm, chấp nhận cả 'data' và 'Data' (Linux/Mac phân biệt hoa thường)."""
    for name in ("data", "Data"):
        candidate = ROOT / name / "test_images"
        if candidate.is_dir():
            return candidate
    return ROOT / "data" / "test_images"


IMAGE_DIR = _find_image_dir()

# Nhãn đúng do nhóm tự gán, dùng cho chế độ --reference.
# Chỉ những ảnh thực sự có trong thư mục test_images mới được dùng;
# thêm tên ảnh và nhóm đúng vào đây khi gán nhãn thêm.
REFERENCE_LABELS = {
    "image_0001.jpg": "Thủy tinh",
    "image_0363.jpg": "Thủy tinh",
    "image_0726.jpg": "Nhựa",
    "image_1089.jpg": "Kim loại",
    "image_1452.jpg": "Giấy",
    "image_1814.jpg": "Giấy",
    "image_2177.jpeg": "Giấy",
    "image_2540.jpg": "Nhựa",
    "image_2903.jpg": "Nhựa",
    "image_3266.jpg": "Nhựa",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nhận diện rác hàng loạt (Roboflow Workflow) và lưu kết quả CSV/ảnh."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help="xử lý toàn bộ ảnh trong thư mục test_images",
    )
    selection.add_argument(
        "--reference",
        action="store_true",
        help="chỉ xử lý các ảnh đã có nhãn đúng trong REFERENCE_LABELS",
    )
    selection.add_argument(
        "--limit",
        type=int,
        default=10,
        help="số ảnh lấy ngẫu nhiên (mặc định: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="hạt giống để lấy mẫu lặp lại được (mặc định: ngẫu nhiên)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="lưu ảnh đã đánh dấu (luôn bật khi lấy mẫu hoặc dùng --reference)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="ngưỡng độ tin cậy (mặc định 0.30)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="kích thước ảnh gửi đi sau khi letterbox (mặc định 960)",
    )
    return parser.parse_args(argv)


def select_images(args: argparse.Namespace) -> tuple[list[Path], bool]:
    """Chọn danh sách ảnh cần xử lý. Trả về (danh sách ảnh, có lưu ảnh đánh dấu không)."""
    # DataStreamer lo việc liệt kê ảnh (theo tên) và báo lỗi nếu thư mục không tồn tại hoặc trống
    image_paths = list(DataStreamer(str(IMAGE_DIR)).stream())

    if args.reference:
        by_name = {path.name.lower(): path for path in image_paths}
        found = [name for name in REFERENCE_LABELS if name.lower() in by_name]
        missing = [name for name in REFERENCE_LABELS if name.lower() not in by_name]
        if not found:
            raise FileNotFoundError(
                "Không có ảnh nào trong REFERENCE_LABELS nằm trong thư mục: " f"{IMAGE_DIR}"
            )
        if missing:
            print(
                "Cảnh báo: bỏ qua các ảnh chưa có trong thư mục: " + ", ".join(missing),
                file=sys.stderr,
            )
        return [by_name[name.lower()] for name in found], True

    if args.all:
        return image_paths, args.save_images

    if args.limit <= 0:
        raise ValueError("--limit phải lớn hơn 0.")
    selected = random.Random(args.seed).sample(
        image_paths, min(args.limit, len(image_paths))
    )
    return selected, True


def draw_detections(image: Any, detections: list[dict[str, Any]]) -> Any:
    """Vẽ khung và nhãn tiếng Việt lên ảnh.

    Vẽ bằng PIL vì cv2.putText không hiển thị được dấu tiếng Việt.
    Đầu vào và đầu ra đều là mảng BGR của OpenCV để dùng tiếp với cv2.imwrite.
    """
    ve = TrashDetector.ve_ket_qua(image, detections)
    return np.ascontiguousarray(np.array(ve)[..., ::-1])


def run(args: argparse.Namespace) -> Path:
    """Chạy nhận diện trên các ảnh đã chọn, ghi CSV và (tùy chọn) ảnh đánh dấu."""
    if not args.all and not args.reference and args.seed is None:
        args.seed = random.SystemRandom().randrange(0, 2**32)

    image_paths, save_images = select_images(args)
    output_dir = ROOT / "results" / "automatic_run" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = TrashDetector(confidence_threshold=args.conf, image_size=args.imgsz)
    csv_path = output_dir / "detection_summary.csv"
    annotated_dir = output_dir / "annotated_images"
    if save_images:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    reference_hits = 0
    reference_exact = 0
    reference_total = 0
    total_seconds = 0.0

    with csv_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "image",
                "expected_class",
                "predicted_classes",
                "expected_class_detected",
                "detection_count",
                "confidences",
                "processing_seconds",
                "annotated_image",
            ],
        )
        writer.writeheader()

        for index, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Không đọc được ảnh: {image_path}")

            # Chỉ đo thời gian của bước nhận diện (gồm cả độ trễ mạng khi gọi dịch vụ)
            started = time.perf_counter()
            detections = detector.detect(image)
            elapsed = time.perf_counter() - started
            total_seconds += elapsed

            classes = [str(item["class_name"]) for item in detections]
            class_counts = Counter(classes)
            expected = REFERENCE_LABELS.get(image_path.name.lower())
            expected_detected = expected in classes if expected is not None else None
            if expected is not None:
                reference_total += 1
                reference_hits += bool(expected_detected)
                reference_exact += bool(classes) and all(c == expected for c in classes)

            annotated_name = ""
            if save_images:
                annotated_name = f"annotated_{image_path.name}"
                annotated = draw_detections(image.copy(), detections)
                annotated_path = annotated_dir / annotated_name
                if not cv2.imwrite(str(annotated_path), annotated):
                    raise OSError(f"Không lưu được ảnh đánh dấu: {annotated_path}")

            writer.writerow(
                {
                    "image": image_path.name,
                    "expected_class": expected or "",
                    "predicted_classes": "; ".join(classes),
                    "expected_class_detected": (
                        expected_detected if expected_detected is not None else ""
                    ),
                    "detection_count": len(detections),
                    "confidences": "; ".join(
                        f"{float(item['confidence']):.4f}" for item in detections
                    ),
                    "processing_seconds": f"{elapsed:.4f}",
                    "annotated_image": annotated_name,
                }
            )

            summary = (
                ", ".join(f"{name}: {count}" for name, count in sorted(class_counts.items()))
                if class_counts
                else "không phát hiện vật nào"
            )
            print(
                f"[{index}/{len(image_paths)}] {image_path.name}: "
                f"{len(detections)} vật ({summary}) - {elapsed:.3f}s"
            )

    print(f"\nKết quả lưu tại: {output_dir}")
    print(f"CSV: {csv_path}")
    print(f"Số ảnh đã xử lý: {len(image_paths)}")
    print(f"Tổng thời gian nhận diện (gồm độ trễ mạng): {total_seconds:.2f}s")
    if image_paths:
        print(f"Thời gian trung bình mỗi ảnh: {total_seconds / len(image_paths):.2f}s")
    if not args.all and not args.reference:
        print(f"Hạt giống lấy mẫu: {args.seed}")
    if reference_total:
        print(
            "Kết quả theo nhãn đúng: "
            f"{reference_hits}/{reference_total} ảnh có phát hiện đúng nhóm mong đợi; "
            f"{reference_exact}/{reference_total} ảnh chỉ phát hiện đúng nhóm mong đợi."
        )
    return output_dir


def main() -> int:
    try:
        run(parse_args())
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError, RuntimeError) as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
