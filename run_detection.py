from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2

from modules.trash_detector import TrashDetector


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "test_images"
MODEL_PATH = ROOT / "models" / "best_model.pt"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REFERENCE_LABELS = {
    "image_0001.jpg": "Glass",
    "image_0363.jpg": "Glass",
    "image_0726.jpg": "Plastic",
    "image_1089.jpg": "Metal",
    "image_1452.jpg": "Paper",
    "image_1814.jpg": "Paper",
    "image_2177.jpeg": "Paper",
    "image_2540.jpg": "Plastic",
    "image_2903.jpg": "Plastic",
    "image_3266.jpg": "Plastic",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO trash detection and save CSV/image results."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help="process every image in data/test_images",
    )
    selection.add_argument(
        "--reference",
        action="store_true",
        help="evaluate the 10 images with user-provided reference labels",
    )
    selection.add_argument(
        "--limit",
        type=int,
        default=10,
        help="random sample size (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional seed for repeatable sampling (random by default)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="save annotated images (always enabled for a sample)",
    )
    return parser.parse_args(argv)


def select_images(args: argparse.Namespace) -> tuple[list[Path], bool]:
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {IMAGE_DIR}")

    if args.reference:
        by_name = {path.name.lower(): path for path in image_paths}
        missing = [
            name for name in REFERENCE_LABELS if name.lower() not in by_name
        ]
        if missing:
            raise FileNotFoundError(
                "Reference images missing from the dataset: " + ", ".join(missing)
            )
        return [by_name[name.lower()] for name in REFERENCE_LABELS], True

    if args.all:
        return image_paths, False

    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")
    selected = random.Random(args.seed).sample(
        image_paths, min(args.limit, len(image_paths))
    )
    return selected, True


def draw_detections(
    image: Any, detections: list[dict[str, Any]]
) -> Any:
    for detection in detections:
        x1, y1, x2, y2 = (
            int(round(coordinate)) for coordinate in detection["bbox"]
        )
        label = (
            f"{detection['class_name']} "
            f"{float(detection['confidence']):.2f}"
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 220, 40),
            1,
            cv2.LINE_AA,
        )
    return image


def run(args: argparse.Namespace) -> Path:
    if not args.all and not args.reference and args.seed is None:
        args.seed = random.SystemRandom().randrange(0, 2**32)

    image_paths, save_images = select_images(args)
    output_dir = ROOT / "results" / "automatic_run" / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = TrashDetector(str(MODEL_PATH), confidence_threshold=0.25, image_size=640)
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
                raise ValueError(f"Could not read image: {image_path}")

            started = time.perf_counter()
            detections = detector.detect(image)
            elapsed = time.perf_counter() - started
            total_seconds += elapsed

            classes = [str(item["class_name"]) for item in detections]
            expected = REFERENCE_LABELS.get(image_path.name.lower())
            expected_detected = (
                expected in classes if expected is not None else None
            )
            if expected is not None:
                reference_total += 1
                reference_hits += bool(expected_detected)
                reference_exact += bool(classes) and all(
                    class_name == expected for class_name in classes
                )

            annotated_name = ""
            if save_images:
                annotated_name = f"annotated_{image_path.name}"
                annotated = draw_detections(image.copy(), detections)
                annotated_path = annotated_dir / annotated_name
                if not cv2.imwrite(str(annotated_path), annotated):
                    raise OSError(f"Could not save annotated image: {annotated_path}")

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
            print(
                f"[{index}/{len(image_paths)}] {image_path.name}: "
                f"{', '.join(classes) if classes else 'no detections'} "
                f"({elapsed:.3f}s)"
            )

    print(f"\nResults saved to: {output_dir}")
    print(f"CSV: {csv_path}")
    print(f"Images processed: {len(image_paths)}")
    print(f"Total inference time: {total_seconds:.2f}s")
    if not args.all and not args.reference:
        print(f"Sampling seed: {args.seed}")
    if reference_total:
        print(
            "Reference-label results: "
            f"{reference_hits}/{reference_total} images contain the expected class; "
            f"{reference_exact}/{reference_total} contain only the expected class."
        )
    return output_dir


def main() -> int:
    try:
        run(parse_args())
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
