import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from modules.image_processor import ImageProcessor
from modules.trash_detector import TrashDetector


class FakeBox:
    def __init__(self, class_id: int, confidence: float, coordinates: list[float]) -> None:
        self.cls = np.array([class_id])
        self.conf = np.array([confidence])
        self.xyxy = np.array([coordinates])

class FakeResult:
    def __init__(self, boxes: list[FakeBox]) -> None:
        self.boxes = boxes

class FakeYOLO:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.names = {0: "Glass", 1: "Plastic"}
        self.results: list[FakeResult] = []
        self.last_source: np.ndarray | None = None
        self.last_verbose: bool | None = None

    def predict(self, source: np.ndarray, verbose: bool = True) -> list[FakeResult]:
        self.last_source = source
        self.last_verbose = verbose
        return self.results

class ImageProcessorTests(unittest.TestCase):
    def test_validate_accepts_nonempty_three_dimensional_array(self) -> None:
        self.assertTrue(ImageProcessor().validate(np.zeros((4, 5, 3))))

    def test_validate_rejects_none_and_wrong_type(self) -> None:
        processor = ImageProcessor()
        for image in (None, [[1, 2, 3]]):
            with self.subTest(image=image), self.assertRaisesRegex(ValueError, "NumPy ndarray"):
                processor.validate(image)

    def test_validate_rejects_wrong_dimensions_and_empty_images(self) -> None:
        processor = ImageProcessor()
        for image in (np.zeros((4, 5)), np.zeros((0, 5, 3))):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                processor.validate(image)

    def test_process_letterboxes_small_and_large_images_to_target_size(self) -> None:
        processor = ImageProcessor(target_size=8)
        small = processor.process(np.zeros((2, 3, 3), dtype=np.uint8))
        large = processor.process(np.zeros((1200, 1600, 3), dtype=np.uint8))

        self.assertEqual(small.shape, (8, 8, 3))
        self.assertEqual(large.shape, (8, 8, 3))
        np.testing.assert_array_equal(small[0], np.full((8, 3), 114, dtype=np.uint8))
        np.testing.assert_array_equal(large[0], np.full((8, 3), 114, dtype=np.uint8))
        np.testing.assert_array_equal(small[1:6], np.zeros((5, 8, 3), dtype=np.uint8))
        np.testing.assert_array_equal(large[1:7], np.zeros((6, 8, 3), dtype=np.uint8))

class TrashDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "best_model.pt"
        self.model_path.touch()
        self.fake_yolo = FakeYOLO
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = self.fake_yolo
        self.module_patch = patch.dict(sys.modules, {"ultralytics": fake_ultralytics})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)
        self.detector = TrashDetector(str(self.model_path))

    def test_detect_returns_all_boxes_in_the_documented_format(self) -> None:
        self.detector.model.results = [
            FakeResult(
                [
                    FakeBox(0, 0.92, [1, 2, 30, 40]),
                    FakeBox(1, 0.81, [5, 6, 50, 60]),
                ]
            )
        ]

        detections = self.detector.detect(np.zeros((64, 64, 3)))

        self.assertEqual(
            detections,
            [
                {
                    "class_name": "Glass",
                    "confidence": 0.92,
                    "bbox": [1.0, 2.0, 30.0, 40.0],
                },
                {
                    "class_name": "Plastic",
                    "confidence": 0.81,
                    "bbox": [5.0, 6.0, 50.0, 60.0],
                },
            ],
        )
        self.assertIsInstance(detections, list)
        for detection in detections:
            self.assertEqual(set(detection), {"class_name", "confidence", "bbox"})
            self.assertIsInstance(detection["confidence"], float)
            self.assertEqual(len(detection["bbox"]), 4)

    def test_detect_returns_empty_list_when_no_boxes_are_found(self) -> None:
        self.detector.model.results = [FakeResult([])]

        self.assertEqual(self.detector.detect(np.zeros((64, 64, 3))), [])

    def test_missing_model_file_has_clear_error(self) -> None:
        missing_path = Path(self.temp_dir.name) / "missing.pt"
        with self.assertRaisesRegex(FileNotFoundError, "YOLO model file was not found"):
            TrashDetector(str(missing_path))

if __name__ == "__main__":
    unittest.main()