from pathlib import Path
from typing import Any

class TrashDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("image_size must be greater than 0.")

        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"YOLO model file was not found: {path}")

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "TrashDetector requires the 'ultralytics' package to load a YOLO model."
            ) from error

        self.model: Any = YOLO(str(path))
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

    def detect(self, image: Any) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
        )
        if results is None:
            return []

        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0].item())
                detections.append(
                    {
                        "class_name": self.model.names[class_id],
                        "confidence": float(box.conf[0].item()),
                        "bbox": [float(coordinate) for coordinate in box.xyxy[0].tolist()],
                    }
                )

        return detections