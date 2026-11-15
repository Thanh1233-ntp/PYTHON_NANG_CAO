import numpy as np


class ImageProcessor:
    def __init__(self, target_size: int = 640) -> None:
        if target_size <= 0:
            raise ValueError("target_size must be greater than 0.")
        self.target_size = target_size

    def validate(self, image: object) -> bool:
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a NumPy ndarray.")
        if image.ndim != 3:
            raise ValueError("image must have three dimensions (height, width, channels).")
        if any(dimension <= 0 for dimension in image.shape):
            raise ValueError("image dimensions must all be greater than 0.")
        return True

    def process(self, image: object) -> np.ndarray:
        self.validate(image)
        height, width, _ = image.shape
        scale = min(self.target_size / height, self.target_size / width)
        resized_height = max(1, round(height * scale))
        resized_width = max(1, round(width * scale))

        rows = np.linspace(0, height - 1, resized_height).astype(np.intp)
        columns = np.linspace(0, width - 1, resized_width).astype(np.intp)
        resized = image[rows[:, np.newaxis], columns]

        processed = np.full(
            (self.target_size, self.target_size, image.shape[2]),
            fill_value=114,
            dtype=image.dtype,
        )
        top = (self.target_size - resized_height) // 2
        left = (self.target_size - resized_width) // 2
        processed[top : top + resized_height, left : left + resized_width] = resized
        return processed