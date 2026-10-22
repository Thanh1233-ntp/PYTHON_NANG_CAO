import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from run_detection import parse_args, select_images


class RunDetectionTests(unittest.TestCase):
    def test_sampling_is_random_by_default(self) -> None:
        self.assertIsNone(parse_args([]).seed)

    def test_explicit_seed_repeats_the_same_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            for index in range(10):
                (image_dir / f"image_{index:02}.jpg").touch()

            args = Namespace(all=False, reference=False, limit=4, seed=17)
            with patch("run_detection.IMAGE_DIR", image_dir):
                first_sample, _ = select_images(args)
                second_sample, _ = select_images(args)

        self.assertEqual(first_sample, second_sample)
        self.assertEqual(len(first_sample), 4)


if __name__ == "__main__":
    unittest.main()
