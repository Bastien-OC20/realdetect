import unittest
from PIL import Image
from backend.detection import detect_objects

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestDetection(unittest.TestCase):
    def test_detect_objects(self):
        image = Image.new('RGB', (100, 100))
        detections = detect_objects(image)
        self.assertIsInstance(detections, list)

if __name__ == '__main__':
    unittest.main()