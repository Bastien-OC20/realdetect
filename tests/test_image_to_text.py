import unittest
from PIL import Image
from backend.image_to_text import generate_caption

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestImageToText(unittest.TestCase):
    def test_generate_caption(self):
        image = Image.new('RGB', (100, 100))
        caption = generate_caption(image)
        self.assertIsInstance(caption, str)

if __name__ == '__main__':
    unittest.main()