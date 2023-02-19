import numpy as np
import unittest

class TestVignetteCorners(unittest.TestCase):
    def test_black_vignette(self):
        class TestImage:
            def __init__(self, image_data):
                self.image = image_data
        image_data = np.ones((5, 5, 3)) * 100
        test_image = TestImage(image_data)
        test_image.vignette_corners(dim_factor=0.5, is_black=True)
        expected_output = np.array([[50., 50., 50.],
                                    [50., 50., 50.],
                                    [50., 50., 50.],
                                    [50., 50., 50.],
                                    [50., 50., 50.]])
        self.assertTrue(np.array_equal(test_image.image, expected_output))

    def test_white_vignette(self):
        class TestImage:
            def __init__(self, image_data):
                self.image = image_data
        image_data = np.ones((5, 5, 3)) * 100
        test_image = TestImage(image_data)
        test_image.vignette_corners(dim_factor=0.5, is_black=False)
        expected_output = np.array([[150., 150., 150.],
                                    [150., 150., 150.],
                                    [150., 150., 150.],
                                    [150., 150., 150.],
                                    [150., 150., 150.]])
        self.assertTrue(np.array_equal(test_image.image, expected_output))

if __name__ == '__main__':
    unittest.main()
