from camera.shared.data import load_sample_image
from camera.interpolation.nearest_neighbour import interpolate_bayer
from camera.interpolation.metrics import difference
import numpy as np

def test_difference():
    image = load_sample_image()
    interpolated_image = interpolate_bayer(image)
    diff = difference(image, interpolated_image, quantization=2, threshold=3)
    assert all(np.unique(diff) == np.array([-3, -2, -1, 0, 1, 2, 3]))
