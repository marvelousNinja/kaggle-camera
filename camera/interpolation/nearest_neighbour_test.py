from camera.interpolation.nearest_neighbour import interpolate_bayer
from camera.shared.data import load_sample_image
from skimage import io
from matplotlib import pyplot as plt

def test_interpolate_bayer():
    image = load_sample_image()
    interpolated_image = interpolate_bayer(image)
    assert interpolated_image.shape == image.shape
    # io.imshow(load_sample_image() - interpolated_image)
    # plt.show()
