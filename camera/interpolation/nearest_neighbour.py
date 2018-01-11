import numpy as np
# assuming it is in RGB
# G  B
# R  G
def interpolate_bayer(source_image):
    image = np.array(source_image)
    red = image[1::2, 0::2, 0]
    # TODO AS: Disabling interpolation for all channels except red
    # green = image[1::2, 1::2, 1]
    # blue = image[0::2, 1::2, 2]

    # Green 1
    image[0::2, 0::2, 0] = red
    # image[0::2, 0::2, 2] = blue

    # Green 2
    image[1::2, 1::2, 0] = red
    # image[1::2, 1::2, 2] = blue

    # Blue
    image[0::2, 1::2, 0] = red
    # image[0::2, 1::2, 1] = green

    # Red
    # image[1::2, 0::2, 1] = green
    # image[1::2, 0::2, 2] = blue

    return image
