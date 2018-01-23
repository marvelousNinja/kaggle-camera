def crop_center(size, image):
    center_x = image.shape[0] // 2 - 1
    center_y = image.shape[1] // 2 - 1
    top_x, top_y = center_x - size // 2, center_y - size // 2
    return image[top_x:top_x + size, top_y:top_y + size]
