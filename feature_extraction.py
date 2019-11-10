from skimage import feature, filters


def otsu_filter(image):
    threshold = filters.threshold_otsu(image)
    binary = image > threshold
    return binary


def edge_detection(image):
    return feature.canny(image)
