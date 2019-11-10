import numpy as np
from glob import glob
from preprocessing import pre_processing


DETECTION_PATH = "dataset/detection-images/"


def load_detection_images():
    total_files = glob(DETECTION_PATH + "*.jpg")

    # img1 = Image.open(total_files[0])
    # img2 = Image.open(total_files[1])
    #
    # img1 = np.asarray(img1) / 255
    # img2 = np.asarray(img2) / 255

    img1 = pre_processing(total_files[0])
    img2 = pre_processing(total_files[1])

    return img1, img2


def is_character(imgarray):
    """
    Returns True if sliding window has more than 50% False values
    """
    array_size = imgarray.shape
    truthy_values = np.sum(imgarray)
    falsy_values = array_size-truthy_values
    return falsy_values/array_size > 0.5


def sliding_window(image):
    h = 20
    w = 20
    for i in range(len(image[1, :])):
        for j in range(len(image[:, 1])):

            # load_detection_images()
