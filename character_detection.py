import numpy as np
from glob import glob
from preprocessing import pre_processing
# from models.CNN import loadModel


DETECTION_PATH = "dataset/detection-images/"


def load_detection_images():
    total_files = glob(DETECTION_PATH + "*.jpg")

    img1 = pre_processing(total_files[0])
    img2 = pre_processing(total_files[1])

    return img1, img2


def is_character(window, threshold):
    """
    Returns True if sliding window has more than a threshold of non-whitespace
    """
    imgarray = window.flatten()
    total_pixels = imgarray.size
    not_whitespace = np.sum(np.array([(not x == 1) for x in imgarray]))
    return not_whitespace/total_pixels > threshold


def pick_classifiaction(x, y, prediction, dic, h, w):
    """
    x, y:           coordinates of window predicted
    prediction:     all probabilities of the window containing a letter predicted  
    dicationary:    { index (int): ((x,y), [prob_array]) }
    """
    for key, value in dic:
        x_curr, y_curr = value[0]
        if x in range(x_curr, x_curr + h) or y in range(y_curr, y_curr + w):
            if np.amax(prediction) > np.amax(value[1]):
                dic[key] = ((x, y), prediction)
        else:
            n = len(dic)
            dic[n] = ((x, y), prediction)
    return dic


def sliding_window(image, dictionary, step, h, w, model):
    for x in range(0, len(image[1, :])-h+1, step):
        for y in range(0, len(image[:, 1])-w+1, step):
            window = image[x:x+h, y:y+w]
            if is_character(window, 0.8):
                print('lokomotiv!')
                # # dimension_image = add_dimensions()
                # prediction = model.predict(window)
                # dictionary, n = pick_classifiaction(
                #     x, y, prediction, dictionary, h, w)


def main():
    dictionary = {}
    test_img, pred_img = load_detection_images()

    step = 5
    h = 20  # height of sliding window
    w = 20  # width of sliding window
    # model = loadModel()

    sliding_window(test_img, dictionary, step, h, w, None)


main()
