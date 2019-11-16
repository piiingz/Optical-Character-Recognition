import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from preprocessing import pre_processing
from models.CNN import load_cnn_model
from PIL import Image, ImageDraw


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


def pick_classifiaction(x0, y0, prediction, dic, h, w):
    """
    Keeps track of the most probable letter of all frames recognized in an image.
    If there are overlapping frames, the frame classified with the best probability
    is chosen.

    x, y:           coordinates of window predicted
    prediction:     all probabilities of the window containing a letter predicted  
    dicationary:    { index (int): ((x,y), [prob_array]) }
    """

    n = len(dic)

    # first iteration
    if(n < 0):
        dic[0] = ((x0, y0), prediction)
        return dic

    flag = False

    for i in range(n):
        # get coordinates for frame in dic
        x1, y1 = dic[i][0]

        # if the window is overlapping an existing frame
        if ((abs(x0-x1) < h/2) and (abs(y0-y1) < w/2)):
            flag = True

            # choose the frame with best prediction
            if max(prediction) > max(dic[i][1]):
                dic[i] = ((x0, y0), prediction)

    # if the window does not overlap with any frames already found
    if not flag:

        # add new frame to dic
        dic[n] = ((x0, y0), prediction)
    return dic


def sliding_window(image, dictionary, step, h, w, model):
    """
    Sliding window that checks if an image contains a charater and predicts
    which letter it is
    """
    for y in range(0, len(image[:, 1])-w+1, step):
        for x in range(0, len(image[1, :])-h+1, step):
            window = image[y:y+w, x:x+h]
            if is_character(window, 0.8):
                window = window.reshape((1,) + window.shape + (1,))
                prediction = model.predict(window)[0]
                dictionary = pick_classifiaction(
                    x, y, prediction, dictionary, h, w)
    return dictionary


def show_letter(image):
    """
    Show the current letter recognized for debugging purposes
    """
    fig, (ax1) = plt.subplots(nrows=1, ncols=1,
                              figsize=(20, 20), sharex=True, sharey=True)
    ax1.imshow(image, cmap=plt.cm.gray)
    fig.tight_layout()
    plt.show()


def draw_ocr(imagepath, result_dict, h, w):
    """
    Draw bounding box for letters recognized in an image
    """
    image = Image.open(imagepath).convert('RGB')
    draw = ImageDraw.Draw(image)
    for key in result_dict:
        x, y = result_dict[key][0]
        magenta = '#FF00FF'
        draw.line((x, y) + (x+h, y), fill=magenta)
        draw.line((x, y) + (x, y+w), fill=magenta)
        draw.line((x+h, y) + (x+h, y+w), fill=magenta)
        draw.line((x, y+w) + (x+h, y+w), fill=magenta)
    del draw
    image.show()


def run_character_detection():
    test_img, pred_img = load_detection_images()

    step = 5
    h = 20  # height of sliding window
    w = 20  # width of sliding window
    model = load_cnn_model('./models/cnn_trained_model.h5')

    test_result = sliding_window(test_img, {}, step, h, w, model)
    result = sliding_window(pred_img, {}, step, h, w, model)

    draw_ocr(DETECTION_PATH + 'detection-1.jpg', test_result, h, w)
    draw_ocr(DETECTION_PATH + 'detection-2.jpg', result, h, w)
