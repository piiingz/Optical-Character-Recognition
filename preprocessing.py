from glob import glob
from sklearn.model_selection import train_test_split
from feature_extraction import otsu_filter
import string
import numpy as np
from PIL import Image
from models.SVM import svc_model
from sklearn.metrics import accuracy_score
from skimage.transform import rotate
import random
from models.CNN import build_CNN_model
from keras.utils import to_categorical

import matplotlib.pyplot as plt

CHAR_PATH = "dataset/chars74k-lite/"
LETTERS = [i for i in string.ascii_lowercase]


def pre_processing(path):
    image = Image.open(path)
    data = np.asarray(image)/255
    # data = otsu_filter(data)

    # data = data.reshape(1, 400)
    return data


def split_data(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.20, random_state=23)
    return X_train, X_test, y_train, y_test


def load_images():
    total_files = len(glob(CHAR_PATH + "**/*.jpg"))
    images = np.zeros([total_files, 20, 20])
    labels = np.zeros(total_files)

    n = 0

    for i in range(len(LETTERS)):
        print(LETTERS[i])
        paths = glob(CHAR_PATH + LETTERS[i] + "/*.jpg")
        for path in paths:

            image_pp = pre_processing(path)
            images[n] = image_pp
            labels[n] = i

            n += 1

    return images, labels


def rotate_pictures(images, labels):
    extended_images = np.zeros(
        [4*images.shape[0], images.shape[1], images.shape[2]])
    extended_images[:images.shape[0]] = images

    extended_labels = np.zeros([4*labels.shape[0]])
    extended_labels[:labels.shape[0]] = labels

    for i in range(3):
        degree = random.randint(-25, 25)
        extended_labels[labels.shape[0]*(i+1):labels.shape[0]*(i+2)] = labels
        for j in range(len(images)):
            extended_images[images.shape[0] *
                            (1+i) + j] = rotate(images[j], degree)

    return extended_images, extended_labels


def SaveAndLoggCNN(model):
    history = model.history.history
    print("\nKEYS: ", history.keys())
    # Loss plot
    plt.figure(figsize=(12, 8))
    plt.plot(history["loss"], label="Training loss")
    plt.legend()
    plt.show()

    # Accuracy plot
    plt.figure(figsize=(12, 8))
    plt.plot(history["accuracy"], label="Training accuracy")
    plt.legend()
    plt.show()

    # Save model
    model.save("./models/cnn_trained_model.h5")


def main():
    images, labels = load_images()
    im_train, im_test, label_train, label_test = split_data(images, labels)
    # im_train, label_train = rotate_pictures(im_train, label_train)

    # predictions = svc_model(im_train, label_train, im_test)
    cnn_accuracy, cnn_model = build_CNN_model(
        im_train, label_train, im_test, label_test)

    # print("Accuracy: ", accuracy_score(label_test, predictions))
    print("Accuracy: ", cnn_accuracy)

    SaveAndLoggCNN(cnn_model)


main()


# test_image = np.array([im_test[1]])

# show_image = test_image[0].reshape(20, 20)
#
# fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(20, 20), sharex=True, sharey=True)
#
# ax1.imshow(show_image, cmap=plt.cm.gray)
#
# fig.tight_layout()
# plt.show()
#
# print(LETTERS[int(label_test[1])])


#
# image = Image.open(image_path)
# image.show()
#
# data = np.asarray(image)/255
# print(data)
#
# edges = feature.canny(data, sigma=1)
# print(edges)
#
# # display results
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 20),
#                                     sharex=True, sharey=True)
#
# ax1.imshow(data, cmap=plt.cm.gray)
# ax1.axis('off')
# ax1.set_title('noisy image', fontsize=20)
#
# ax2.imshow(edges, cmap=plt.cm.gray)
# ax2.axis('off')
# ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)
#
# print(ax2)
# fig.tight_layout()
#
# plt.show()
