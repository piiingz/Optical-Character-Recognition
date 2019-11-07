from skimage import feature
from scipy import ndimage, misc
from glob import glob
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

import string
import numpy as np
from PIL import Image

BASE_PATH = "dataset/chars74k-lite/"
LETTERS = [i for i in string.ascii_lowercase]


def pre_processing(path):
    image = Image.open(path)
    data = np.asarray(image)/255
    data = data.reshape(1, 400)
    return data


def split_data(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.20, random_state=23)
    return X_train, X_test, y_train, y_test


def load_images():
    total_files = len(glob("dataset/chars74k-lite/**/*.jpg"))
    images = np.zeros([total_files, 400])
    labels = np.zeros(total_files)

    n = 0

    for i in range(len(LETTERS)):
        print(LETTERS[i])
        paths = glob(BASE_PATH + LETTERS[i] + "/*.jpg")
        for path in paths:

            image_pp = pre_processing(path)
            images[n] = image_pp
            labels[n] = i

            n += 1

    return images, labels


def svc_model(images, labels, test_image):
    # # Create a pipeline that finds the optimal kernel function for the SVC
    # scaler_and_classifier = [
    #     ('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
    # pipeline = Pipeline(scaler_and_classifier)
    #
    # # Parameters to evaluate
    # parameters = {'SVM__C': [0.001, 0.1, 100, 10e5],
    #               'SVM__gamma': [10, 1, 0.1, 0.01]}
    #
    # # Perform extensive search on all parameters to find the optimal parameter combination
    # svc = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    print("Predicting.....")
    svc = SVC(gamma="scale", kernel='rbf')
    svc.fit(images, labels)
    prediction = svc.predict(test_image)
    print("Done predicting")

    return prediction


def find_accuracy(predictions, labels):
    counter = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            counter += 1

    return counter / len(predictions)


def main():
    images, labels = load_images()
    im_train, im_test, label_train, label_test = split_data(images, labels)

    test_image = np.array([im_test[1]])

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

    predictions = svc_model(im_train, label_train, im_test)

    accuracy = find_accuracy(predictions, label_test)

    print(accuracy)


main()





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