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
from models.CNN import build_CNN_model, load_cnn_model, cnn_predict_single_im
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


def save_cnn_plot_history(model, path):
    print("\nPlotting CNN training history..")
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
    print("\nSaving model to: ", path)
    model.save(path)


def cnn_preprocess(im_train, im_test, label_train, label_test):
    # Set labels to correct output dim
    cnn_labels_train = to_categorical(label_train, 26)
    cnn_labels_test = to_categorical(label_test, 26)

    # Keras.fit expects 4D array
    cnn_im_train = np.expand_dims(im_train, axis=-1)
    cnn_im_test = np.expand_dims(im_test, axis=-1)

    return cnn_im_train, cnn_im_test, cnn_labels_train, cnn_labels_test


def run_preprocess_train_test(train_svm, train_cnn):
    # Load and split dataset
    images, labels = load_images()
    im_train, im_test, label_train, label_test = split_data(images, labels)

    # SVM wih rotation for dataset
    if (train_svm):
        print("\nTraining and testing SVM\n")
        im_train_svm, label_train_svm = rotate_pictures(im_train, label_train)
        predictions = svc_model(im_train_svm, label_train_svm, im_test)
        print("Accuracy on training set: ",
              accuracy_score(label_test, predictions))

    # CNN preprocess
    cnn_im_train, cnn_im_test, cnn_label_train, cnn_label_test = cnn_preprocess(
        im_train, im_test, label_train, label_test)

    if (train_cnn):
        print("\nTraining the CNN\n")
        cnn_accuracy, cnn_model = build_CNN_model(
            cnn_im_train, cnn_label_train, cnn_im_test, cnn_label_test)
        print("Accuracy: ", cnn_accuracy)
        # Save model and logg history
        save_cnn_plot_history(cnn_model, "./models/newly_trained_cnn.h5")

    print("\nRunning the pretrained CNN\n")
    # Load our pretrained CNN-Model
    loaded_model = load_cnn_model(
        "./models/cnn_trained_model.h5")

    print("\nPredicting one image with the CNN:\n")
    cnn_im_test = np.expand_dims(im_test, axis=-1)
    cnn_predict_single_im(loaded_model, cnn_im_test, label_test, LETTERS)

    print("\nCNN accuracy on traning set: \n")
    print(loaded_model.evaluate(cnn_im_test, cnn_label_test))
