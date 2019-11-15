from keras import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation
from keras.models import Sequential, load_model
from keras.regularizers import l2
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def build_dense_model2(images_train, labels_train, images_test, labels_test):
    images_train = images_train.reshape([images_train.shape[0], 400])
    images_test = images_test.reshape([images_test.shape[0], 400])

    labels_train = to_categorical(labels_train, 26)
    labels_test = to_categorical(labels_test, 26)

    input_layer = Input(shape=images_train[0].shape)

    layer1 = Dense(units=200, activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.3)(layer1)
    layer2 = Dense(units=100, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.3)(layer2)

    output_layer = Dense(units=26, activation='softmax')(dropout2)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=["accuracy"])

    model.summary()
    model.fit(images_train, labels_train, epochs=100, batch_size=64, verbose=2)

    history = model.evaluate(images_test, labels_test)
    return history[1]


def build_CNN_model(images_train, labels_train, images_test, labels_test):
    # Function Parameter parsing
    # images_train = images_train.reshape([images_train.shape[0], 400])
    # images_test = images_test.reshape([images_test.shape[0], 400])
    labels_train = to_categorical(labels_train, 26)
    labels_test = to_categorical(labels_test, 26)

    new_images_train = np.expand_dims(images_train, axis=-1)
    new_images_test = np.expand_dims(images_test, axis=-1)

    print("\nLabels train: ", labels_train.shape)
    print(labels_train)
    print("\nLabels test: ", labels_test.shape)
    print(labels_test)
    print("\nImages train: ", images_train.shape)
    print("\nImages test: ", images_test.shape)
    print("\nLen(Images_train): ", len(images_train))
    print("\nNew images train: ", new_images_train.shape, "\n")
    print("\nLen(new_images_train): ", len(new_images_train), "\n")

    # Hyperparameters
    weight_decay = 0.005
    dropout = 0.30
    epoch_number = 50
    batch_size = 64

    input_dimension = (20, 20, 1)
    output_dimension = 26

    # CNN
    model = Sequential()

    # Convblock 1
    model.add(Conv2D(64, kernel_size=3, padding='same', kernel_regularizer=l2(
        weight_decay), input_shape=input_dimension))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=3, padding='same',
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convblock 2
    model.add(Conv2D(128, kernel_size=3, padding='same',
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=3, padding='same',
                     kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Dense output
    # Flatten to get proper dimensions for output
    model.add(Flatten())

    model.add(Dense(2048, kernel_regularizer=l2(weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(output_dimension, activation="softmax"))

    Adam(learning_rate=0.001, beta_1=0.9,
         beta_2=0.999, amsgrad=False)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # Train model
    model.fit(new_images_train, labels_train,
              epochs=epoch_number, batch_size=batch_size, verbose=2)

    # Test model
    model_result = model.evaluate(new_images_test, labels_test)

    return model_result, model


def loadModel(modelPath, images_test, labels_test):
    images_test = np.expand_dims(images_test, axis=-1)
    labels_test = to_categorical(labels_test, 26)

    model = load_model(modelPath)

    # Test the loaded model
    print(model.evaluate(images_test, labels_test))
    return model


def cnnPredict(model, image):
    predictions = model.predict(image)
    print(predictions)
