from keras import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation
from keras.models import Sequential, load_model
from keras.regularizers import l2
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def build_CNN_model(images_train, labels_train, images_test, labels_test):
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
    model.fit(images_train, labels_train,
              epochs=epoch_number, batch_size=batch_size, verbose=2)

    # Test model
    model_result = model.evaluate(images_test, labels_test)

    return model_result, model


def load_cnn_model(modelPath):
    model = load_model(modelPath)
    return model


def cnn_predict_single_im(model, image, label, LETTERS):
    predictions = model.predict(image)
    for i in range(len(LETTERS)):
        print("Letter ", LETTERS[i], ": ", predictions[0][i])
    print("\nMost likely letter: ",
          LETTERS[np.argmax(predictions[0])])
    print("Actual letter: ", LETTERS[(int(label[0]))])
