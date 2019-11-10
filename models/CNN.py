from keras.layers import Dense, Input, Dropout
from keras import Model
from keras.utils import to_categorical
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def build_dense_model(images_train, labels_train, images_test, labels_test):
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

    model.summary()
    model.fit(images_train, labels_train, epochs=100, batch_size=64, verbose=2)

    history = model.evaluate(images_test, labels_test)

    return history[1]
