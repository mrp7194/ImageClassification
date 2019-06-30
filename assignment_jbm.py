import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
from sklearn import preprocessing
import keras
from keras import optimizers
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


DATADIR = "data/"
CATEGORIES = ["defects", "healthy"]
IMG_SIZE = 50


def create_training_data():
    training_data = []
    for category in CATEGORIES:  # do defects and healthy

        path = os.path.join(DATADIR,category)  # create path to defects and healthy
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=defects 1=healthy

        for img in tqdm(os.listdir(path)):  # iterate over each image per defects and healthy
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

    return training_data


def convert_image_to_array(image_path):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # convert to array
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def create_model(X,y,save=False):
    model = Sequential()
    model.add(Conv2D(512, (3, 3), input_shape=X.shape[1:]))
    model.add(Conv2D(512, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = optimizers.Adam(lr = 0.01)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=1)

    if save:
        model.save(DATADIR+"model.h5")

    return model


def prepare_data_for_training():
    training_data = create_training_data()
    random.shuffle(training_data)

    X = np.array([a for a, b in training_data])  # features
    y = np.array([b for a, b in training_data])  # label
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    print(X.shape)
    X = X / 255  # Normalizing the features between 0 and 1

    return X,y


def image_classification_main(train=False,test_image_path=None):
    if train:
        X,y = prepare_data_for_training()
        model = create_model(X,y,save=True)
    else:
        model = load_model(DATADIR+"model.h5")

    X_test = convert_image_to_array(test_image_path)
    pred = model.predict(X_test)

    if pred==1:
        print("the given image is classified as healthy")
    else:
        print("the given image is classified as defected")

    keras.backend.clear_session()


if __name__=='__main__':
    image_classification_main(test_image_path="data/defects/IMG20180905143945.jpg")
