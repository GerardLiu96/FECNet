import os
from PIL import Image
import numpy as np
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
import keras.backend as K
from keras.models import load_model
import FEC


height=224
width=224
channels=3
batch_size=32
nb_classes = 7
batch_num = 16

train_path = './RAF/Train/'
test_path = './RAF/Test/'


def readData(train_path,test_path):
    subpath1 = []
    imdb1 = []
    for item in os.listdir(train_path):
        subpath1.append(os.path.join(train_path, item))

    for items1 in subpath1:
        for items2 in os.listdir(items1):
            p = os.path.join(items1, items2)
            imdb1.append(p)

    trainX = []
    trainY = []
    for imagepath in imdb1:
        image = Image.open(imagepath)
        img = image.resize((224, 224))
        trainX.append(np.array(img))
        image_class = imagepath.split('/')[-2]
        trainY.append(int(image_class))

    Xtrain = np.array(trainX)
    Y_train = np.array(trainY)
    X_train = Xtrain.reshape(Xtrain.shape[0],224,224,3)

    subpath2 = []
    imdb2 = []
    for item in os.listdir(test_path):
        subpath2.append(os.path.join(test_path, item))

    for items1 in subpath2:
        for items2 in os.listdir(items1):
            p = os.path.join(items1, items2)
            imdb2.append(p)

    testX = []
    testY = []
    for imagepath in imdb2:
        image = Image.open(imagepath)
        img = image.resize((224, 224))
        testX.append(np.array(img))
        image_class = imagepath.split('/')[-2]
        testY.append(int(image_class))

    Xtest = np.array(testX)
    Y_test = np.array(testY)
    X_test = Xtest.reshape(Xtest.shape[0], 224, 224, 3)
    return X_train, Y_train, X_test, Y_test

x, img_input = FEC.create_model()
X_train, Y_train, X_test, Y_test = readData(train_path, test_path)

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_train = Y_train.reshape((Y_train.shape[0],1,Y_train.shape[1]))
Y_test = np_utils.to_categorical(Y_test, nb_classes)
Y_test = Y_test.reshape((Y_test.shape[0],1,Y_test.shape[1]))

x = Dense(1024, activation='relu', name='final_dense1')(x)
x = Dropout(0.5, name='final_drop')(x)

predictions = Dense(7, activation='softmax',name='final_classifi')(x)

reduce_learning_rate = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, min_lr=0.00001, verbose=1)
callbacks = [reduce_learning_rate]

model = Model(inputs=img_input,outputs=predictions)
model.summary()
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

model.fit(X_train, Y_train,
              batch_size=32, nb_epoch=100, verbose=1, callbacks=callbacks)

score = model.evaluate(X_test, Y_test, batch_size=32)
print(score)

