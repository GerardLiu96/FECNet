import os
import pandas as pd
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Convolution2D
from keras.layers import Flatten, Dense, Dropout,BatchNormalization, Activation, Lambda
from keras.regularizers import l2
from keras.layers import Input, Concatenate, concatenate
import keras.backend as K
import tensorflow as tf
from keras.models import Model,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model,np_utils
from keras import regularizers
import cv2


DATA_FORMAT='channels_last' # Theano:'channels_first' Tensorflow:'channels_last'
WEIGHT_DECAY=0.0005
LRN2D_NORM=False
USE_BN=True
IM_WIDTH=224
IM_HEIGHT=224
batch_num = 16


#normalization
def conv2D_lrn2d(x,filters,kernel_size,strides=(1,1),padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,lrn2d_norm=LRN2D_NORM,weight_decay=WEIGHT_DECAY,name=None):
    #l2 normalization
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None

    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint,name=name)(x)

    if lrn2d_norm:
        #batch normalization
        x=BatchNormalization()(x)

    return x



def inception_module(x,params,concat_axis,padding='same',data_format=DATA_FORMAT,dilation_rate=(1,1),activation='relu',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,weight_decay=None):
    (branch1,branch2,branch3,branch4)=params
    if weight_decay:
        kernel_regularizer=regularizers.l2(weight_decay)
        bias_regularizer=regularizers.l2(weight_decay)
    else:
        kernel_regularizer=None
        bias_regularizer=None
    #1x1
    if branch1[1]>0:
        pathway1=Conv2D(filters=branch1[1],kernel_size=(1,1),strides=branch1[0],padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    #1x1->3x3
    pathway2=Conv2D(filters=branch2[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway2=Conv2D(filters=branch2[1],kernel_size=(3,3),strides=branch1[0],padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway2)

    #1x1->5x5
    pathway3=Conv2D(filters=branch3[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(x)
    pathway3=Conv2D(filters=branch3[1],kernel_size=(5,5),strides=branch1[0],padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway3)

    #3x3->1x1
    pathway4=MaxPooling2D(pool_size=(3,3),strides=branch1[0],padding=padding,data_format=DATA_FORMAT)(x)
    if branch4[0]>0:
        pathway4=Conv2D(filters=branch4[0],kernel_size=(1,1),strides=1,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=activation,use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer,kernel_constraint=kernel_constraint,bias_constraint=bias_constraint)(pathway4)
    if branch1[1]>0:
        return concatenate([pathway1,pathway2,pathway3,pathway4],axis=concat_axis)
    else:
        return concatenate([pathway2, pathway3, pathway4], axis=concat_axis)




def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    x = Activation('relu')(input)
    x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate

    return x, nb_filter


def l2_norm(x):
    x = x ** 2
    x = K.sum(x, axis=1)
    x = K.sqrt(x)
    return x


def triplet_loss(y_true, y_pred):
    batch = batch_num
    ref1 = y_pred[0:batch,:]
    pos1 = y_pred[batch:batch+batch,:]
    neg1 = y_pred[batch+batch:3*batch,:]
    dis_pos = K.sum(K.square(ref1 - pos1), axis=1, keepdims=True)
    dis_neg = K.sum(K.square(ref1 - neg1), axis=1, keepdims=True)
    #dis_pos = K.sqrt(dis_pos)
    #dis_neg = K.sqrt(dis_neg)
    a1pha = 0.2
    d1 = K.maximum(0.0,(dis_pos-dis_neg)+a1pha)
    d2 = K.maximum(0.0,(dis_pos-dis_neg)+alpha) 
    d = K.sum(d1,d2)
    return K.mean(d)



def create_model():
    #Data format:tensorflow,channels_last;theano,channels_last
    if DATA_FORMAT=='channels_first':
        INP_SHAPE=(3,224,224)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=1
    elif DATA_FORMAT=='channels_last':
        INP_SHAPE=(224,224,3)
        img_input=Input(shape=INP_SHAPE)
        CONCAT_AXIS=3
    else:
        raise Exception('Invalid Dim Ordering')

    x=conv2D_lrn2d(img_input,64,(7,7),2,padding='same',lrn2d_norm=False,name="FaceNet_NN2_conv2D")
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    x=BatchNormalization()(x)

    x=conv2D_lrn2d(x,64,(1,1),1,padding='same',lrn2d_norm=False)

    x=conv2D_lrn2d(x,192,(3,3),1,padding='same',lrn2d_norm=True)
    x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)

    x=inception_module(x,params=[(1,64),(96,128),(16,32),(32,)],concat_axis=CONCAT_AXIS) #3a
    x=inception_module(x,params=[(1,64),(96,128),(32,64),(64,)],concat_axis=CONCAT_AXIS) #3b
    #x=MaxPooling2D(pool_size=(3,3),strides=2,padding='same',data_format=DATA_FORMAT)(x)
    x = inception_module(x, params=[(2,0), (128, 256), (32, 64), (0,)], concat_axis=CONCAT_AXIS)  # 3c

    x=inception_module(x,params=[(1,256),(96,192),(32,42),(128,)],concat_axis=CONCAT_AXIS) #4a
    x=inception_module(x,params=[(1,224),(112,224),(32,64),(128,)],concat_axis=CONCAT_AXIS) #4b
    x=inception_module(x,params=[(1,192),(128,256),(32,64),(128,)],concat_axis=CONCAT_AXIS) #4c
    x=inception_module(x,params=[(1,160),(144,288),(32,64),(128,)],concat_axis=CONCAT_AXIS) #4d
    x=inception_module(x,params=[(2,0),(160,256),(64,128),(0,)],concat_axis=CONCAT_AXIS) #4e
    #x=MaxPooling2D(pool_size=(1,1),strides=1,padding='same',data_format=DATA_FORMAT,name="EndOfNN2")(x)

    x = Convolution2D(512, (1, 1), kernel_initializer="he_uniform", padding="same", name="DenseNet_initial_conv2D", use_bias=False,
                      kernel_regularizer=l2(WEIGHT_DECAY))(x)

    x = BatchNormalization()(x)

    x, nb_filter = dense_block(x, 5, 512, growth_rate=64,dropout_rate=0.5)

    x = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid', data_format=DATA_FORMAT)(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16)(x)
    #x = K.l2_normalize(x)
    x = Lambda(lambda x: K.l2_normalize(x))(x)

    return x, img_input


def load_triplet_images(csvpath,target_size):
    data = pd.read_csv(csvpath,error_bad_lines=False)
    trainX = []
    print(data)
    trainX1 = []
    trainX2 = []
    trainX3 = []
    for i in range(0,int(target_size/3)):
        mode = data.iloc[i, 5]
        #print(mode)
        img1 = cv2.imread(data.iloc[i, 1])
        img2 = cv2.imread(data.iloc[i, 2])
        img3 = cv2.imread(data.iloc[i, 3])
        #print(img1)
        if img1 is None or img2 is None or img3 is None:
            continue
        if mode == 1:
            trainX1.append(np.array(img2))
            trainX2.append(np.array(img3))
            trainX3.append(np.array(img1))
        elif mode == 2:
            trainX1.append(np.array(img3))
            trainX2.append(np.array(img1))
            trainX3.append(np.array(img2))
        elif mode == 3:
            trainX1.append(np.array(img1))
            trainX2.append(np.array(img2))
            trainX3.append(np.array(img3))
        #print(len(trainX1))
        if len(trainX1) == batch_num:
            #print("Add")
            trainX.extend(trainX1)
            trainX.extend(trainX2)
            trainX.extend(trainX3)
            trainX1 = []
            trainX2 = []
            trainX3 = []


    Xtrain = np.array(trainX)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 224, 224, 3)
    print(Xtrain.shape)
    Ytrain = np.zeros(shape=(Xtrain.shape[0],1,1,1))
    return Xtrain,Ytrain


if __name__=='__main__':
    train_x,train_y = load_triplet_images('labels.csv',43200)
    x, img_input = create_model()
    model = Model(inputs=img_input,outputs=[x])
    model.summary()
    from keras.optimizers import Adam
    model.compile(loss=triplet_loss, optimizer=Adam(lr=0.0005))# Follow the original paper
    #In original paper, they train 50K iterations
    model.fit(x=train_x, y=train_y, nb_epoch=50, batch_size=batch_num*3,shuffle=False)
    model.save("FECNet1.h5")





