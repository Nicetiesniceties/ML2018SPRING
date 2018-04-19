# -*- coding: utf-8 -*-
"""Copy of Copy of cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YTEllqO3N-2HI3evwQ5HFLFaJDbahXai
"""
#residual learning
import numpy as np
import sys
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import csv
import pandas as pd
from keras.utils import np_utils
from pandas import to_numeric
import random
np.random.seed(25)
# %matplotlib inline
training_set = pd.read_csv(sys.argc[1])
train_y = training_set[["label"]]
raw_train_x = training_set[["feature"]]
# print(train_x)
train_y = train_y.astype("int")
# print(train_y)
raw_train_x = np.array(raw_train_x)
temp_train_x = []
for i in raw_train_x:
  temp_train_x.append(i[0].split(" "))
train_x = np.zeros((len(temp_train_x), 48 * 48))
for i in range(len(temp_train_x)):
  for j in range(len(temp_train_x[i])):
    train_x[i][j] = float(temp_train_x[i][j])

from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2, random_state=25)
# print(train_y, val_y)

# print(train_x)
train_x = train_x.reshape(len(train_x), 48, 48)
val_x = val_x.reshape(len(val_x), 48, 48)
print(train_x.shape)
print(val_x.shape)

# add channel
train_x = np.expand_dims(train_x, axis=3)
# x_test = np.expand_dims(x_test, axis=3)
val_x = np.expand_dims(val_x, axis=3)
print(train_x.shape)
# print(x_test.shape)
train_x_normalized = train_x / 255
# x_test_normalized = x_test / 255
val_x_normalized = val_x / 255

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.show()

from keras.utils import *
from keras.models import Sequential, Input
from keras.layers import Dense, Activation, Flatten, Dropout, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))
 
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(7, activation='softmax'))


# print(train_x)
train_x_normalized = train_x_normalized.reshape(len(train_x_normalized), 48, 48, 1)
print(train_x_normalized.shape)
val_x_normalized = val_x_normalized.reshape(len(val_x_normalized), 48, 48, 1)
print(val_x_normalized.shape)

# 定義訓練方式
# loss function: 交叉熵
# optimizer: Adam
# 評估模型: 準確率
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

train_y_onehot = np_utils.to_categorical(train_y)
val_y_onehot = np_utils.to_categorical(val_y)

gen = ImageDataGenerator(featurewise_center=False,
                         samplewise_center=False,
                         rotation_range=6, 
                         width_shift_range=0.08, 
                         shear_range=0.3,
                         height_shift_range=0.08, 
                         zoom_range=0.08,
                         data_format="channels_last")

gen.fit(train_x_normalized)
train_generator = gen.flow(train_x_normalized, train_y_onehot, batch_size=300)

#test_gen = ImageDataGenerator(data_format="channels_last")
#test_gen.fit(x_test_normalized)
#test_generator = test_gen.flow(x_test_normalized, y_test, batch_size=300)

from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
learning_rate_function = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# 開始訓練
train_history = model.fit_generator(train_generator, steps_per_epoch=300, epochs=20, verbose=1, validation_data = (val_x_normalized, val_y_onehot))# , callbacks=[learning_rate_function])

# train_history = model.fit(train_x_normalized, train_y_onehot, validation_split=0.2, epochs=100, batch_size=500, verbose=1)

show_train_history(train_history, 'acc', 'val_acc')

show_train_history(train_history, 'loss', 'val_loss')

