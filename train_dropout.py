import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Embedding, Input, merge
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.regularizers import l2, l1
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from mnist_dmodel import model
from mnist_loader import mnist_data 

test_labs=keras.utils.to_categorical(mnist_data["test_labels"])
train_labs=keras.utils.to_categorical(mnist_data["train_labels"])

model.fit(mnist_data["train_images"], train_labs, batch_size=128, epochs=20, verbose=1, 
        validation_data=(mnist_data["test_images"], test_labs), shuffle=True)
score = model.evaluate(mnist_data["test_images"], test_labs, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

savename="mnist_dropout_weights2.h5"
model.save_weights(savename, overwrite=True)
