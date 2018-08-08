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

test_imgs="/home/jenny/prac/luaprac/mnist/t10k-images-idx3-ubyte"
test_labels="/home/jenny/prac/luaprac/mnist/t10k-labels-idx1-ubyte"

train_imgs="/home/jenny/prac/luaprac/mnist/train-images-idx3-ubyte"
train_labels="/home/jenny/prac/luaprac/mnist/train-labels-idx1-ubyte"

#read in train image data:
with open(train_imgs, "rb") as f:
    byte=f.read(4)
    magic_num=int.from_bytes(byte, byteorder='big')
    byte=f.read(4)
    num_imgs=int.from_bytes(byte, byteorder='big')
    byte=f.read(4)
    img_height=int.from_bytes(byte, byteorder='big')
    byte=f.read(4)
    img_width=int.from_bytes(byte, byteorder='big')
    img_size=img_width*img_height
    assert(magic_num==2051)
    train_imgs=np.zeros((num_imgs, img_width, img_height))
    for n in range(0, num_imgs):
        im_bytes=f.read(img_size)
        for i in range(0, img_height):
            for j in range(0, img_width):
                train_imgs[n, i, j]=im_bytes[i*img_width+j]

#read in train labels 
with open(train_labels, "rb") as f:
    byte=f.read(4)
    magic_num=int.from_bytes(byte, byteorder='big')
    byte=f.read(4)
    num_labs=int.from_bytes(byte, byteorder='big')
    assert(magic_num==2049)
    raw_labels=f.read(num_labs)
    train_labs=[i for i in raw_labels]

#read in test image data:
with open(test_imgs, "rb") as f:
    byte=f.read(4)
    magic_num=int.from_bytes(byte, byteorder='big')
    byte=f.read(4)
    num_imgs=int.from_bytes(byte, byteorder='big')
    byte=f.read(4)
    img_height=int.from_bytes(byte, byteorder='big')
    byte=f.read(4)
    img_width=int.from_bytes(byte, byteorder='big')
    img_size=img_width*img_height
    assert(magic_num==2051)
    test_imgs=np.zeros((num_imgs, img_width, img_height))
    for n in range(0, num_imgs):
        im_bytes=f.read(img_size)
        for i in range(0, img_height):
            for j in range(0, img_width):
                test_imgs[n, i, j]=im_bytes[i*img_width+j]

#read in test labels 
with open(test_labels, "rb") as f:
    byte=f.read(4)
    magic_num=int.from_bytes(byte, byteorder='big')
    byte=f.read(4)
    num_labs=int.from_bytes(byte, byteorder='big')
    assert(magic_num==2049)
    raw_labels=f.read(num_labs)
    test_labs=[i for i in raw_labels]
         
ncols=20
nrows=20
rand_indices=(10000*np.random.random([ncols*nrows])).astype("int")
img_mat=np.zeros((img_height*nrows, img_width*ncols))

mat_rselector=np.matlib.repmat(np.floor(np.arange(0, nrows*img_height)/img_height), ncols*img_width, 1).transpose()
mat_cselector=np.matlib.repmat(np.floor(np.arange(0, ncols*img_width)/img_width), nrows*img_height, 1)

for i in range(0, nrows):
    labels=[]
    for j in range(0, ncols):
        np.place(img_mat, (mat_rselector==i) & (mat_cselector==j), train_imgs[rand_indices[i*ncols+j]])
        labels.append(train_labs[rand_indices[i*ncols+j]])
    print(labels)
    

plt.imshow(img_mat, cmap="Greys")
plt.show()

num_classes=10
test_labs=keras.utils.to_categorical(test_labs)
train_labs=keras.utils.to_categorical(train_labs)
test_imgs=test_imgs.reshape(test_imgs.shape[0], img_height, img_width, 1)
train_imgs=train_imgs.reshape(train_imgs.shape[0], img_height, img_width, 1)
test_imgs=test_imgs.astype('float32')
train_imgs=train_imgs.astype('float32')
test_imgs/=255
train_imgs/=255

model=Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(img_height, img_width, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(train_imgs, train_labs, batch_size=128, epochs=20, verbose=1, validation_data=(test_imgs, test_labs), shuffle=True)
score = model.evaluate(test_imgs, test_labs, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

savename="mnist_weights.h5"
model.save_weights(savename, overwrite=True)
