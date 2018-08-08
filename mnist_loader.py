import numpy as np
import numpy.matlib

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

test_imgs=test_imgs.reshape(test_imgs.shape[0], img_height, img_width, 1)
train_imgs=train_imgs.reshape(train_imgs.shape[0], img_height, img_width, 1)
test_imgs=test_imgs.astype('float32')
train_imgs=train_imgs.astype('float32')
test_imgs/=255
train_imgs/=255

num_classes=10
 
mnist_data={}
mnist_data["test_labels"]=test_labs
mnist_data["train_labels"]=train_labs
mnist_data["test_images"]=test_imgs
mnist_data["train_images"]=train_imgs
mnist_data["num_classes"]=num_classes
