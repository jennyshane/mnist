import random
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import keras
from keras import backend as K
from mnist_loader import mnist_data
from mnist_dmodel import model

test_labels=keras.utils.to_categorical(mnist_data["test_labels"])

model.load_weights('mnist_dropout_weights2.h5')
score=model.evaluate(mnist_data["test_images"], test_labels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

weights=model.get_layer(index=1).get_weights()[0]

filt_height=5
filt_width=5
ncols=8
nrows=4

filt_mat=np.zeros(((filt_height+2)*nrows, (filt_width+2)*ncols))

#apparently numpy has a meshgrid command, so this was probably unecessary
mat_rselector=np.matlib.repmat(np.floor(np.arange(0, nrows*(filt_height+2))/(filt_height+2)), ncols*(filt_width+2), 1).transpose()
mat_cselector=np.matlib.repmat(np.floor(np.arange(0, ncols*(filt_width+2))/(filt_width+2)), nrows*(filt_height+2), 1)

for i in range(0, ncols):
    for j in range(0, nrows*(filt_height+2)):
        mat_rselector[j, (i+1)*7-1]=-1
        mat_rselector[j, (i+1)*7-2]=-1
        mat_cselector[j, (i+1)*7-1]=-1
        mat_cselector[j, (i+1)*7-2]=-1

for i in range(0, nrows):
    for j in range(0, ncols*(filt_width+2)):
        mat_rselector[(i+1)*7-1, j]=-1
        mat_rselector[(i+1)*7-2, j]=-1
        mat_cselector[(i+1)*7-1, j]=-1
        mat_cselector[(i+1)*7-2, j]=-1

for i in range(0, nrows):
    for j in range(0, ncols):
        filt=np.squeeze(weights[:, :, :, i*ncols+j])
        filt=filt-filt.min()
        filt=filt*(255.0/filt.max())
        np.place(filt_mat, (mat_rselector==i) & (mat_cselector==j), filt)

plt.imshow(filt_mat, cmap="Greys", interpolation="nearest")
plt.savefig("level_1_filters_dropout2.png")
plt.show()

first_layer_output=K.function([model.layers[0].input], [model.layers[0].output])
test_input=np.expand_dims(mnist_data["test_images"][int(random.random()*10000)], axis=0)
test_output=first_layer_output([test_input])[0]

out_height=test_output.shape[1]
out_width=test_output.shape[2]
ncols=8
nrows=4
out_mat=np.zeros((out_height*nrows, out_width*ncols))

mat_rselector=np.matlib.repmat(np.floor(np.arange(0, nrows*out_height)/out_height), ncols*out_width, 1).transpose()
mat_cselector=np.matlib.repmat(np.floor(np.arange(0, ncols*out_width)/out_width), nrows*out_height, 1)

for i in range(0, nrows):
    for j in range(0, ncols):
        temp=np.squeeze(test_output[:, :, :, i*ncols+j])
        temp=temp-temp.min()
        temp=temp*(255.0/temp.max())
        np.place(out_mat, (mat_rselector==i) & (mat_cselector==j), temp)
    
plt.imshow(out_mat, cmap="Greys", interpolation="nearest")
plt.savefig("level_1_outputs_dropout2.png")
plt.show()

