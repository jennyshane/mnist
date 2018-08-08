import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import keras
from keras import backend as K
#from mnist_loader import mnist_data
from mnist_dmodel import model

#train_labels=keras.utils.to_categorical(mnist_data["train_labels"])
#test_labels=keras.utils.to_categorical(mnist_data["test_labels"])

model.load_weights('weights/mnist_dropout_weights2.h5')

layer_dict=dict([(layer.name, layer) for layer in model.layers])

layer_name='conv2d_2'
filter_index=20

input_image=model.input
layer_output=layer_dict[layer_name].output
loss=K.mean(layer_output[:, :, :, filter_index])
grads=K.gradients(loss, input_image)[0]
grads/=(K.sqrt(K.mean(K.square(grads)))+1e-5)

iterate=K.function([input_image], [loss, grads])

input_img_data=np.random.random((1, 28, 28, 1))+.5

for i in range(300):
    loss_value, grads_value=iterate([input_img_data])
    input_img_data+=grads_value*.03
    print(loss_value)

input_img_data=input_img_data-input_img_data.min()
input_img_data=input_img_data*(255.0/input_img_data.max())

plt.imshow(np.squeeze(input_img_data), cmap="Greys", interpolation="nearest")
plt.show()

