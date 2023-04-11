import numpy as np
import math
import mnist
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.utils import to_categorical, plot_model

#activation functions
x = np.array([1.3, 0.32, 0.21]) # softmax
y = 2 # sigmoid
a = -9 # relu
z = 4

def softmax(x):
    return np.exp(x) / sum(np.exp(x))
print(softmax(x))

def sigmoid(y):
  return 1 / (1 + math.exp(-y))
print(sigmoid(y))

def relu(a):
    return max(0.0, a)
print(relu(a))

def tanh(z):
    return np.sinh(z) / np.cosh(z) or -1j * np.tan(1j*z)
print(tanh(z))

# downloading MNIST dataset
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# normalizing the image data by scaling pixels - making 0 the neutral value instead of 0.5 to improve performance
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

print(train_images.shape)
print(test_images.shape)

# reshaping image data - flattening a 28x28 2D array into a 28*28=784 1D array where first 28 values represent pixels of the top edge of the image(first row from the top)
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

print(train_images.shape)

# Sequential model is a type of model where each next layer is connected to its predecessor and takes its output as the input, here we create 3 layers and define the
# shape of the input
model = Sequential([
  Dense(50, activation='relu', input_shape=(784,)),
  Dense(30, activation='relu'),
  Dense(10, activation='softmax'),
])

# compiling the model with chosen settings to make it ready to be trained
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# here we define how many epochs(runs) our model will do, we also define a batch size which defines how many images there will be in one batch. If we define batch_size as 30 for example
# this means that if we have 60 000 images then our model will do 60000/30=2000 updates (to its weights) during one epoch
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=5,
  batch_size=30,
)
#  TASK 1
print(model.summary())
plot_model(model, to_file='model_plot_5epochs.png', show_shapes=True, show_layer_names=True)

# TASK 2
# reloading the model:
model = Sequential([
  Dense(50, activation='relu', input_shape=(784,)),
  Dense(30, activation='relu'),
  Dense(10, activation='softmax'),
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# with 100 epochs
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=100,
  batch_size=30,
)
print(model.summary())
plot_model(model, to_file='model_plot_100epochs.png', show_shapes=True, show_layer_names=True)
'''
TASK 2 MY ANSWER:
I would say that it is really hard to say where the model stabilized because it heavily depends on how we define "stabilized"

It can very well be claimed to have stabilized around epoch 6 as the gains from this point rarely, if ever, exceed 0.003. Alternatively, it can be said that
it has truly stabilized around epoch 14 as the gains after that point are even smaller while the accuracy of 97,25% doesn't sound bad, so for the sake of having
an answer I will say "epoch 14" but I believe that this conclusion could be moved a lot to the left or right depending on the requirements
'''

# TASK 3
model = Sequential([
  Dense(50, activation='relu', input_shape=(784,)),
  Dense(30, activation='relu'),
  Dense(10, activation='softmax'),
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# with 100 epochs
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=5,
  batch_size=20,
)
print(model.summary())
'''
TASK 3 MY ANSWER:
The results compared to out baseline "5 epochs, 30 batch size" settings appear to be unaffected, the small differences can be assigned to the randomness
'''

# TASK 4
model = Sequential([
  Dense(50, input_shape=(784,)),
  LeakyReLU(alpha=0.1),
  Dense(30),
  LeakyReLU(alpha=0.1),
  Dropout(0.2),
  Dense(10, activation='softmax')
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=5,
  batch_size=30,
)
'''
TASK 4 MY ANSWER:
No, not really...
'''

# TASK 5

model = Sequential([
  Dense(256, activation='linear', input_shape=(784,)),
  Dense(256, activation='linear'),
  Dense(256, activation='linear'),
  Dense(128, activation='linear'),
  Dense(128, activation='linear'),
  Dense(128, activation='linear'),
  Dense(64, activation='linear'),
  Dense(64, activation='linear'),
  Dense(64, activation='linear'),
  Dense(10, activation='linear'),
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=10,
  batch_size=30,
)

# second model
model = Sequential([
  Dense(256, activation='linear', input_shape=(784,)),
  Dense(128, activation='linear'),
  Dense(64, activation='linear'),
  Dense(10, activation='linear'),
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=10,
  batch_size=30,
)
'''
TASK 5 MY ANSWER:
The values don't change as we iterate over the next epochs because activation functions cannot be linear, if they are linear then the model will only be effective
for 1 layer depth as nothing changes between input and output afterwards
'''