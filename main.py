import matplotlib.pyplot as plt 
import matplotlib
import numpy as np 
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
import tkinter
matplotlib.use('TkAgg')

##Loading the data
(train_images, train_labels), (test_images, test_labels) =  mnist.load_data()

##Reshaping and normalization step
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = train_images.astype('float32')/255 #dividing by 255 normalize inputs from 0-255 to 0-1

test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images = test_images.astype('float32')/255

##Changing the label from single digit to one-hot format
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)

##Three Convolution layers and one fully-connected layer
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)), #16 different 3*3 kernels
    MaxPooling2D(pool_size=(2,2)), #Pool the max values over a 2*2 kernel
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(16, (3,3), activation='relu'),
    Flatten(), #Flatten final 3*3*16 output matrix into a 144-length vector
    Dense(15, activation='relu'),
    Dense(10, activation='softmax'),
])

#Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #Or 'rmsprop' for the optimizer ?

##Training the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, verbose=1, validation_data=(test_images,test_labels))
history_dict = history.history

##Plotting the evolution of the accuracy during the training
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

n = len(acc_values)
epochs = range(1, n+1)

#plt.subplot(2,1,1)
plt.plot(epochs, acc_values, 'bo', label='Training accuracy') #bo is for blue dot
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy') #b is for "solid blue line"
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

##Plotting the evolution of the loss during the training
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

#plt.subplot(2,1,1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
