from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np
import matplotlib.pyplot as plt
# fix random seed for reproducibility
np.random.seed(7)
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print X_train.shape


#Our MNIST images only have a depth of 1, but we must explicitly declare that.

#In other words, we want to transform our dataset from having shape (n, width, height) to (n, depth, width, height).
#plt.matshow(X_train[0]) #imshow
#plt.show()
	
#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0],28, 28,1)

# no use that
#plt.matshow(X_train[0][0]) #imshow

print X_train.shape
# (60000, 1, 28, 28)

# convert to float ( we have unint 8 so far)

	
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
### building the model
model = Sequential()
#model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))#for  theano
# for tensor flow its 
model.add(Convolution2D(32, (3, 3),strides=(1,1), activation='relu', input_shape=(28,28,1))) # little difference for new api
# test stride strides=(2, 2). Is faster but not so good
#*Note: The step size (stride) is (1,1) by default, and it can be tuned using the 'subsample' parameter.
# 32 is the number ( depth of filters)	
# interval 1: from 1 to 3. interval2: from 2 to 4 ...... interval 26: from 26 to 28 -> # =28 -size+1 for stride=1.
print model.output_shape
# (None, 32, 26, 26)
# add more layers
model.add(Convolution2D(32,( 3, 3), activation='relu')) # new API (3,3)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# fit the training data
model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1)# one epoch has still acc= 99.09 %
score = model.evaluate(X_test, Y_test, verbose=0)# accuracy in the test set= 99.16%
# Returns the loss value & metrics values for the model in test mode.
# batch size=100, epoch= 1   ->>> acc = 99.29 %
#            100         5              99.31%
#            5000        1              99.32%
