import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

numpy.random.seed(42)

# Load datasets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Размер мини-выборки
batch_size = 32
# Количество классов изображений
nb_classes = 10
# Количество эпох для обучения
nb_epoch = 25
# Размер изображений
img_rows, img_cols = 32, 32
# Количество каналов в изображении: RGB
img_channels = 3


# Normalize pixels intensivity
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert classes to categories
Y_train =np_utils.to_categorical(y_train, 10)
Y_test =np_utils.to_categorical(y_test, 10)

# Create model
model = Sequential()

# The first convolutional layer
model.add(Conv2D(32, (3, 3), input_shape=(32,32,3), activation='relu', padding='same'))

# Second convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu'))

# Pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Regularization layer
model.add(Dropout(0.25))

# Third
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Classifier
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# Result layer
model.add(Dense(10, activation='softmax'))

# Compile
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Learn
model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0.1,
              shuffle=True,
              verbose=2)



# Test
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy on test data: %.2f%%" % (scores[1] * 100))
