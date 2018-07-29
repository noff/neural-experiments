from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import Adam

train_dir = 'cats-and-dogs/train'
val_dir = 'cats-and-dogs/validation'
test_dir = 'cats-and-dogs/test'

img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)

batch_size = 64

nb_train_samples = 17500
nb_validation_samples = 3750
nb_test_samples = 3750

vgg16_net = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze weights of VGG16
vgg16_net.trainable = False

# Make complex network
model = Sequential()
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Fine tuning
vgg16_net.trainable = True
trainable = False
for layer in vgg16_net.layers:
    if layer.name == 'block5_conv1':
        trainable = True
    layer.trainable = trainable

model.summary()

# Compile
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])



# Load data
datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Accuracy: %.2f%%" % (scores[1]*100))



# Base accuracy: 90.81%
# V2 - 512 neurons: 91.43%
# V2 - 128 neurons: 91.27%
# V2 - 256 neurons: 91.11%
# V2 - 1024 neurons: 91.30%
# V2 - 512 > 256: 91.62%
