from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.misc import toimage
%matplotlib inline

# Каталог с данными для обучения
train_dir = 'small/train'
# Каталог с данными для проверки
val_dir = 'small/validation'
# Каталог с данными для тестирования
test_dir = 'small/test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Размер мини-выборки
batch_size = 64
# Количество изображений для обучения
nb_train_samples = 17500
# Количество изображений для проверки
nb_validation_samples = 3750
# Количество изображений для тестирования
nb_test_samples = 3750


train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

image_file_name = train_dir + '/dogs/dog.1.jpg'
img = image.load_img(image_file_name, target_size=(150,150))
plt.imshow(img)


x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in train_datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width,img_height), batch_size=batch_size, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = test_datagen.flow_from_directory(val_dir, target_size=(img_width,img_height), batch_size=batch_size, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width,img_height), batch_size=batch_size, class_mode='binary')


vgg16_net = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
vgg16_net.trainable = False
vgg16_net.summary()


model = Sequential()
model.add(vgg16_net)
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=val_generator,validation_steps=25)


scores = model.evaluate_generator(test_generator,50)
print(scores[1]*100)
