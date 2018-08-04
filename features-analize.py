from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import Adam
import numpy as np

# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Размер мини-выборки
batch_size = 10
# Количество изображений для обучения
nb_train_samples = 17500
# Количество изображений для проверки
nb_validation_samples = 3750
# Количество изображений для тестирования
nb_test_samples = 3750

vgg16_net = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))


datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)


features_train = vgg16_net.predict_generator(
        train_generator, nb_train_samples // batch_size)
np.save(open('features_train.npy', 'wb'), features_train)
features_val = vgg16_net.predict_generator(
        val_generator, nb_validation_samples // batch_size)
np.save(open('features_val.npy', 'wb'), features_val)
features_test = vgg16_net.predict_generator(
        test_generator, nb_test_samples // batch_size)
np.save(open('features_test.npy', 'wb'), features_test)

labels_train =  np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))
labels_val =  np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))
labels_test =  np.array(
        [0] * (nb_test_samples // 2) + [1] * (nb_test_samples // 2))

features_train = np.load(open('features_train.npy', 'rb'))
features_val = np.load(open('features_val.npy', 'rb'))
features_test = np.load(open('features_test.npy', 'rb'))


model = Sequential()
model.add(Flatten(input_shape=features_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='Adam',
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(features_train, labels_train,
              epochs=15,
              batch_size=64,
              validation_data=(features_val, labels_val), verbose=2)

scores = model.evaluate(features_test, labels_test, verbose=1)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))


# Save architecture
model_json = model.to_json()
json_file = open("cats_features.json", "w")
json_file.write(model_json)
json_file.close()

model.save_weights("cats_features.h5")
