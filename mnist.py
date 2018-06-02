import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


# Устанавливаем seed для повторяемости результатов
numpy.random.seed(42)


# Загружаем данные
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование размерности изображений
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Нормализация данных
X_train = X_train.astype('float32')
X_train /= 255
X_test = X_test.astype('float32')
X_test /= 255


# Преобразуем метки в категории
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# Создаем последовательную модель
model = Sequential()

# Добавляем уровни сети
model.add(Dense(1200, input_dim=784, activation="relu", kernel_initializer="normal"))
model.add(Dense(10, activation="softmax", kernel_initializer="normal"))

# Компилируем модель
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
print(model.summary())

# Save architecture
model_json = model.to_json()
json_file = open("mnist_model.json", "w")
json_file.write(model_json)
json_file.close()

# Обучаем сеть
# epochs 125 the best
# batch size 400
# Dense  -
model.fit(X_train, Y_train, batch_size=400, epochs=125, validation_split=0.2, verbose=2)

# Save learning data
model.save_weights("mnist_model.h5")

# Оцениваем качество обучения сети на тестовых данных
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
