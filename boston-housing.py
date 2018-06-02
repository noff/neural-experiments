import numpy as np
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(42)

# Load datasets
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Standartize
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

# Create model
model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(x_train.shape[1],)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Educate
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

# QA
mse, mae = model.evaluate(x_test, y_test, verbose=0)
print(mae)

# Forecast
pred = model.predict(x_test)
print(pred[1][0], y_test[1])
print(pred[50][0], y_test[50])
print(pred[100][0], y_test[100])
