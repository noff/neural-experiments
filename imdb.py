import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import DEnse, Activation, Embedding
from keras.layers import LSTM, SpatialDropout1D
from keras.datasets import imdb

np.random.seed(42)

max_features = 5000
maxlen = 80

# Load data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Fill and truncate reviews
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# Create network
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Teach
model.fit(X_train, y_train, batch_size=64, epochs=7, validation_data=(X_test, y_test), verbose=2)

# Test
scores = model.evaluate(X_test, y_test, batch_size=64)

print("Accuracy on test data: %.2f%%" % (scores[1] * 100))


# Base accuracy - 83.22%

# epochs
# 7 - 83.22%
# 5 - 83.65%
# 10 - 82.90%
# 15 - 82.61%

# LSTM neurons
# 100 - 83.22%
# 50 - 83.24%
# 125 - 83.52%, overlearn
# 150 - 83.19%

# Optimization algorithm
# adam - 83.22%
# SGD - 51.49%

# Max values - 83.24%
