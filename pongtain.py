import json
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

with open('training_data-100k.json') as f:
	data = json.load(f)
	xs = np.array(data['xs'])
	ys = np.array(data['ys'])

x_train = xs[:-10000]
y_train = ys[:-10000]
x_test = xs[-10000:]
y_test = ys[-10000:]

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=6))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))