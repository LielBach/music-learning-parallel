from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape, Input
import pickle
from settings import *
import numpy as np

features = open(LSTM_FEATURES, 'rb')
d = pickle.load(features)
X_train, Y_train = d['train']
X_val, Y_val = d['validation']
X_test, Y_test = d['test']

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_val = np.array(X_val)
Y_val = np.array(Y_val)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

"""defining feed forward for lstm features"""
input0 = Input(shape=(250,))
first = Dense(units=128, activation='relu', input_shape=(250, ))(input0)
second = (Dense(units=64, activation='relu'))(first)
third = (Dense(units=32, activation='relu'))(second)
out = (Dense(units=NUM_GENRES, activation='softmax'))(third)
model = Model(input0, out)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=15, epochs=500)

score, accuracy = model.evaluate(X_test, Y_test, verbose=1)

print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)
model.save("./feed_forward")

