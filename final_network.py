from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Reshape, Input
from keras.layers.merge import concatenate
from keras import callbacks
import pickle
from LSTM_model.settings import *
import numpy as np
from CNN_model.feature_extraction_cnn import load_scattered_datasets_cnn
from CNN_model.genre_classification_cnn import reshape_data_to_correct_format

cnn = load_model(CNN)
lstm = load_model(LSTM_NET)

#lstm features
features = open(LSTM_FEATURES, 'rb')
d = pickle.load(features)
X_train, _ = d['train']
X_val, _ = d['validation']
X_test, _ = d['test']

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

# import train, validation and test feature vectors
datasets = load_scattered_datasets_cnn()
X_train_cnn, Y_train = zip(*datasets['train'])
X_test_cnn, Y_test = zip(*datasets['test'])
X_validation_cnn, Y_validation = zip(*datasets['validation'])

train_data = reshape_data_to_correct_format(np.array(X_train_cnn))
train_labels = np.array(Y_train)
test_data = reshape_data_to_correct_format(np.array(X_test_cnn))
test_labels = np.array(Y_test)
validation_data = reshape_data_to_correct_format(np.array(X_validation_cnn))
validation_labels = np.array(Y_validation)

# train_data = np.array(X_train_cnn).reshape(700, 80, 630, 1)
# train_labels = np.array(Y_train)
# test_data = np.array(X_test_cnn).reshape(100, 80, 630, 1)
# test_labels = np.array(Y_test)
# validation_data = np.array(X_validation_cnn).reshape(200, 80, 630, 1)
# validation_labels = np.array(Y_validation)

cnn_X_train = np.array(cnn.predict(train_data))
lstm_X_train = np.array(lstm.predict(X_train))
cnn_X_val = np.array(cnn.predict(validation_data))
lstm_X_val = np.array(lstm.predict(X_val))
cnn_X_test = np.array(cnn.predict(test_data))
lstm_X_test = np.array(lstm.predict(X_test))


X_train_final = np.hstack((cnn_X_train, lstm_X_train))
X_val_final = np.hstack((cnn_X_val, lstm_X_val))
X_test_final = np.hstack((cnn_X_test, lstm_X_test))

print(X_train_final.shape)
print(X_val_final.shape)
print(X_test_final.shape)
print(train_labels)
print(validation_labels)
print(test_labels)


model = Sequential()
model.add(Dense(128, input_shape=((20, )), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_GENRES, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

model.fit(X_train_final, train_labels, validation_data=(X_val_final, validation_labels),
           batch_size=32, epochs=1000)

score, accuracy = model.evaluate(X_test_final, test_labels, verbose=1)

print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

model.save(FINAL_NET)
