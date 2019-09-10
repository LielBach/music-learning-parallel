import os
import numpy as np
from feature_extraction_cnn import load_scattered_datasets_cnn
from keras.models import Sequential
from keras.optimizers import Adam, SGD
import keras.backend as keras_backend
from keras.models import load_model
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.utils.np_utils import to_categorical
from settings_cnn import *



def reshape_data_to_correct_format(data):
    return data.reshape(len(data), data.shape[1],  data.shape[2], 1)


def print_results(results):
    with open('./results', 'w') as results_file:
        for result, metric_name in zip(results, model.metrics_names):
            str = "{}: {}".format(metric_name, result)
            print (str, file=results_file)


# import train, validation and test feature vectors
datasets = load_scattered_datasets_cnn()

X_train, Y_train = zip(*datasets['train'])
X_test, Y_test = zip(*datasets['test'])
X_validation, Y_validation = zip(*datasets['validation'])

train_data = reshape_data_to_correct_format(np.array(X_train))
train_labels = np.array(Y_train)
test_data = reshape_data_to_correct_format(np.array(X_test))
test_labels = np.array(Y_test)
validation_data = reshape_data_to_correct_format(np.array(X_validation))
validation_labels = np.array(Y_validation)


model = Sequential()


model.add(Conv2D(128, kernel_size=(513, 4), activation='relu',
                 input_shape=train_data.shape[1:4]))

model.add(AveragePooling2D((1, 2)))

model.add(Conv2D(128, kernel_size=(1, 4), activation='relu'))

model.add(AveragePooling2D((1, 2)))

model.add(Conv2D(256, kernel_size=(1, 4), activation='relu'))

model.add(AveragePooling2D((1, 26)))

model.add(Dense(300, activation='relu'))

model.add(Dense(150, activation='relu'))

model.add(Flatten())

model.add(Dense(NUM_GENRES, activation='softmax'))

model.compile(optimizer=Adam(clipvalue=0.5, clipnorm=1.), loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.summary()

if(os.path.isfile(MODEL_PATH)):
    model = load_model(MODEL_PATH)
else:
    model.fit(train_data, train_labels, validation_data=(
        validation_data, validation_labels), epochs=EPOCH_NUMBER, batch_size=BATCH_SIZE)

    model.save(MODEL_PATH)

results = model.evaluate(test_data, test_labels, batch_size=BATCH_SIZE)

print_results(results)
