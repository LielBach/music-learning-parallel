import numpy as np
from feature_extraction import load_scattered_datasets
from keras.models import Sequential
import keras.backend as keras_backend
from keras.layers import LSTM, Dense, BatchNormalization, Bidirectional
from settings import *
#import keras_attention
import keras_self_attention as ksa

def lst_feat_layer(X):
    return \
        keras_backend.variable(keras_backend.map_fn(get_lstm_features, X))


def get_lstm_features(x):
    print(x.shape)
    a = keras_backend.max(x)
    print("a = " + str(a.shape))
    b = keras_backend.min(x)
    print("b " + str(b.shape))
    c = keras_backend.mean(x)
    print("c " + str(c.shape))
    d = keras_backend.mean(keras_backend.map_fn(
                                           lambda y: keras_backend.greater(y, K), x))
    return keras_backend.variable(keras_backend.stack([a, b, c, d], axis=0))


#import train, validation and test feature vectors
print("getting files")
datasets = load_scattered_datasets()
X_train, Y_train = datasets['train']
X_val, Y_val = datasets['validation']
X_test, Y_test = datasets['test']
print("finished getting files")

Y_train = [np.array([elem for i in range(NUM_FRAMES)]) for elem in Y_train]
Y_val = [np.array([elem for i in range(NUM_FRAMES)]) for elem in Y_val]
Y_test = [np.array([elem for i in range(NUM_FRAMES)]) for elem in Y_test]
model = Sequential()

model.add(LSTM(NUM_SCATT_FEAT, dropout=0.05, activation='relu',
               recurrent_dropout=0.35, return_sequences=True, input_shape=(NUM_FRAMES, NUM_SCATT_FEAT)))

model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0005, center=True, scale=True, beta_initializer='zeros',
                             gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))

#model.add(ksa.SeqSelfAttention(attention_activation='relu'))
#model.add(keras_attention.SelfAttention())

#model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0005, center=True, scale=True, beta_initializer='zeros',
#                             gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))


model.add(Dense(units=NUM_GENRES, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()

model.fit(np.array(X_train), np.array(Y_train), validation_data=(np.array(X_val),
                                                                 np.array(Y_val)), batch_size=15, epochs=300)


print("\nTesting ...")
score, accuracy = model.evaluate(
    np.array(X_test), np.array(Y_test), verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)
model.save(AUTO_ENCODER_PATH)
