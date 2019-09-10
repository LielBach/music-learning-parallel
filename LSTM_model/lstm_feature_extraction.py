import numpy as np
import keras as ks
from settings import *
from feature_extraction import load_scattered_datasets
from keras_self_attention import SeqSelfAttention
import pickle


def get_labels(x):
    label_lst = []
    for arr in x:
        max_ind = np.where(arr == np.amax(arr))[0][0]
        label_lst.append(max_ind)
    return label_lst


def get_mean(X):
    return np.mean(X, axis=0)


def get_min(X):
    min_labels = [np.min(x) for x in X]
    return np.array(min_labels)


def get_max(X):
    min_labels = [np.max(x) for x in X]
    return np.array(min_labels)


def larger_than_k(X):
    k_larger = [len(np.where(x > K)[0]) for x in X]
    return (1/NUM_FRAMES)*np.array(k_larger)


def extract_lstm_features(model, X):
    lst = []
    for x in X:
        prediction = model.predict(np.array([x]))[0]
        mean = get_mean(prediction)
        minimum = get_min(prediction)
        maximum = get_max(prediction)
        larger = larger_than_k(prediction)
        res = np.concatenate([mean, minimum, maximum, larger])
        lst.append(res)
    return lst


def main():
    model = ks.models.load_model(AUTO_ENCODER_PATH, custom_objects={'SeqSelfAttention': SeqSelfAttention})
    datasets = load_scattered_datasets()
    X_test, Y_test = datasets['test']
    X_val, Y_val = datasets['validation']
    X_train, Y_train = datasets['train']

    res = {'test': (extract_lstm_features(model, X_test), Y_test),
           'train': (extract_lstm_features(model, X_train), Y_train),
           'validation': (extract_lstm_features(model, X_val), Y_val)}
    with open(LSTM_FEATURES, "wb+") as file:
        pickle.dump(res, file)


if __name__ == "__main__":
    main()

