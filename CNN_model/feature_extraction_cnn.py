import librosa
import pickle
import os
from settings_cnn import *
import numpy as np


genre_dictionary = {
    'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
    'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9
}


def extract_raw_features(track_file_path):
    """extracts raw features of audio file using librosa"""
    raw_track, _ = librosa.load(track_file_path, sr=SAMPLING_RATE, duration=29, mono=True)

    return raw_track


def scatter_dataset(dataset_path):
    """creates scattered features for data set"""

    # get track names
    all_tracks_names = os.listdir(dataset_path)

    # get track labels
    labels = [genre_dictionary[track_file_name.split(
        '.')[0]] for track_file_name in all_tracks_names]

    # create raw data from tracks
    raw_data_samples = map(
        lambda track_file_name: extract_raw_features(
            dataset_path + '/' + track_file_name),
        all_tracks_names)

    stft_data_samples = map(
        lambda sample: np.abs(librosa.core.stft(sample, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)), raw_data_samples)

    return zip(stft_data_samples, labels)


def create_dataset_dictionary(train, test, validation):
    return {
        'train': train,
        'test': test,
        'validation': validation
    }


def get_all_feature_vectors():

    # get folder list
    dirs = os.listdir("genres")
    paths = [os.path.join("genres", dirs[i]) for i in range(len(dirs))]

    # finds feature vectors for all files by folders(genres)
    all_data = list(map(scatter_dataset, paths))

    # randomly shuffles all feature vectors
    all_data = [item for lst in all_data for item in lst]
    np.random.seed(5)
    np.random.shuffle(all_data)

    return all_data


def load_scattered_datasets_cnn():
    """loads scatter features of dataset, or creates them if there aren't any"""

    # if feature file already exist -> load them and return
    if(os.path.exists('./train-scattered_cnn') and
       os.path.exists('./test-scattered_cnn') and
            os.path.exists('./validation-scattered_cnn')):
        with open('./train-scattered_cnn', 'rb') as train_file, \
                open('./test-scattered_cnn', 'rb') as test_file, \
                open('./validation-scattered_cnn', 'rb') as validation_file:
            return create_dataset_dictionary(
                pickle.load(train_file),
                pickle.load(test_file),
                pickle.load(validation_file)
            )

    # in any other case, create features
    all_data = get_all_feature_vectors()

    # divide data into train, test, and validation sets
    size = len(all_data)

    train_index = int(size*TRAIN_SIZE)
    scattered_train = all_data[0:train_index]

    test_index = train_index + int(size*TEST_SIZE)
    scattered_test = all_data[train_index:test_index]

    scattered_validation = all_data[test_index:]

    # save data in corresponding files
    with open('./train-scattered_cnn', 'wb+') as train_file, \
            open('./test-scattered_cnn', 'wb+') as test_file, \
            open('./validation-scattered_cnn', 'wb+') as validation_file:
        pickle.dump(scattered_train, train_file)
        pickle.dump(scattered_test, test_file)
        pickle.dump(scattered_validation, validation_file)

    # return dictionary of the three feature vector sets
    return create_dataset_dictionary(
        scattered_train,
        scattered_test,
        scattered_validation
    )


load_scattered_datasets_cnn()