from kymatio import Scattering1D
from os import listdir
import librosa
import torch
import pickle
import os
from settings import *
import numpy as np

t = 0


genre_dictionary = {
    'metal': 1, 'disco': 2, 'classical': 3, 'hiphop': 4, 'jazz': 5,
    'country': 6, 'pop': 7, 'blues': 8, 'reggae': 9, 'rock': 10}


def extract_raw_features(track_file_path):
    """extracts raw features of audio file using librosa"""
    raw_track, _ = librosa.load(track_file_path)
    return raw_track


def split_sample(sample):
    """splits an audio sample (track) into frames"""
    splitted_track = [sample[i:i + frame_length]
                      for i in range(0, len(sample), frame_length)]
    return splitted_track



def to_normalized_tensor(sample_data):
    """converts an audio sample into a normalized tensor"""
    sample_data_tensor = torch.Tensor(sample_data).float()
    sample_data_tensor /= sample_data_tensor.abs().max()
    sample_data_tensor = sample_data_tensor.reshape(1,-1)

    return sample_data_tensor


def calculateScatter(scatter_function, sample):
    """calculates the scatter of a given librosa audio sample"""
    global t
    t += 1
    print(t)
    mfcc = librosa.feature.mfcc(sample, sr=22050)
    mfcc_tensor = to_normalized_tensor(mfcc)
    res = scatter_function.forward(mfcc_tensor)
    return res


def scatter_dataset(dataset_path):
    """creates scattered features for data set"""

    #define scattering function
    scattering_function = Scattering1D(6, mfcc_length, 8)

    #get track names
    all_tracks_names = listdir(dataset_path)

    #get track labels
    labels = [genre_dictionary[track_file_name.split('.')[0]] for track_file_name in all_tracks_names]

    #create raw data from tracks
    raw_data_samples = map(
        lambda track_file_name: extract_raw_features(
            dataset_path + '/' + track_file_name),
        all_tracks_names)

    #split tracks into frames
    raw_data_splitted_tracks = [split_sample(sample) for sample in raw_data_samples]

    #remove last frame from every track. meaning frames that have length<frame_length
    uni_sized_data_splitted_tracks = [[raw_data_splitted_tracks[i][j]
                                       for j in (range(len(raw_data_splitted_tracks[i])-1))] for i in
                                       range(len(raw_data_splitted_tracks))]

    # #turn data into tensors
    # tensor_data_tracks = [map(to_normalized_tensor, track)
    #                       for track in uni_sized_data_splitted_tracks]

    #scatter data to get final features
    scattered_data_samples = [[
        calculateScatter(scattering_function, frame) for frame in track] for track in uni_sized_data_splitted_tracks]

    return zip(scattered_data_samples, labels)


def create_dataset_dictionary(train, test, validation):
    return {
        'train': train,
        'test': test,
        'validation': validation
    }

def get_all_feature_vectors():

    #get folder list
    dirs = os.listdir("genres")
    paths = [os.path.join("genres", dirs[i]) for i in range(len(dirs))]

    #finds feature vectors for all files by folders(genres)
    all_data = list(map(scatter_dataset, paths))

    # randomly shuffles all feature vectors
    all_data = [item for lst in all_data for item in lst]
    np.random.seed(5)
    np.random.shuffle(all_data)

    return all_data


def to_numpy_arr_list(arr):
    for i in range(len(arr)):
        arr[i] = np.array(arr[i].numpy().flatten())
    res = np.array(arr[:NUM_FRAMES])
    return res


def format_data(data):
    # get training features
    X = np.array([to_numpy_arr_list(sample[SAMPLE_DATA]) for sample in data])

    # get training labels, for example if label='rock', y=[0,0,0,0,0,0,0,0,0,1]
    Y = [[1 if sample[SAMPLE_LABEL] == i else 0 for i in range(1, NUM_GENRES + 1)]
               for sample in data]

    return (X, Y)


def load_scattered_datasets():
    """loads scatter features of dataset, or creates them if there aren't any"""

    #if feature file already exist -> load them and return
    if(os.path.exists('./train-scattered') and
       os.path.exists('./test-scattered') and
            os.path.exists('./validation-scattered')):
        with open('./train-scattered', 'rb') as train_file, open('./test-scattered', 'rb') as test_file, open('./validation-scattered', 'rb') as validation_file:
            return create_dataset_dictionary(
                pickle.load(train_file),
                pickle.load(test_file),
                pickle.load(validation_file)
            )

    #in any other case, create features
    all_data = get_all_feature_vectors()

    #divide data into train, test, and validation sets
    size = len(all_data)
    train_index = int(size*TRAIN_SIZE)
    scattered_train = format_data(all_data[0:train_index])

    test_index = train_index + int(size*TEST_SIZE)
    scattered_test = format_data(all_data[train_index:test_index])

    scattered_validation = format_data(all_data[test_index:])

    #save data in corresponding files
    with open('./train-scattered', 'wb+') as train_file, open('./test-scattered', 'wb+') as test_file, open('./validation-scattered', 'wb+') as validation_file:
        pickle.dump(scattered_train, train_file)
        pickle.dump(scattered_test, test_file)
        pickle.dump(scattered_validation, validation_file)

    #return dictionary of the three feature vector sets
    return create_dataset_dictionary(
        scattered_train,
        scattered_test,
        scattered_validation
    )


def map_labels(vector):
    d = {}
    for item in vector:
        if item[1] in d:
            d[item[1]] += 1
        else:
            d[item[1]] = 1
    return d


def map_dataset():
    with open('./train-scattered', 'rb') as train_file, open('./test-scattered', 'rb') as test_file, open(
            './validation-scattered', 'rb') as validation_file:
        d = create_dataset_dictionary(
            pickle.load(train_file),
            pickle.load(test_file),
            pickle.load(validation_file)
        )
        for vector in d.values():
          print(map_labels(vector))


load_scattered_datasets()