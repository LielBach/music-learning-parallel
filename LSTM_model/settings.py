#525 ms of audio in 22050 hz
frame_length = 8158
mfcc_length = 320
SAMPLE_DATA = 0
SAMPLE_LABEL = 1
AUTO_ENCODER_PATH = 'auto_encoder'
CNN = 'model.hdf5'
LSTM_NET = 'feed_forward'
LSTM_NO_ATTENTION_NET = 'feed_forward_no_attention'
FINAL_NET = 'final_net'
K = 0.6
#30  seconds of audio in 22050 hz
TRACK_SIZE = 661794
NUM_FRAMES = 80
NUM_GENRES = 10
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.1
NUM_SCATT_FEAT = 630
LSTM_FEATURES = "./lstm_features"