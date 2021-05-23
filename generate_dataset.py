import os
import random
from time import sleep
from math import sqrt

import numpy as np
import soundfile as sf

from audio_utilities import FeatureExtraction

# Dataset Dir
noise_database_path = "Prototyping/Dataset Structure/Dataset/Noise"
speech_database_path_timit = "Prototyping/Dataset Structure/Dataset/Speech/TIMIT CORPUS"
speech_database_path_tsp = "Prototyping/Dataset Structure/Dataset/Speech/TSP"

save_directory = "Generated Features/"
feature_filename = ["feature_dataset_timit.npz", "feature_dataset_tsp.npz"]

# Audio Configuration
sampling_rate = 16000
frame_length, window_length, hop_length = 1024, 1024, 512
window_function = "vorbis"
number_of_melbands, number_of_features = 22, 42

# Required SNR
snr_req = [-5, 0, 5]

audio_utils = FeatureExtraction(sampling_rate=sampling_rate,
                                frame_length=frame_length, hop_length=hop_length,
                                window_length=window_length, window_function=window_function)


def get_filepaths(dataset_dir, filter_format="wav", get_duration=False):
    filepath_array = []
    speech_duration = 0

    for i, (path, dir_name, file_name) in enumerate(os.walk(dataset_dir)):
        for file in file_name:
            if filter_format in file:
                filepath = path + "/" + file
                filepath_array.append(filepath)

                if get_duration:
                    load_file = sf.SoundFile(filepath)
                    speech_duration += len(load_file) / load_file.samplerate

    print("Got {} .{} Files".format(len(filepath_array), filter_format))
    if get_duration:
        print("Total Duration: {}m".format(int(speech_duration / 60)))

    return filepath_array


def add_noise_speech(speech, noise, snr=5):
    rms_speech = sqrt(np.mean(speech ** 2))
    rms_noise_req = sqrt(rms_speech ** 2 / pow(10, snr / 10))

    rms_noise = sqrt(np.mean(noise ** 2))
    noise_mod = noise * (rms_noise / rms_noise_req)

    return speech + noise_mod


def get_melbands_gain(clean_speech_stft, noisy_speech_stft, melbands=22):
    clean_mel = audio_utils.get_melspectrogram(audio_stft=clean_speech_stft, number_of_melbands=melbands)
    noisy_mel = audio_utils.get_melspectrogram(audio_stft=noisy_speech_stft, number_of_melbands=melbands)

    gains_speech = np.sqrt(np.divide(clean_mel, noisy_mel))
    # gains_speech = np.where(gains_speech <= 0.01, 0.01, gains_speech)
    # gains_speech = np.divide(gains_speech, np.max(gains_speech))

    return gains_speech


def get_features(clean_speech, noisy_speech, melbands=22, delta_melbands=9):
    # Extract MFCC & Relative Derivatives
    noisy_speech_stft = audio_utils.stft(noisy_speech)
    noisy_speech_mfcc = audio_utils.get_mfccs_from_spectrogram(noisy_speech_stft,
                                                               number_of_melbands=melbands)
    noisy_speech_mfcc_delta, noisy_speech_mfcc_delta2 = audio_utils.get_mfccs_delta(noisy_speech_mfcc,
                                                                                    number_of_melbands=delta_melbands)

    # Extract Spectral Centroid & Bandwidth
    noisy_speech_spec_centroid = audio_utils.get_spectral_centroid(audio_stft=noisy_speech_stft)
    noise_speech_spec_bandwidth = audio_utils.get_spectral_bandwidth(audio_stft=noisy_speech_stft)

    # Extract Gains
    speech_concat_stft = audio_utils.stft(clean_speech)
    speech_melband_gains = get_melbands_gain(clean_speech_stft=speech_concat_stft,
                                             noisy_speech_stft=noisy_speech_stft,
                                             melbands=melbands)

    return noisy_speech_mfcc, noisy_speech_mfcc_delta, noisy_speech_mfcc_delta2, noisy_speech_spec_centroid, noise_speech_spec_bandwidth, speech_melband_gains


def generate_dataset(noise_dir, speech_dir, snr=None):
    # Define Array Features
    features_speech = np.ndarray((number_of_features, 0))
    features_gain = np.ndarray((number_of_melbands, 0))

    if snr is None:
        snr = [-5, 0, 5]

    noise_file_paths = get_filepaths(dataset_dir=noise_dir, get_duration=True)
    speech_file_paths = get_filepaths(dataset_dir=speech_dir, get_duration=True)

    sleep(5)

    speech_iterator, noise_iterator = 0, 0

    while speech_iterator < len(speech_file_paths):

        # Iterate Through Noise
        noise_iterator += 1

        # Reset Noise Iterator
        if noise_iterator >= len(noise_file_paths):
            noise_iterator = 0

        # Define Concatenated Speech
        speech_concat = np.array([])

        # Load Noise
        noise_file = audio_utils.load_audiofile(noise_file_paths[noise_iterator])
        print("[{}]".format(noise_iterator), "Currently Used Noise:", noise_file_paths[noise_iterator], len(noise_file))

        while len(speech_concat) < len(noise_file):
            # Load Speech
            speech_file = audio_utils.load_audiofile(speech_file_paths[speech_iterator])
            print("Currently Used Speech:", speech_file_paths[speech_iterator], len(speech_file))

            # Concat Speech
            speech_concat = np.concatenate((speech_concat, speech_file))
            speech_iterator += 1

            if speech_iterator >= len(speech_file_paths):
                break

        # Truncate Speech Array to Noise Length
        if len(speech_concat) >= len(noise_file):
            speech_concat = speech_concat[:len(noise_file)]
        else:
            noise_file = noise_file[:len(speech_concat)]

        # Add Noise to Speech
        random_snr = random.randint(0, len(snr_req) - 1)
        noisy_speech = add_noise_speech(speech_concat, noise_file, snr=snr_req[random_snr])

        # Get Features
        mfcc, mfcc_d, mfcc_d2, spec_centroid, spec_bandwidth, gains = get_features(clean_speech=speech_concat,
                                                                                   noisy_speech=noisy_speech)

        # Add Features to Array
        features = np.concatenate((mfcc, mfcc_d, mfcc_d2,
                                   spec_bandwidth, spec_centroid), axis=0)

        features_speech = np.concatenate((features_speech, features), axis=1)
        features_gain = np.concatenate((features_gain, gains), axis=1)

        print("[{}]".format(speech_iterator), "Added Noise to Speech: ",
              features_speech.shape, features_gain.shape, "\n")

    return features_speech, features_gain


if __name__ == "__main__":
    # Generate Dataset

    # =================TIMIT=================

    # Define Feature Arrays
    features_speech_1 = np.ndarray((number_of_features, 0))
    features_gain_1 = np.ndarray((number_of_melbands, 0))

    # Save even if interrupted
    try:
        features_speech_1, features_gain_1 = generate_dataset(noise_dir=noise_database_path,
                                                              speech_dir=speech_database_path_timit,
                                                              snr=snr_req)
    except KeyboardInterrupt:
        print("Generation was Interrupted")

    # Save to File
    print("\nSaving To File {}\n".format(save_directory + feature_filename[0]))
    print("Shape: {} & {}".format(features_speech_1.shape, features_gain_1.shape))

    np.savez_compressed(save_directory + feature_filename[0],
                        speech_features=features_speech_1, gains=features_gain_1)

    # Clear Memory
    features_speech_1, features_gain_1 = 0, 0

    # =======================================

    # ==================TSP==================

    # Define Feature Arrays
    features_speech_2 = np.ndarray((number_of_features, 0))
    features_gain_2 = np.ndarray((number_of_melbands, 0))

    # Save even if interrupted
    try:
        features_speech_2, features_gain_2 = generate_dataset(noise_dir=noise_database_path,
                                                              speech_dir=speech_database_path_tsp,
                                                              snr=snr_req)
    except KeyboardInterrupt:
        print("Generation was Interrupted")

    # Save to File
    print("\nSaving To File {}\n".format(save_directory + feature_filename[1]))
    print("Shape: {} & {}".format(features_speech_2.shape, features_gain_2.shape))

    np.savez_compressed(save_directory + feature_filename[1],
                        speech_features=features_speech_2, gains=features_gain_2)

    # Clear Memory
    features_speech_2, features_gain_2 = 0, 0

    # =======================================

# ==============================================

# # Load from file
# print("\nLoading from File\n")
#
# # Clear Variables
# features_speech_1, features_gain_1 = 0, 0
# features_speech_2, features_gain_2 = 0, 0
#
# with np.load(filename) as data:
#     speech_features_1 = data["speech_features_1"]
#     gains_1 = data["gains_1"]
#     speech_features_2 = data["speech_features_2"]
#     gains_2 = data["gains_2"]
#
#     print("Training Set: {} & {}".format(speech_features_1.shape, gains_1.shape))
#     print("Test Set: {} & {}".format(speech_features_2.shape, gains_2.shape))

# ==============================================
