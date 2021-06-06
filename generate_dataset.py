import os
import random
from time import sleep
from math import sqrt

import numpy as np
import soundfile as sf
import librosa
from scipy.io import wavfile
from python_speech_features import mfcc, fbank

from audio_utilities import FeatureExtraction

# Dataset Dir
noise_database_path = "Prototyping/Dataset Structure/Dataset/Noise"

speech_database_paths = ["Prototyping/Dataset Structure/Dataset/Speech/TIMIT CORPUS",
                         "Prototyping/Dataset Structure/Dataset/Speech/TSP",
                         "Prototyping/Dataset Structure/Dataset/Speech/MS_SNSD"]

save_directory = "Generated Features/"
feature_filename = ["feature_dataset_timit.npz", "feature_dataset_tsp.npz", "feature_dataset_ms_iter.npz"]
generate_from_dataset = [False, False, False]

feature_filename_ms = "feature_dataset_ms_librosa_vad.npz"
generate_from_dataset_ms = True

# Audio Configuration
sampling_rate = 16000
frame_length, window_length, hop_length = 1024, 1024, 512
window_function = "vorbis"
number_of_melbands, number_of_features = 22, 54

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


def normalize(data, n, quantize=True):
    limit = pow(2, n)
    data = np.clip(data, -limit, limit) / limit
    if quantize:
        data = np.round(data * 128) / 128.0
    return data


def get_melbands_gain(clean_speech_stft, noisy_speech_stft, melbands=22):
    clean_mel = audio_utils.get_melspectrogram(audio_stft=clean_speech_stft, number_of_melbands=melbands)
    noisy_mel = audio_utils.get_melspectrogram(audio_stft=noisy_speech_stft, number_of_melbands=melbands)

    gains_speech = np.sqrt(np.divide(clean_mel, noisy_mel))
    # gains_speech = np.where(gains_speech <= 0.01, 0.01, gains_speech)
    # gains_speech = np.divide(gains_speech, np.max(gains_speech))

    return gains_speech


def get_vad(clean_speech_stft, threshold=0.18):

    # Get Mel Spectrum
    _speech_mel_spectrum = audio_utils.get_melspectrogram(audio_stft=clean_speech_stft,
                                                          number_of_melbands=22)
    # _speech_log_spectrum = librosa.power_to_db(_speech_mel_spectrum)

    # Compute Sum of All Bands Per Frame
    energy_frames = np.sum(_speech_mel_spectrum, axis=0)
    # print(energy_frames.shape)

    # Define Required Variables
    voiced_speech = []
    end_offset = True
    start_offset = False

    # Iterate Through Frames
    for count, j in enumerate(energy_frames):
        if j > threshold:
            voiced_speech.append(1)

            # Add Offset of One Hop Length Before Threshold
            if start_offset:
                if count > 2 & count < len(energy_frames):
                    voiced_speech[count - 1] = 1
                    start_offset = False
            end_offset = True
        else:
            voiced_speech.append(0)

            # Add Offset of One Hop Length After Threshold
            if end_offset:
                if voiced_speech[count - 1] == 1:
                    voiced_speech[count] = 1
                    end_offset = False
            start_offset = True

    return np.array(voiced_speech)


def get_features(clean_speech, noisy_speech, melbands=22, delta_melbands=9):
    # Extract MFCC & Relative Derivatives
    noisy_speech_stft = audio_utils.stft(noisy_speech)
    # noisy_speech_mfcc = audio_utils.get_mfccs_from_spectrogram(noisy_speech_stft,
    #                                                            number_of_melbands=melbands)
    noisy_speech_mfcc = audio_utils.get_mfccs(noisy_speech, number_of_melbands=melbands)

    noisy_speech_mfcc_delta, noisy_speech_mfcc_delta2 = audio_utils.get_mfccs_delta(noisy_speech_mfcc,
                                                                                    number_of_melbands=delta_melbands)

    # Chrome CQT
    noisy_speech_chroma_cqt = audio_utils.get_chroma_cqt(noisy_speech)

    # Extract Spectral Centroid & Bandwidth
    noisy_speech_spec_centroid = audio_utils.get_spectral_centroid(audio_stft=noisy_speech_stft)
    noise_speech_spec_bandwidth = audio_utils.get_spectral_bandwidth(audio_stft=noisy_speech_stft)

    print(np.max(noisy_speech_mfcc), np.max(noisy_speech_mfcc_delta), np.max(noisy_speech_mfcc_delta2))
    print(np.min(noisy_speech_mfcc), np.min(noisy_speech_mfcc_delta), np.min(noisy_speech_mfcc_delta2))

    # Extract Gains
    speech_concat_stft = audio_utils.stft(clean_speech)
    speech_melband_gains = get_melbands_gain(clean_speech_stft=speech_concat_stft,
                                             noisy_speech_stft=noisy_speech_stft,
                                             melbands=melbands)

    return noisy_speech_mfcc, noisy_speech_mfcc_delta, noisy_speech_mfcc_delta2, noisy_speech_spec_centroid, noise_speech_spec_bandwidth, noisy_speech_chroma_cqt, speech_melband_gains


def generate_dataset(noise_dir, speech_dir, snr=None, use_random_snr=False, normalize=False):
    # Define Array Features
    generated_features_speech = np.ndarray((number_of_features, 0))
    generated_features_gain = np.ndarray((number_of_melbands, 0))

    if snr is None:
        snr = [-5, 0, 5]

    noise_file_paths = get_filepaths(dataset_dir=noise_dir, get_duration=True)
    speech_file_paths = get_filepaths(dataset_dir=speech_dir, get_duration=True)

    sleep(5)

    speech_iterator, noise_iterator = 0, 0

    try:
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
            print("[{}]".format(noise_iterator), "Currently Used Noise:", noise_file_paths[noise_iterator],
                  len(noise_file))

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
            noisy_speech = np.array([])
            random_snr = 0

            if use_random_snr:
                random_snr = random.randint(0, len(snr_req) - 1)
                noisy_speech = add_noise_speech(speech_concat, noise_file, snr=snr_req[random_snr])
                print("Concat Speech Shape : {}, Noisy Speech Shape: {}"
                      .format(speech_concat.shape, noisy_speech.shape))

            else:
                # Add Noise with every SNR
                for snr in snr_req:
                    noisy_speech_temp = add_noise_speech(speech_concat, noise_file, snr=snr)
                    noisy_speech = np.concatenate((noisy_speech, noisy_speech_temp))

                # Concat Speech to Match Noisy Speech Length
                speech_concat_temp = speech_concat
                for j in range(0, len(snr_req) - 1):
                    speech_concat = np.concatenate((speech_concat, speech_concat_temp))

                print("Concat Speech Shape : {}, Noisy Speech Shape: {}"
                      .format(speech_concat.shape, noisy_speech.shape))

            # # Speech Loaded in Multiple of Window Length
            # speech_length = int(len(speech_concat) / frame_length)
            # noise_length = int(len(noisy_speech) / frame_length)
            #
            # # Truncate to the Multiple of Window Length
            # speech_concat = speech_concat[:speech_length * frame_length]
            # noisy_speech = noisy_speech[:noise_length * frame_length]

            # Get Features
            _mfcc, mfcc_d, mfcc_d2, spec_centroid, spec_bandwidth, chroma_cqt, gains = get_features(
                clean_speech=speech_concat,
                noisy_speech=noisy_speech
            )

            # print(mfcc.shape[1], mfcc_d.shape[1], mfcc_d.shape[1],
            #       spec_bandwidth.shape[1], spec_bandwidth.shape[1], gains.shape[1])

            # Truncate to Minimum Length
            length_features = [_mfcc.shape[1], spec_bandwidth.shape[1], spec_bandwidth.shape[1],
                               chroma_cqt.shape[1], gains.shape[1]]
            min_length = min(length_features)

            # Add Features to Array
            features = np.concatenate((_mfcc[:, :min_length], mfcc_d[:, :min_length], mfcc_d2[:, :min_length],
                                       spec_bandwidth[:, :min_length], spec_centroid[:, :min_length],
                                       chroma_cqt[:, :min_length]), axis=0)
            gains = gains[:, :min_length]

            generated_features_speech = np.concatenate((generated_features_speech, features), axis=1)
            generated_features_gain = np.concatenate((generated_features_gain, gains), axis=1)

            print("[{}]".format(speech_iterator), "Added Noise {}dB to Speech: "
                  .format(snr_req[random_snr] if use_random_snr else snr_req),
                  generated_features_speech.shape, generated_features_gain.shape, "\n")

    except KeyboardInterrupt:
        print("Generation was Interrupted")

    if normalize:
        print("Normalizing Generated Features")
        generated_features_speech = librosa.util.normalize(generated_features_speech)
        # generated_features_gain = util.normalize(generated_features_gain)

    return generated_features_speech, generated_features_gain


def generate_dataset_ms(noisy_speech_dir, clean_speech_dir, snr_levels=4):
    clean_speech_files = get_filepaths(
        dataset_dir=clean_speech_dir,
        get_duration=True
    )
    noise_speech_files = get_filepaths(
        dataset_dir=noisy_speech_dir,
        get_duration=True
    )

    # mfccs = np.ndarray((0, number_of_melbands))
    # gains = np.ndarray((0, number_of_melbands))
    mfccs = np.ndarray((number_of_melbands, 0))
    gains = np.ndarray((number_of_melbands, 0))
    # spec_centroid = np.ndarray((0, 1))
    # spec_bandwidth = np.ndarray((0, 1))
    # total_energy = np.array([])
    vad = np.array([])

    clean_speech_iterator = 0

    for clean_speech_file in clean_speech_files:

        # Get filename without extension .wav
        filename = clean_speech_file.split("/")[-1]

        print("[{}] Clean Speech File: {}".format(clean_speech_iterator, filename))
        clean_speech_iterator += 1

        # Find corresponding files from noisy speech
        _temp_noisy_files = []
        for noisy_speech_file in noise_speech_files:
            if filename in noisy_speech_file:
                _temp_noisy_files.append(noisy_speech_file)

        # Check Whether Array Has 4 SNR Files
        if len(_temp_noisy_files) != snr_levels:
            print("Couldn't get {} Noisy Files for {}".format(snr_levels, clean_speech_file))
            break

        # Load Speech
        (rate, _speech) = wavfile.read(clean_speech_file)
        _speech = _speech / 32768

        # Take STFT
        _speech_stft = audio_utils.stft(_speech)

        # Find VAD
        _vad = get_vad(_speech_stft, threshold=0.18)

        # Compute Band Energy of Clean Speech
        # _band_energy_speech, _total_energy_speech = fbank(
        #     signal=_speech, samplerate=16000, winlen=0.032, winstep=0.016,
        #     nfft=1024, nfilt=22, lowfreq=20, highfreq=8000
        # )

        # Iterate through Noisy Speech
        noisy_speech_iterator = 0
        for file in _temp_noisy_files:
            # Load Noisy Signal
            (rate, _noise) = wavfile.read(file)
            _noise = _noise / 32768

            # Take STFT
            _noise_stft = audio_utils.stft(_noise)

            # Compute MFCC
            # _mfcc = mfcc(
            #     signal=_noise, samplerate=16000, winlen=0.032, winstep=0.016,
            #     nfft=1024, nfilt=22, numcep=22, lowfreq=20, highfreq=8000
            # )
            _mfcc = audio_utils.get_mfccs_from_spectrogram(audio_stft=_noise_stft,
                                                           number_of_melbands=number_of_melbands)

            # Compute Band Energy of Noisy Speech
            # _band_energy_noise, _total_energy_noise = fbank(
            #     signal=_noise, samplerate=16000, winlen=0.032, winstep=0.016,
            #     nfft=1024, nfilt=22, lowfreq=20, highfreq=8000
            # )

            _gains = get_melbands_gain(clean_speech_stft=_speech_stft, noisy_speech_stft=_noise_stft,
                                       melbands=22)

            # Compute the Gain of Bands
            # _gains = np.sqrt(np.divide(_band_energy_speech, _band_energy_noise))
            # _gains = np.clip(_gains, 0, 1)

            #  # Extract Spectral Centroid & Bandwidth
            # _stft = audio_utils.stft(_noise)
            # _spec_centroid = audio_utils.get_spectral_centroid(audio_stft=_stft)
            # _spec_bandwidth = audio_utils.get_spectral_bandwidth(audio_stft=_stft)

            # Append MFCC and Gains
            mfccs = np.concatenate((mfccs, _mfcc), axis=1)
            gains = np.concatenate((gains, _gains), axis=1)
            # total_energy = np.concatenate((total_energy, _total_energy_noise))
            # spec_centroid = np.concatenate((spec_centroid, _spec_centroid.T))
            # spec_bandwidth = np.concatenate((spec_bandwidth, _spec_bandwidth.T))
            vad = np.concatenate((vad, _vad))

            print("[{}] Used Noisy Signal: {}".format(noisy_speech_iterator, file.split("/")[-1]))
            noisy_speech_iterator += 1

        print(mfccs.shape, gains.shape, vad.shape)

    # Normalize MFCCs
    # mfccs = normalize(data=mfccs, n=3, quantize=False)

    # Compute Delta and Concat
    # gains = gains.T
    # mfccs = mfccs.T
    mfcc_d, mfcc_d2 = audio_utils.get_mfccs_delta(mfccs, number_of_melbands=9)
    mfccs = librosa.util.normalize(mfccs)
    mfcc_d = librosa.util.normalize(mfcc_d)
    mfcc_d2 = librosa.util.normalize(mfcc_d2)
    _features = np.concatenate((mfccs, mfcc_d, mfcc_d2))

    # print(spec_centroid.shape, spec_bandwidth.shape)
    # spectral_features = np.concatenate((spec_centroid, spec_bandwidth))

    print(_features.shape, gains.shape, vad.shape)

    return _features, gains, vad


if __name__ == "__main__":
    # Generate Dataset

    # Iterate through speech
    for i in range(len(speech_database_paths)):

        # Check boolean for generation from a particular path
        if generate_from_dataset[i]:
            print("\nGenerating from {}".format(speech_database_paths[i]))

            # Define Feature Arrays
            # features_speech = np.ndarray((number_of_features, 0))
            # features_gain = np.ndarray((number_of_melbands, 0))

            # Save even if interrupted
            features_speech, features_gain = generate_dataset(noise_dir=noise_database_path,
                                                              speech_dir=speech_database_paths[i],
                                                              snr=snr_req, use_random_snr=True,
                                                              normalize=True)

            # Save to File
            print("\nSaving To File {}\n".format(save_directory + feature_filename[i]))
            print("Shape: {} & {}".format(features_speech.shape, features_gain.shape))

            np.savez_compressed(save_directory + feature_filename[i],
                                speech_features=features_speech, gains=features_gain)

            # Clear Memory
            features_speech, features_gain = 0, 0
        else:
            print("\nSkipping {}".format(speech_database_paths[i]))

    # Generate from Scalable Database Directories
    if generate_from_dataset_ms:
        # Microsoft Scalable Database Directories
        ms_noisy_dataset = "Prototyping/Dataset Structure/Dataset/MS/NoisySpeech_training"
        ms_clean_dataset = "Prototyping/Dataset Structure/Dataset/MS/CleanSpeech_training"

        print("\nGenerating from {}".format(ms_noisy_dataset))

        speech_features, band_gains, voiced = generate_dataset_ms(
            noisy_speech_dir=ms_noisy_dataset, clean_speech_dir=ms_clean_dataset, snr_levels=4
        )

        print("\nSaving as {} in {}".format(feature_filename_ms, save_directory))
        np.savez_compressed(save_directory + feature_filename_ms,
                            speech_features=speech_features,
                            gains=band_gains,
                            vad=voiced
                            )

    print("\nGeneration Completed")

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
