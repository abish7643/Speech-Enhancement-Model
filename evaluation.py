import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import matplotlib.pyplot as plt
import librosa.display
from audio_utilities import FeatureExtraction
from librosa import util
from equalizer import equalize_noisy_signal
from generate_dataset import normalize, get_melbands_gain
from scipy.io import wavfile
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model
from pystoi import stoi
from pypesq import pesq

# Load Model
saved_model_dir = "Assets/Trained Model"
saved_model = "model_vad.h5"

# Evaluate Files
eval_clean_speech_dir = "Assets/Datasets/MS/CleanSpeech_training"
eval_noisy_speech_dir = "Assets/Datasets/MS/NoisySpeech_training"
eval_clean_speech_file = "clnsp14.wav"
eval_noisy_speech_file = "noisy14_SNRdb_15.0_clnsp14.wav"

# Evaluate Batch
snr_req = [25, 20, 15, 10]
STOI, PESQ, eval_batch = [], [], False
# eval_batch_clean_files = ["clnsp3.wav", "clnsp14.wav", "clnsp9.wav", "clnsp13.wav",
#                           "clnsp3.wav", "clnsp4.wav", "clnsp4.wav", "clnsp4.wav"]
# eval_batch_noise_labels = ["babble", "engine", "white", "hammer",
#                            "nature", "rock", "cafeteria", "traffic"]
eval_batch_clean_files = ["clnsp14.wav", "clnsp9.wav", "clnsp4.wav"]
# eval_batch_clean_files = ["clnsp15.wav", "clnsp1.wav", "clnsp12.wav"]
eval_batch_noise_labels = ["engine", "white", "traffic"]
# eval_batch_noise_labels = ["engine", "white", "traffic"]
eval_batch_noisy_files = []

# Save Audio File
save_wav_dir = "Assets/Generated Features/Equalized Audio"
save_noisy_speech_file = "noisy_speech_original.wav"
save_noisy_speech_eq_file = "noisy_speech_equalized.wav"

# Audio Configurations
sampling_rate = 16000
frame_length, window_length, hop_length = 1024, 1024, 512
window_function = "vorbis"
number_of_melbands, number_of_features = 22, 40
audio_utils = FeatureExtraction(sampling_rate=sampling_rate,
                                frame_length=frame_length, hop_length=hop_length,
                                window_length=window_length, window_function=window_function)


def get_stoi_pesq(filtered_speech, original_speech, sampling_rate=16000):
    # Compute STOI
    stoi_value = stoi(x=original_speech, y=filtered_speech, fs_sig=sampling_rate, extended=False)

    # Compute PESQ
    pesq_value = pesq(ref=original_speech, deg=filtered_speech, fs=sampling_rate)

    return stoi_value, pesq_value


def plot_score(score, noise_type, snr=None, title="", save=True):
    if snr is None:
        snr = [0, 5, 10, 15]

    # Reshaping to SNR Levels x Score
    noise_count = int(len(score)/len(snr))
    score = np.array(score)
    score = score.reshape(noise_count, len(snr))

    plt.figure()
    plt.plot(snr, score.T)
    plt.xticks(snr, snr)
    plt.legend(noise_type)
    plt.xlabel("SNR Levels (dB)")
    plt.ylabel("Score")
    plt.title(title)
    if save:
        plt.savefig("".join(["../", title, ".png"]), bbox_inches='tight')
    plt.show()


def plot_gains(feature, sr=16000, title="", save=False):
    plt.figure(figsize=(18, 5))
    librosa.display.specshow(data=feature, sr=sr,
                             x_axis="time", y_axis="mel")
    plt.colorbar(format="%+.2f")
    plt.title((title + " - {}").format(len(feature)))
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    if save:
        plt.savefig("".join(["../", title, ".png"]), bbox_inches='tight')
    plt.show()
    return True


def get_features(clean_speech_file, noisy_speech_file, truncate_length=None):
    # Load Noise and Speech
    (_rate, _clean_speech) = wavfile.read(clean_speech_file)
    (_rate, _noisy_speech) = wavfile.read(noisy_speech_file)

    if truncate_length is not None:
        _clean_speech = _clean_speech[:truncate_length * sampling_rate]
        _noisy_speech = _noisy_speech[:truncate_length * sampling_rate]

    # Scale between -1 to 1
    _clean_speech = _clean_speech / 32768
    _noisy_speech = _noisy_speech / 32768

    # Compute STFT
    _clean_speech_stft = audio_utils.stft(audio=_clean_speech)
    _noisy_speech_stft = audio_utils.stft(audio=_noisy_speech)

    # # Calculate Ideal Gains (For Reference)
    # _band_energy_speech, _total_energy_speech = fbank(
    #     signal=_clean_speech, samplerate=sampling_rate, winlen=0.032, winstep=0.016,
    #     nfft=1024, nfilt=22, lowfreq=20, highfreq=8000
    # )
    # _band_energy_noise, _total_energy_noise = fbank(
    #     signal=_noisy_speech, samplerate=sampling_rate, winlen=0.032, winstep=0.016,
    #     nfft=1024, nfilt=22, lowfreq=20, highfreq=8000
    # )
    # _gains = np.sqrt(np.divide(_band_energy_speech, _band_energy_noise))
    #
    # # Calculate MFCC
    # _mfccs = mfcc(
    #     signal=_noisy_speech, samplerate=16000, winlen=0.032, winstep=0.016,
    #     nfft=1024, nfilt=22, numcep=22, lowfreq=20, highfreq=8000
    # )
    #
    # # Normalize
    # _mfccs = normalize(data=_mfccs, n=3)
    # _mfccs = _mfccs.T

    _gains = get_melbands_gain(clean_speech_stft=_clean_speech_stft,
                               noisy_speech_stft=_noisy_speech_stft,
                               melbands=22)
    _gains = np.clip(_gains, 0, 1)

    _mfccs = audio_utils.get_mfccs_from_spectrogram(_noisy_speech_stft,
                                                    number_of_melbands=22)

    # # Get Delta
    # _mfccs_d, _mfccs_d2 = audio_utils.get_mfccs_delta(_mfccs, number_of_melbands=9)
    # _generated_features = np.concatenate((_mfccs, _mfccs_d, _mfccs_d2))
    # print("Generated Features: {}".format(_generated_features.shape))

    _mfccs_d, _mfccs_d2 = audio_utils.get_mfccs_delta(_mfccs, number_of_melbands=9)

    _spec_centroid = audio_utils.get_spectral_centroid(audio_stft=_noisy_speech_stft)
    _spec_bandwidth = audio_utils.get_spectral_bandwidth(audio_stft=_noisy_speech_stft)
    _spec = np.concatenate((_spec_centroid, _spec_bandwidth))

    _mfccs = util.normalize(_mfccs)
    _mfccs_d = util.normalize(_mfccs_d)
    _mfccs_d2 = util.normalize(_mfccs_d2)
    _spec = util.normalize(_spec)

    _generated_features = np.concatenate((_mfccs, _mfccs_d, _mfccs_d2, _spec))
    _generated_features = _generated_features[:number_of_features]
    print("Generated Features: {}".format(_generated_features.shape))

    _generated_features = _generated_features.T

    # Reshape
    _window_size = _mfccs.shape[1]
    _number_of_sequences = int(_generated_features.shape[0] / _window_size)
    _generated_features = _generated_features[:_number_of_sequences * _window_size]
    _generated_features = np.reshape(
        _generated_features, (_window_size, _number_of_sequences, number_of_features)
    )
    print("Modified Shape to {} with Window Size {}".format(_generated_features.shape,
                                                            _window_size))

    return _generated_features, _gains, _clean_speech, _noisy_speech


def predict_equalize(clean_speech_file, noisy_speech_file, visualize=False, save=False, save_audio=False):
    print("\nLoading Model {} from {}".format(saved_model_dir, saved_model))

    _model = load_model(filepath="/".join([saved_model_dir, saved_model]))
    print("Model Loaded")

    # Model Summary
    if visualize:
        _model.summary()

    # Generate Features
    print("\nGenerating Features")
    generated_features, gains_ref, clean_speech, noisy_speech = get_features(
        clean_speech_file=clean_speech_file,
        noisy_speech_file=noisy_speech_file,
        truncate_length=5
    )
    print("Generated Features : {}".format(generated_features.shape))

    # Predict Gains Using Model
    print("\nPredicting Gains")
    _prediction = _model.predict(generated_features, batch_size=32)
    # print("Predicted Gains : {}".format(gains_predicted.shape))

    gains_predicted = _prediction[0]
    vad_predicted = _prediction[1]
    print("Predicted Gains : {}, VAD : {}".format(gains_predicted.shape, vad_predicted.shape))

    # Reshape
    window_size = gains_predicted.shape[0]
    number_of_sequences = gains_predicted.shape[1]
    gains_predicted = np.reshape(
        gains_predicted, (window_size * number_of_sequences, 22)
    )
    vad_predicted = np.reshape(
        vad_predicted, (window_size * number_of_sequences, 1)
    )
    print("Reshaped Gains : {}, VAD : {}".format(gains_predicted.shape, vad_predicted.shape))
    # print(vad_predicted)

    # Clip Gains
    gains_predicted = np.clip(gains_predicted, 0, 1)

    # Make VAD 0 or 1
    # _vad_threshold = 0.47
    # vad_predicted = np.where(vad_predicted <= _vad_threshold, 0, vad_predicted)
    # vad_predicted = np.where(vad_predicted > _vad_threshold, 1, vad_predicted)

    # Apply VAD to Gains
    gains_predicted_vad = gains_predicted * vad_predicted

    # Equalize
    equalized_signal = equalize_noisy_signal(
        noisy_speech=noisy_speech, gains=gains_predicted_vad,
        sampling_rate=sampling_rate, hop_length=hop_length
    )
    ideal_equalized_signal = equalize_noisy_signal(
        noisy_speech=noisy_speech, gains=gains_ref.T,
        sampling_rate=sampling_rate, hop_length=hop_length
    )
    print("Equalized Signal : {}".format(equalized_signal.shape))

    # Saving
    if save_audio:
        print("Saving to {} as {} & {}".format(
            save_wav_dir, save_noisy_speech_eq_file, save_noisy_speech_file)
        )
        sf.write("/".join([save_wav_dir, save_noisy_speech_eq_file]),
                 equalized_signal, samplerate=sampling_rate)
        sf.write("/".join([save_wav_dir, save_noisy_speech_file]),
                 noisy_speech, samplerate=sampling_rate)

    # Get STOI and PESQ
    stoi_eq, pesq_eq = get_stoi_pesq(
        equalized_signal, noisy_speech, sampling_rate=sampling_rate
    )
    if eval_batch:
        STOI.append(stoi_eq)
        PESQ.append(pesq_eq)
    else:
        print("STOI : {}, PESQ: {}".format(stoi_eq, pesq_eq))

    if visualize:
        # Spectrum
        noisy_speech_stft = audio_utils.stft(audio=noisy_speech, title="Noisy Speech",
                                             visualize=True, save=save)
        equalized_signal_stft = audio_utils.stft(audio=equalized_signal, title="Equalized Speech",
                                                 visualize=True, save=save)
        # ideal_equalized_signal_stft = audio_utils.stft(audio=ideal_equalized_signal, title="Ideal Equalized Speech",
        #                                                visualize=True, save=save)
        # clean_speech_stft = audio_utils.stft(audio=clean_speech, title="Clean Speech",
        #                                      visualize=True, save=save)

        # plt.figure(figsize=(18, 5))
        # plt.plot(vad_predicted)
        # plt.show()

        # plot_gains(gains_ref, title="Gains", save=save)
        # plot_gains(gains_predicted.T, title="Predicted Gains", save=save)
        # plot_gains(gains_predicted_vad.T, title="Predicted Gains With VAD", save=save)


if __name__ == "__main__":
    if not eval_batch:
        predict_equalize(clean_speech_file="/".join([eval_clean_speech_dir, eval_clean_speech_file]),
                         noisy_speech_file="/".join([eval_noisy_speech_dir, eval_noisy_speech_file]),
                         visualize=True, save=False, save_audio=True
                         )
    else:
        # Create Noise Filenames
        for file in eval_batch_clean_files:
            for _snr in snr_req:
                noisy_file_template = "noisy{}_SNRdb_{}.0_{}".format(
                    file.replace("clnsp", "").split(".")[0], _snr, file
                )
                eval_batch_noisy_files.append(noisy_file_template)

        print("Clean Speech Files : {}, Noisy Speech Files : {}".format(
            len(eval_batch_clean_files), len(eval_batch_noisy_files)
        ))

        # Iterate Through Noise Files
        for noise_file in eval_batch_noisy_files:
            clean_file_no = math.floor(len(eval_batch_clean_files) / len(snr_req))
            clean_file = eval_batch_clean_files[clean_file_no]
            predict_equalize(clean_speech_file="/".join([eval_clean_speech_dir, clean_file]),
                             noisy_speech_file="/".join([eval_noisy_speech_dir, noise_file]),
                             visualize=False, save=False
                             )

        # Plot Scores
        print("STOI : {}\nPESQ : {}".format(STOI, PESQ))
        plot_score(noise_type=eval_batch_noise_labels,
                   score=STOI, snr=[15, 10, 5, 0], title="STOI")
        plot_score(noise_type=eval_batch_noise_labels,
                   score=PESQ, snr=[15, 10, 5, 0], title="PESQ")

