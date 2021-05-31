from python_speech_features import fbank, mfcc
from audio_utilities import FeatureExtraction
from librosa import util
from equalizer import equalize_noisy_signal
from generate_dataset import normalize, get_melbands_gain
from scipy.io import wavfile
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model
from pystoi import stoi

# from pesq import pesq

# Load Model
saved_model_dir = "Prototyping/Trained Model"
saved_model = "model.h5"

# Evaluate Files
eval_clean_speech_dir = "Prototyping/Dataset Structure/Dataset/MS/CleanSpeech_training"
eval_noisy_speech_dir = "Prototyping/Dataset Structure/Dataset/MS/NoisySpeech_training"
eval_clean_speech_file = "clnsp14.wav"
eval_noisy_speech_file = "noisy14_SNRdb_25.0_clnsp14.wav"

# Save Audio File
save_wav_dir = "Generated Features"
save_noisy_speech_file = "noisy_speech_original.wav"
save_noisy_speech_eq_file = "noisy_speech_equalized.wav"

# Audio Configurations
sampling_rate = 16000
frame_length, window_length, hop_length = 1024, 1024, 512
window_function = "vorbis"
number_of_melbands, number_of_features = 22, 40
snr_req = [-5, 0, 5, 10]
audio_utils = FeatureExtraction(sampling_rate=sampling_rate,
                                frame_length=frame_length, hop_length=hop_length,
                                window_length=window_length, window_function=window_function)


def get_stoi_pesq(filtered_speech, original_speech, sampling_rate=16000):
    # Compute STOI
    stoi_value = stoi(x=original_speech, y=filtered_speech, fs_sig=sampling_rate, extended=False)

    # Compute PESQ
    # pesq_value = pesq(ref=original_speech, deg=filtered_speech, fs=sampling_rate, mode='wb')

    return stoi_value, 0


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

    _mfccs = util.normalize(_mfccs)
    _mfccs_d = util.normalize(_mfccs_d)
    _mfccs_d2 = util.normalize(_mfccs_d2)

    _generated_features = np.concatenate((_mfccs, _mfccs_d, _mfccs_d2))
    print("Generated Features: {}".format(_generated_features.shape))

    # Reshape
    _window_size = _mfccs.shape[1]
    _number_of_sequences = int(_mfccs.shape[1] / _window_size)
    _generated_features = _generated_features[:_number_of_sequences * _window_size]
    _generated_features = np.reshape(
        _generated_features, (_window_size, _number_of_sequences, 40)
    )
    print("Modified Shape to {} with Window Size {}".format(_generated_features.shape,
                                                            _window_size))

    return _generated_features, _gains, _clean_speech, _noisy_speech


if __name__ == "__main__":
    print("\nLoading Model {} from {}".format(saved_model_dir, saved_model))

    _model = load_model(filepath="/".join([saved_model_dir, saved_model]))
    print("Model Loaded")

    # Model Summary
    _model.summary()

    # Generate Features
    print("\nGenerating Features")
    generated_features, gains_ref, clean_speech, noisy_speech = get_features(
        clean_speech_file="/".join([eval_clean_speech_dir, eval_clean_speech_file]),
        noisy_speech_file="/".join([eval_noisy_speech_dir, eval_noisy_speech_file]),
        truncate_length=10
    )
    print("Generated Features : {}".format(generated_features.shape))

    # Predict Gains Using Model
    print("\nPredicting Gains")
    gains_predicted = _model.predict(generated_features, batch_size=32)
    print("Predicted Gains : {}".format(gains_predicted.shape))

    # Reshape
    window_size = gains_predicted.shape[0]
    number_of_sequences = gains_predicted.shape[1]
    gains_predicted = np.reshape(
        gains_predicted, (window_size * number_of_sequences, 22)
    )
    print("Reshaped Gains : {}".format(gains_predicted.shape))

    # Clip Gains
    gains_predicted = np.clip(gains_predicted, 0, 1)

    # Equalize
    equalized_signal = equalize_noisy_signal(
        noisy_speech=noisy_speech, gains=gains_predicted,
        sampling_rate=sampling_rate, hop_length=512
    )
    print("Equalized Signal : {}".format(equalized_signal.shape))

    # Saving
    print("Saving to {} as \n{} & {}".format(
        save_wav_dir, save_noisy_speech_eq_file, noisy_speech)
    )
    sf.write("/".join([save_wav_dir, save_noisy_speech_eq_file]),
             equalized_signal, samplerate=sampling_rate)
    sf.write("/".join([save_wav_dir, save_noisy_speech_file]),
             noisy_speech, samplerate=sampling_rate)

    # Get STOI and PESQ
    stoi_eq, pesq_eq = get_stoi_pesq(
        equalized_signal, noisy_speech, sampling_rate=sampling_rate
    )

    print("STOI : {}, PESQ: {}".format(stoi_eq, pesq_eq))
