from audio_utilities import FeatureExtraction
import warnings
import librosa
import numpy as np

warnings.filterwarnings("ignore")

sampling_rate = 16000
frame_length = 1024
hop_length = 512
window_length = 1024
window_function = "vorbis"

speech_sample = "Prototyping/Dataset Structure/Dataset/MS/NoisySpeech_training/noisy14_SNRdb_20.0_clnsp14.wav"

feature_extractor = FeatureExtraction(sampling_rate=sampling_rate,
                                      frame_length=frame_length, hop_length=hop_length,
                                      window_length=window_length, window_function=window_function)

speech = feature_extractor.load_audiofile(audio_file=speech_sample, title="Speech Audio", visualize=False)
speech = speech[:5*sampling_rate]

feature_extractor.fft(audio=speech, visualize=False)

vorbis_window = feature_extractor.vorbis_window(visualize=False)

speech_sample_stft = feature_extractor.stft(audio=speech, visualize=False)

speech_sample_stft = np.abs(speech_sample_stft)
pitches, magnitudes = librosa.piptrack(S=speech_sample_stft, sr=sampling_rate)

# speech_sample_mfcc = feature_extractor.get_mfccs(audio=speech, number_of_melbands=22,
#                                                  visualize=True)

speech_sample_mfcc = feature_extractor.get_mfccs_from_spectrogram(audio_stft=speech_sample_stft,
                                                                  number_of_melbands=22,
                                                                  visualize=True,
                                                                  save=False)

speech_sample_mfcc_delta, speech_sample_mfcc_delta2 = feature_extractor.get_mfccs_delta(mfccs=speech_sample_mfcc[:9],
                                                                                        number_of_melbands=9,
                                                                                        delta_delta=True,
                                                                                        visualize=True,
                                                                                        save=False)

# fundamental_freq, voiced_flags, voiced_prob = feature_extractor.get_fundamental_freq(audio=speech,
#                                                                                      freq_min=100, freq_max=400,
#                                                                                      visualize=True)

spectral_centroid = feature_extractor.get_spectral_centroid(audio_stft=speech_sample_stft,
                                                            visualize=False)

spectral_bandwidth = feature_extractor.get_spectral_bandwidth(audio_stft=speech_sample_stft,
                                                              visualize=False)

# feature_extractor.plot_spectral_centroid_bandwidth(audio=speech, save=False)
