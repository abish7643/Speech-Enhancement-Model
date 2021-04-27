from audio_utilities import FeatureExtraction
import warnings

warnings.filterwarnings("ignore")

sampling_rate = 16000
frame_length = 512
hop_length = 256
window_length = 512
window_function = "vorbis"

speech_sample = "Prototyping/Audio Samples/Female_Speech.WAV"

feature_extractor = FeatureExtraction(sampling_rate=sampling_rate,
                                      frame_length=frame_length, hop_length=hop_length,
                                      window_length=window_length, window_function=window_function)

speech = feature_extractor.load_audiofile(audio_file=speech_sample, title="Speech Audio", visualize=False)

feature_extractor.fft(audio=speech, visualize=True)

vorbis_window = feature_extractor.vorbis_window(visualize=True)

speech_sample_stft = feature_extractor.stft(audio=speech, visualize=True)

# speech_sample_mfcc = feature_extractor.get_mfccs(audio=speech, visualize=True)

speech_sample_mfcc = feature_extractor.get_mfccs_from_spectrogram(audio_stft=speech_sample_stft,
                                                                   number_of_melbands=13, visualize=True)

speech_sample_mfcc_delta, speech_sample_mfcc_delta2 = feature_extractor.get_mfccs_delta(mfccs=speech_sample_mfcc,
                                                                                        delta_delta=True,
                                                                                        visualize=True)

fundamental_freq, voiced_flags, voiced_prob = feature_extractor.get_fundamental_freq(audio=speech,
                                                                                     freq_min=100, freq_max=400,
                                                                                     visualize=True)

spectral_centroid = feature_extractor.get_spectral_centroid(audio_stft=speech_sample_stft,
                                                            visualize=True)

spectral_bandwidth = feature_extractor.get_spectral_bandwidth(audio_stft=speech_sample_stft,
                                                              visualize=True)
