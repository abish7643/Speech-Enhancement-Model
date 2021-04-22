import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


class FeatureExtraction:
    def __init__(self, sampling_rate=22050, frame_length=1024, hop_length=512):
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.figure_size = (18, 5)

    def stft(self, audio_file_name, visualize=False):
        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length

        # Load Audio File
        audio_file, _ = librosa.load(audio_file_name, sampling_rate)

        # Find STFT
        audio_file_stft = librosa.stft(audio_file, n_fft=frame_length, hop_length=hop_length)

        if visualize:
            # Find The Magnitude Spectrum
            audio_file_magnitude_spectrum = np.abs(audio_file_stft) ** 2

            # Find The Log Magnitude Spectrum
            audio_file_log_magnitude_spectrum = librosa.power_to_db(audio_file_magnitude_spectrum)

            plt.figure(figsize=self.figure_size)
            librosa.display.specshow(data=audio_file_log_magnitude_spectrum, sr=sampling_rate,
                                     hop_length=hop_length, x_axis="time", y_axis="log")
            plt.title("Log Magnitude Spectrum")
            plt.colorbar(format="%+2.f")
            plt.show()

        return audio_file_stft

    def get_mfccs(self, audio_file, number_of_melbands=13, visualize=False):

        # Load File
        file, _ = librosa.load(audio_file, self.sampling_rate)

        # Find MFCCs Directly From File
        mfccs = librosa.feature.mfcc(y=file, n_mfcc=number_of_melbands, sr=self.sampling_rate)

        if visualize:
            plt.figure(figsize=self.figure_size)
            librosa.display.specshow(data=mfccs, sr=self.sampling_rate, x_axis="time")
            plt.colorbar(format="%+2.f")
            plt.title("MFCCs {}".format(number_of_melbands))
            plt.show()

        return mfccs

    def get_mfccs_from_spectrogram(self, audio_file_stft, number_of_melbands=13, visualize=False):

        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length

        # Calculate Magnitude Spectrum
        audio_file_magnitude_spectrum = np.abs(audio_file_stft) ** 2

        # Convert to Mel Scale -> Mel Spectrogram
        audio_file_mel_spectrogram = librosa.feature.melspectrogram(S=audio_file_magnitude_spectrum,
                                                                    sr=sampling_rate, n_fft=frame_length,
                                                                    hop_length=hop_length, n_mels=number_of_melbands)

        # Convert to Log Mel Spectrogram
        audio_file_log_mel_spectrogram = librosa.power_to_db(audio_file_mel_spectrogram)

        # Find Cepstral Coefficients
        mfccs = librosa.feature.mfcc(S=audio_file_log_mel_spectrogram,
                                     sr=sampling_rate, n_mfcc=number_of_melbands)

        if visualize:
            plt.figure(figsize=self.figure_size)
            librosa.display.specshow(data=audio_file_log_mel_spectrogram, sr=sampling_rate,
                                     x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.f")
            plt.title("Log Mel Spectrogram")
            plt.show()

            plt.figure(figsize=self.figure_size)
            librosa.display.specshow(data=mfccs, sr=sampling_rate, x_axis="time")
            plt.colorbar(format="%+2.f")
            plt.title("MFCCs {}".format(number_of_melbands))
            plt.show()

        return mfccs

    @staticmethod
    def get_mfccs_delta(mfccs, delta_delta=True):

        # Compute Derivative of MFCC Per Frame
        delta_mfcc = librosa.feature.delta(mfccs)

        if delta_delta:
            # Compute Double Derivative of MFCC Per Frame
            delta2_mfcc = librosa.feature.delta(mfccs, order=2)
        else:
            delta2_mfcc = False

        return delta_mfcc, delta2_mfcc

    def get_fundamental_freq(self, audio_file, freq_min=100, freq_max=500, visualize=False):

        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length

        # Find Fundamental Frequencies, Voiced Flags (True or False), Voice Probability
        fundamental_frequencies, voiced_flags, voiced_prob = librosa.pyin(y=audio_file, sr=sampling_rate,
                                                                          frame_length=frame_length,
                                                                          hop_length=hop_length,
                                                                          fmin=freq_min, fmax=freq_max)
        if visualize:
            # Create Time Axis
            # time_axis = librosa.times_like(fundamental_frequencies, sr=sampling_rate,
            #                                n_fft=frame_length, hop_length=hop_length)

            female_speech_frames = range(len(fundamental_frequencies))
            time_axis = librosa.frames_to_time(female_speech_frames, hop_length=hop_length)

            # Find Log Magnitude Spectrum
            audio_file_stft = librosa.stft(y=audio_file, n_fft=frame_length, hop_length=hop_length)
            audio_file_magnitude_spectrum = np.abs(audio_file_stft) ** 2
            audio_file_log_magnitude_spectrum = librosa.power_to_db(audio_file_magnitude_spectrum)

            # Plot
            plt.figure(figsize=self.figure_size)
            # Show Spectrum
            librosa.display.specshow(data=audio_file_log_magnitude_spectrum, sr=sampling_rate,
                                     hop_length=hop_length, x_axis="time", y_axis="log")
            # Plot Fundamental Frequencies
            plt.plot(time_axis, fundamental_frequencies, color='cyan', linewidth=3)
            plt.colorbar(format="%+2.f")
            plt.show()

        return fundamental_frequencies, voiced_flags, voiced_prob

    def get_spectral_centroid(self, audio_file, from_spectrogram=False, visualize=False):

        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length

        if from_spectrogram:
            # Calculate STFT
            audio_file_stft = librosa.stft(audio_file, n_fft=frame_length, hop_length=hop_length)

            # Calculate Magnitude Spectrum
            audio_file_magnitude_spectrum, _ = librosa.magphase(audio_file_stft)

            # Calculate Spectral Centroid from Magnitude Spectrum
            audio_file_spectral_centroid = librosa.feature.spectral_centroid(S=audio_file_magnitude_spectrum,
                                                                             sr=sampling_rate,
                                                                             n_fft=frame_length, hop_length=hop_length)
        else:
            # Calculate Spectral Centroid From Audio File
            audio_file_spectral_centroid = librosa.feature.spectral_centroid(y=audio_file, sr=sampling_rate,
                                                                             n_fft=frame_length, hop_length=hop_length)
        if visualize:
            plt.figure(figsize=self.figure_size)

            frames = range(len(audio_file_spectral_centroid[0]))
            time_axis = librosa.time_to_frames(frames, sr=sampling_rate,
                                          n_fft=frame_length, hop_length=hop_length)

            plt.plot(time_axis, audio_file_spectral_centroid[0])
            plt.show()

        return audio_file_spectral_centroid[0]

    def get_spectral_bandwidth(self, audio_file, from_spectrogram=False, visualize=False):

        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length

        if from_spectrogram:
            # Calculate STFT
            audio_file_stft = librosa.stft(audio_file, n_fft=frame_length, hop_length=hop_length)

            # Calculate Magnitude Spectrum
            audio_file_magnitude_spectrum, _ = librosa.magphase(audio_file_stft)

            # Calculate Spectral Bandwidth from Magnitude Spectrum
            audio_file_spectral_bandwidth = librosa.feature.spectral_bandwidth(S=audio_file_magnitude_spectrum,
                                                                               sr=sampling_rate,
                                                                               hop_length=hop_length,
                                                                               n_fft=frame_length)
        else:
            # Calculate Spectral Bandwidth From Audio File
            audio_file_spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_file, sr=sampling_rate,
                                                                               hop_length=hop_length,
                                                                               n_fft=frame_length)
        if visualize:
            plt.figure(figsize=self.figure_size)

            frames = range(len(audio_file_spectral_bandwidth[0]))
            time_axis = librosa.time_to_frames(frames, sr=sampling_rate,
                                          n_fft=frame_length, hop_length=hop_length)

            plt.plot(time_axis, audio_file_spectral_bandwidth[0])
            plt.show()

        return audio_file_spectral_bandwidth[0]
