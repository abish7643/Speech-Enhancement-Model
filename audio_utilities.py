import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import python_speech_features


class FeatureExtraction:
    def __init__(self, sampling_rate=22050, frame_length=1024, hop_length=512,
                 window_length=1024, window_length_t=0.032, window_function="hamming", debug=False):

        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.window_length = window_length
        self.window_length_t = window_length_t
        self.window_function = window_function
        self.figure_size = (18, 5)
        self.debug = debug
        plt.rcParams['font.size'] = '16'

        if self.window_function == "vorbis":
            self.window_function_array = self.vorbis_window()
        elif self.window_function == "hamming":
            self.window_function_array = self.hamming_window()
        else:
            raise Exception("No Such Window Function")

    def stft(self, audio, visualize=False, title="", save=False):

        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length
        window_length = self.window_length
        window_function_array = self.window_function_array

        # Find STFT
        audio_stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length,
                                  win_length=window_length, window=window_function_array)

        if visualize:
            # Find The Magnitude Spectrum
            audio_magnitude_spectrum = np.abs(audio_stft) ** 2

            # Find The Log Magnitude Spectrum
            audio_log_magnitude_spectrum = librosa.power_to_db(audio_magnitude_spectrum)

            plt.figure(figsize=self.figure_size)
            librosa.display.specshow(data=audio_log_magnitude_spectrum, sr=sampling_rate,
                                     hop_length=hop_length, x_axis="time", y_axis="log")
            plt.title("Log Magnitude Spectrum ({})".format(title))
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.colorbar(format="%+2.f")
            if save:
                plt.savefig("".join(["../", title, ".png"]))
            plt.show()

        return audio_stft

    def get_mfccs(self, audio, number_of_melbands=13, visualize=False):

        mfccs = librosa.feature.mfcc(y=audio, n_mfcc=number_of_melbands,
                                     sr=self.sampling_rate, hop_length=self.hop_length)

        # mfccs = python_speech_features.mfcc(signal=audio, samplerate=self.sampling_rate,
        #                                     numcep=number_of_melbands, nfilt=number_of_melbands,
        #                                     winlen=self.window_length_t, winstep=self.window_length_t,
        #                                     nfft=self.window_length)
        # mfccs = mfccs.transpose()

        if visualize:
            plt.figure(figsize=self.figure_size)
            librosa.display.specshow(data=mfccs, sr=self.sampling_rate,
                                     x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.f")
            plt.title("MFCCs ({})".format(number_of_melbands))
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.show()

        return mfccs

    def get_mfccs_from_spectrogram(self, audio_stft, number_of_melbands=13, visualize=False, save=False):

        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length
        window_function_array = self.window_function_array

        # Calculate Magnitude Spectrum
        audio_magnitude_spectrum = np.abs(audio_stft) ** 2

        # Convert to Mel Scale -> Mel Spectrogram
        audio_mel_spectrogram = librosa.feature.melspectrogram(S=audio_magnitude_spectrum,
                                                               sr=sampling_rate, n_fft=frame_length,
                                                               hop_length=hop_length, n_mels=number_of_melbands,
                                                               window=window_function_array)

        # Convert to Log Mel Spectrogram
        audio_log_mel_spectrogram = librosa.power_to_db(audio_mel_spectrogram)

        # Find Cepstral Coefficients
        mfccs = librosa.feature.mfcc(S=audio_log_mel_spectrogram,
                                     sr=sampling_rate, n_mfcc=number_of_melbands)

        if visualize:
            plt.rc('font', size=20)
            plt.figure(figsize=self.figure_size)
            librosa.display.specshow(data=mfccs, sr=sampling_rate, hop_length=hop_length,
                                     x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.f")
            plt.title("MFCCs ({})".format(number_of_melbands))
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            if save:
                plt.savefig("".join(["../", "MFCCs ({})".format(number_of_melbands), ".png"]), bbox_inches='tight')
            plt.show()

        return mfccs

    def get_melspectrogram(self, audio_stft, number_of_melbands=13, visualize=False):

        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length
        window_function_array = self.window_function_array

        # Calculate Magnitude Spectrum
        audio_magnitude_spectrum = np.abs(audio_stft) ** 2

        # Convert to Mel Scale -> Mel Spectrogram
        audio_mel_spectrogram = librosa.feature.melspectrogram(S=audio_magnitude_spectrum,
                                                               sr=sampling_rate, n_fft=frame_length,
                                                               hop_length=hop_length, n_mels=number_of_melbands,
                                                               window=window_function_array)

        # Convert to Log Mel Spectrogram
        # audio_log_mel_spectrogram = librosa.power_to_db(audio_mel_spectrogram)

        if visualize:
            plt.figure(figsize=self.figure_size)
            librosa.display.specshow(data=audio_mel_spectrogram, sr=sampling_rate,
                                     x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.f")
            plt.title("Log Mel Spectrogram")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.show()

        return audio_mel_spectrogram

    def get_mfccs_delta(self, mfccs, delta_delta=True, number_of_melbands=13, visualize=False, save=False):

        # Compute Derivative of MFCC Per Frame
        delta_mfcc = librosa.feature.delta(mfccs)[:number_of_melbands]

        if delta_delta:
            # Compute Double Derivative of MFCC Per Frame
            delta2_mfcc = librosa.feature.delta(mfccs, order=2)[:number_of_melbands]
        else:
            delta2_mfcc = False

        if visualize:
            plt.rc('font', size=20)
            plt.figure(figsize=self.figure_size)
            librosa.display.specshow(data=delta_mfcc, sr=self.sampling_rate,
                                     x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.f")
            plt.title("Delta MFCC ({})".format(number_of_melbands))
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            if save:
                plt.savefig("".join(["../", "Delta MFCC ({})".format(number_of_melbands), ".png"]), bbox_inches='tight')
            plt.show()

            if delta_delta:
                plt.rc('font', size=20)
                plt.figure(figsize=self.figure_size)
                librosa.display.specshow(data=delta2_mfcc, sr=self.sampling_rate,
                                         x_axis="time", y_axis="mel")
                plt.colorbar(format="%+2.f")
                plt.title("Delta Delta MFCC ({})".format(number_of_melbands))
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz)")
                if save:
                    plt.savefig("".join(["../", "Delta Delta MFCC ({})".format(number_of_melbands), ".png"]),
                                bbox_inches='tight')
                plt.show()

        return delta_mfcc, delta2_mfcc

    def get_fundamental_freq(self, audio, freq_min=100, freq_max=500, visualize=False):

        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length

        # audio, _ = librosa.load(audio, sampling_rate)

        # Find Fundamental Frequencies, Voiced Flags (True or False), Voice Probability
        fundamental_frequencies, voiced_flags, voiced_prob = librosa.pyin(y=audio, sr=sampling_rate,
                                                                          frame_length=frame_length,
                                                                          hop_length=hop_length,
                                                                          fmin=freq_min, fmax=freq_max)
        if visualize:
            # Create Time Axis
            # time_axis = librosa.times_like(fundamental_frequencies, sr=sampling_rate,
            #                                n_fft=frame_length, hop_length=hop_length)

            audio_frames = range(len(fundamental_frequencies))
            time_axis = librosa.frames_to_time(audio_frames, sr=sampling_rate, hop_length=hop_length)

            # Find Log Magnitude Spectrum
            audio_stft = librosa.stft(y=audio, n_fft=frame_length, hop_length=hop_length)
            audio_magnitude_spectrum = np.abs(audio_stft) ** 2
            audio_log_magnitude_spectrum = librosa.power_to_db(audio_magnitude_spectrum)

            # Plot
            plt.figure(figsize=self.figure_size)

            # Show Spectrum
            librosa.display.specshow(data=audio_log_magnitude_spectrum, sr=sampling_rate,
                                     hop_length=hop_length, x_axis="time", y_axis="log")
            # Plot Fundamental Frequencies
            plt.plot(time_axis, fundamental_frequencies, color='cyan', linewidth=3)
            plt.plot(time_axis, voiced_flags * 150, color='r', linewidth=3)
            plt.plot(time_axis, voiced_prob * 100, color='white', linewidth=3)
            plt.colorbar(format="%+2.f")
            plt.title("Fundamental Frequency")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")

            # plt.plot(time_axis, voiced_flags * 150, color='r', linewidth=3)
            # plt.plot(time_axis, voiced_prob * 100, color='b', linewidth=3)

            plt.show()

        return fundamental_frequencies, voiced_flags, voiced_prob

    def get_spectral_centroid(self, audio=None, audio_stft=None, visualize=False):

        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length
        window_function_array = self.window_function_array

        if audio_stft is not None:
            # Calculate Magnitude Spectrum
            audio_magnitude_spectrum, _ = librosa.magphase(audio_stft)

            # Calculate Spectral Centroid from Magnitude Spectrum
            audio_spectral_centroid = librosa.feature.spectral_centroid(S=audio_magnitude_spectrum,
                                                                        sr=sampling_rate,
                                                                        window=window_function_array,
                                                                        n_fft=frame_length, hop_length=hop_length)
        else:
            # Calculate Spectral Centroid From Audio File
            audio_spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sampling_rate,
                                                                        window=window_function_array,
                                                                        n_fft=frame_length, hop_length=hop_length)
        if visualize:
            plt.figure(figsize=self.figure_size)

            # frames = range(len(audio_spectral_centroid[0]))
            # time_axis = librosa.time_to_frames(frames, sr=sampling_rate,
            #                                    n_fft=frame_length, hop_length=hop_length)

            time_axis = librosa.times_like(audio_spectral_centroid, sr=sampling_rate,
                                           n_fft=frame_length, hop_length=hop_length)

            plt.plot(time_axis, audio_spectral_centroid[0])
            plt.title("Spectral Centroid")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.show()

        return audio_spectral_centroid

    def get_spectral_bandwidth(self, audio=None, audio_stft=None, visualize=False):

        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length
        window_function_array = self.window_function_array

        if audio_stft is not None:
            # Calculate Magnitude Spectrum
            audio_magnitude_spectrum, _ = librosa.magphase(audio_stft)

            # Calculate Spectral Bandwidth from Magnitude Spectrum
            audio_spectral_bandwidth = librosa.feature.spectral_bandwidth(S=audio_magnitude_spectrum,
                                                                          sr=sampling_rate,
                                                                          window=window_function_array,
                                                                          hop_length=hop_length,
                                                                          n_fft=frame_length)
        else:
            # Calculate Spectral Bandwidth From Audio File
            audio_spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate,
                                                                          window=window_function_array,
                                                                          hop_length=hop_length,
                                                                          n_fft=frame_length)
        if visualize:
            plt.figure(figsize=self.figure_size)

            # frames = range(len(audio_spectral_bandwidth[0]))
            # time_axis = librosa.time_to_frames(frames, sr=sampling_rate,
            #                                    n_fft=frame_length, hop_length=hop_length)

            time_axis = librosa.times_like(audio_spectral_bandwidth, sr=sampling_rate,
                                           n_fft=frame_length, hop_length=hop_length)

            plt.plot(time_axis, audio_spectral_bandwidth[0])
            plt.title("Spectral Bandwidth")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.show()

        return audio_spectral_bandwidth

    def plot_spectral_centroid_bandwidth(self, audio=None, audio_stft=None, save=False):

        sampling_rate = self.sampling_rate
        frame_length = self.frame_length
        hop_length = self.hop_length
        window_length = self.window_length
        window_function_array = self.window_function_array

        if audio is not None:
            audio_stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length,
                                      win_length=window_length, window=window_function_array)

            audio_spectral_centroid = self.get_spectral_centroid(audio_stft=audio_stft)
            audio_spectral_bandwidth = self.get_spectral_bandwidth(audio_stft=audio_stft)

        else:
            audio_spectral_centroid = self.get_spectral_centroid(audio_stft=audio_stft)
            audio_spectral_bandwidth = self.get_spectral_bandwidth(audio_stft=audio_stft)

        # Find The Magnitude Spectrum
        audio_magnitude_spectrum = np.abs(audio_stft) ** 2

        # Find The Log Magnitude Spectrum
        audio_log_magnitude_spectrum = librosa.power_to_db(audio_magnitude_spectrum)

        plt.rc('font', size=20)
        plt.figure(figsize=self.figure_size)

        librosa.display.specshow(data=audio_log_magnitude_spectrum, sr=sampling_rate,
                                 hop_length=hop_length, x_axis="time", y_axis="log")

        plt.title("Spectral Centroid and Bandwidth")
        plt.colorbar(format="%+2.f")

        time_axis = librosa.times_like(audio_spectral_centroid, sr=sampling_rate,
                                       n_fft=frame_length, hop_length=hop_length)

        plt.plot(time_axis, audio_spectral_centroid[0],
                 color='black', linewidth=3, label="Centroid")
        plt.plot(time_axis, audio_spectral_bandwidth[0],
                 color='red', linewidth=3, label="Bandwidth")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.legend(loc='lower right')
        if save:
            plt.savefig("".join(["../", "Spectral Centroid and Bandwidth", ".png"]), bbox_inches='tight')
        plt.show()

        return True

    def vorbis_window(self, visualize=False):
        """
        The Vorbis Window is defined as,
        W(n) = sin[pi/2*sin(pi*n/N)]

        :param visualize: Pass True if function has to be plotted
        :return: numpy array of set window length
        """

        window_length = self.window_length
        n = np.linspace(0, window_length - 1, window_length)
        pi = np.pi
        vorbis_window = np.sin(pi / 2 * (np.sin(pi * (n / window_length))) ** 2)

        if visualize:
            plt.figure(figsize=self.figure_size)
            plt.plot(vorbis_window)
            plt.title("Vorbis Window")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.show()

        return vorbis_window

    def hamming_window(self, visualize=False):
        """
        The Hamming Window is defined as,
        W(n) = 0.5(1-cos(2*pi*n/N))

        :param visualize: Pass True if function has to be plotted
        :return: numpy array of set window length
        """

        window_length = self.window_length
        # n = np.linspace(0, window_length - 1, window_length)
        # pi = np.pi
        # hamming_window = np.sin(pi / 2 * (np.sin(pi * (n / window_length))) ** 2)
        hamming_window = np.hamming(window_length)

        if visualize:
            plt.figure(figsize=self.figure_size)
            plt.plot(hamming_window)
            plt.title("Hamming Window")
            plt.show()

        return hamming_window

    def load_audiofile(self, audio_file, title="Wave Plot", visualize=False):

        audio, _ = librosa.load(audio_file, sr=self.sampling_rate)

        if visualize:
            plt.figure(figsize=self.figure_size)
            plt.plot(audio)
            plt.title(title)
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.show()

        return audio

    def fft(self, audio, visualize=False):

        # audio = self.load_audiofile(audio_file, visualize=False)

        audio_spectrum = np.fft.fft(audio)

        if visualize:
            audio_magnitude_spectrum = np.abs(audio_spectrum)

            plt.figure(figsize=self.figure_size)

            # Considering Frequencies Upto Nyquist Frequencies
            frequency_bins = int(len(audio_magnitude_spectrum) * 0.5)
            frequency = np.linspace(0, self.sampling_rate, len(audio_magnitude_spectrum))

            plt.plot(frequency[:frequency_bins], audio_magnitude_spectrum[:frequency_bins])
            plt.xlabel("Freq (Hz)")
            plt.title("Magnitude Spectrum")

            plt.show()

        return audio_spectrum

    def get_chroma_cqt(self, audio, visualize=False):

        chrome_cqt = librosa.feature.chroma_cqt(y=audio, hop_length=self.hop_length,
                                                sr=self.sampling_rate)

        if visualize:
            plt.figure(figsize=self.figure_size)
            librosa.display.specshow(data=chrome_cqt, sr=self.sampling_rate,
                                     x_axis="time", y_axis="mel")
            plt.colorbar(format="%+2.f")
            plt.title("Chroma CQT")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.show()

        return chrome_cqt
