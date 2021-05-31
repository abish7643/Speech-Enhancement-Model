import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from audio_utilities import FeatureExtraction
from generate_dataset import add_noise_speech, get_melbands_gain


def iir_filter_design(band_frequency, sampling_rate=16000, order=1):
    # Filter Coefficients
    b, a = [], []

    f = band_frequency / (sampling_rate / 2)

    # Compute Coefficients
    for i in range(1, len(band_frequency) - 1):
        b_, a_ = signal.iirfilter(order,
                                  [f[i] - (f[i] - f[i - 1]) / 2, f[i] + (f[i + 1] - f[i]) / 2],
                                  btype="bandpass", output="ba")

        a.append(a_)
        b.append(b_)

    return b, a


def bandpass_filter(noisy_speech, b, a, hop_length, gains):
    # Define Filtered Output
    y = np.zeros(len(noisy_speech))

    # Define Delay
    state = np.zeros(len(b) - 1)

    # Initial Gain
    g = 0

    # Adjust Gain of a Particular Freq Band in the Whole Signal
    for n in range(0, len(gains)):
        g = max(0.6 * g, gains[n])
        b_ = b * g
        a_ = a
        filtered, state = signal.lfilter(b_, a_,
                                         noisy_speech[n * hop_length: min((n + 1) * hop_length, len(noisy_speech))],
                                         zi=state)
        y[n * hop_length: min((n + 1) * hop_length, len(noisy_speech))] = filtered

    return y


def plot_frequency_response(b, a=None, sampling_frequency=16000, title="", save=False):
    if len(a) != len(b):
        a = np.ones(len(b))

    plt.rcParams.update({'font.size': 26})
    plt.style.use(['dark_background'])
    plt.figure(figsize=(18, 5))

    for i in range(len(b)):
        # Compute Freq Response of Filter
        # w -> frequencies at which response (h) was computed
        w, h = signal.freqz(b[i], a[i])
        plt.plot(w * 0.15915494327 * sampling_frequency,
                 20 * np.log10(np.maximum(abs(h), 1e-5)))

    plt.title(title)
    plt.ylabel('Amplitude (dB)')
    plt.xlabel('Frequency (Hz)')
    if save:
        plt.savefig("".join(["../", title, ".png"]))
    plt.show()


def plot_wave(audio, sampling_rate=16000, evaluate_audio=None, title=None):
    plt.rcParams.update({'font.size': 26})
    plt.style.use(['dark_background'])
    plt.figure(figsize=(18, 5))

    librosa.display.waveplot(audio, sr=sampling_rate, color='c', alpha=1)

    if evaluate_audio is not None:
        librosa.display.waveplot(evaluate_audio, sr=sampling_rate, color='r', alpha=0.6)
    if title is None:
        plt.title("Wave Plot")
    else:
        plt.title(title)

    plt.xlabel("Time (second)")
    plt.ylabel("Amplitude")
    plt.show()


def equalize_noisy_signal(noisy_speech, gains, melbands=22, sampling_rate=16000, hop_length=512, visualize=False):
    # Compute Freq Bands
    freqbands = librosa.filters.mel_frequencies(n_mels=melbands, fmax=sampling_rate / 2,
                                                fmin=20)
    freqbands = freqbands[1:-1]

    # Get Filter Coefficients
    b, a = iir_filter_design(freqbands, sampling_rate=sampling_rate, order=1)

    if visualize:
        plot_frequency_response(b, a, save=True, title="Frequency Response")

    # Define Filtered Signal
    filtered_signal = np.zeros(len(noisy_speech))

    # Filter Each & Every Band
    for i in range(len(b)):
        filtered_signal += bandpass_filter(noisy_speech, b[i].copy(), a[i].copy(),
                                           hop_length=hop_length, gains=gains[:, i])

    return filtered_signal


def example(melbands=22, snr=25, speech_concat=2):
    # Example Audio Files
    noise_file = "Prototyping/Audio Samples/Kindergarten_Noise.wav"
    speech_file = "Prototyping/Audio Samples/Female_Speech.WAV"

    # Audio Parameters
    sampling_rate = 16000
    window_length, frame_length, hop_length = 1024, 1024, 512
    window_function = "vorbis"

    audio_utils = FeatureExtraction(sampling_rate=sampling_rate,
                                    frame_length=frame_length, hop_length=hop_length,
                                    window_length=window_length, window_function=window_function)

    # Load Audio Files
    noise = audio_utils.load_audiofile(noise_file)
    speech = audio_utils.load_audiofile(speech_file)

    # Concatenate Speech (If Reqd.)
    if speech_concat >= 1:
        for i in range(0, speech_concat):
            speech = np.concatenate((speech, speech))

    if len(noise) > len(speech):
        noise = noise[:len(speech)]
    else:
        speech = speech[:len(noise)]

    # Add Noise to Speech
    noisy_speech = add_noise_speech(speech=speech, noise=noise, snr=snr)

    # Compute Spectrum
    noisy_speech_stft = audio_utils.stft(noisy_speech)
    speech_stft = audio_utils.stft(speech)

    # Compute Gains
    gains = get_melbands_gain(clean_speech_stft=speech_stft, noisy_speech_stft=noisy_speech_stft)
    gains = np.clip(gains, 0, 1)
    gains = gains.T

    # Filter Signal
    filtered_signal = equalize_noisy_signal(noisy_speech=noisy_speech, gains=gains,
                                            melbands=melbands, sampling_rate=sampling_rate,
                                            hop_length=hop_length, visualize=True)

    # Plot Waveforms
    plot_wave(speech, sampling_rate=sampling_rate, title="Clean Speech")
    plot_wave(noisy_speech, sampling_rate=sampling_rate, title="Noisy Speech")
    plot_wave(filtered_signal, sampling_rate=sampling_rate, title="Filtered Speech")
    plot_wave(noisy_speech, sampling_rate=sampling_rate, title="Noisy-Filtered Speech",
              evaluate_audio=filtered_signal)


if __name__ == "__main__":
    example(melbands=22, snr=25)
