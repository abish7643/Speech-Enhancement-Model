from pystoi import stoi
from pesq import pesq


def get_stoi_pesq(filtered_speech, original_speech, sampling_rate=16000):

    # Compute STOI
    stoi_value = stoi(x=original_speech, y=filtered_speech, fs_sig=sampling_rate, extended=False)

    # Compute PESQ
    pesq_value = pesq(ref=original_speech, deg=filtered_speech, fs=sampling_rate, mode='wb')

    return stoi_value, pesq_value
