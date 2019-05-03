import numpy as np


def convert_to_samples(milliseconds: int, sampling_freq: int):
    """
    Convert a millisecond duration into the number of samples given the sampling frequency.

    :param milliseconds: duration to be converted to number of samples
    :param sampling_freq: the sampling frequency
    :return: number of samples
    """
    return int(milliseconds * (10 ** (-3)) * sampling_freq)



def my_windowing(v_signal: np.ndarray, sampling_rate: int, frame_length: int, frame_shift: int) -> [np.ndarray,
                                                                                                    np.ndarray]:
    """
    Splits a time domain singal into overlapping frames.

    :param v_signal: 1D signal to be windowed
    :param sampling_rate: the sampling rate in Hz
    :param frame_length: the length of a frame in milliseconds
    :param frame_shift: the frame shift in milliseconds
    :return: a matrix with the frames in its rows (shape [num_frames, num_samples_per_frame]) and a vector containing
    the time instances around which the frames are centered (sahpe [num_frames])
    """

    samples_per_frame = convert_to_samples(frame_length, sampling_rate)
    samples_per_shift = convert_to_samples(frame_shift, sampling_rate)

    m_frames = np.array([v_signal[start_idx:start_idx + samples_per_frame] for start_idx in
                         range(0, len(v_signal) - samples_per_frame + 1, samples_per_shift)])

    v_time_frames = np.array([(start_idx + samples_per_frame / 2) / sampling_rate for start_idx in
                              range(0, len(v_signal) - samples_per_frame + 1, samples_per_shift)])

    return m_frames, v_time_frames



def compute_istft(stft: np.ndarray, sampling_rate: int, frame_shift: int, synthesis_window: np.ndarray) -> [np.ndarray]:
    """
    Compute the inverse short-time Fourier transform.

    :param stft: STFT transformed signal
    :param sampling_rate: the sampling rate in Hz
    :param frame_shift: the frame shift used to compute the STFT in milliseconds
    :param synthesis_window: a numpy array containing a synthesis window function (length must match with time domain
    signal segments that were used to compute the STFT)
    :return: a numpy array containing the time domain signal
    """

    # compute inverse rFFT and apply synthesis window
    time_frames = np.fft.irfft(stft)
    num_frames, samples_per_frame = time_frames.shape
    assert samples_per_frame == len(synthesis_window), "Synthesis window must match the number of samples per frame."
    time_frames *= synthesis_window

    # compute output size
    samples_per_shift = convert_to_samples(frame_shift, sampling_rate)
    output_len = samples_per_frame + (num_frames - 1) * samples_per_shift
    time_signal = np.zeros((output_len))


    # reconstruct signal by adding overlapping windowed segments
    for i in range(num_frames):
        time_signal[i*samples_per_shift:i*samples_per_shift+samples_per_frame] += time_frames[i]

    return time_signal