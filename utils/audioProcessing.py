import librosa
from librosa.effects import trim
import numpy as np

#TODO(CONG SHENG) Add get_MFCC into library
def get_MFCC(x, sampling_rate, hop_length, n):
    mfcc = librosa.feature.mfcc(x, sr= sampling_rate, hop_length = hop_length, n_mfcc = n)
    return mfcc

def remove_silence(signal,top_db = 10):
    """This function removes trailing and leading silence periods of audio 
    signals.

    Uses audio amplitude to determine silence segments in audio signal and 
    removes it. Uses librosa's "trim" function.
    
    Args:
        signal: Audio signal in an array form which can be converted to a 
                np.ndarray.
        sr: Sample rate of the audio signal  
        top_db: The threshold (in decibels) below the reference to consider as
                silence. (See https://librosa.org/doc/main/generated/
                librosa.effects.trim.html)

    Returns:
        yt: Audio signal with silence segments removed.
    """
    y = np.array(signal, dtype = np.float64)
    yt, _ = trim(y, top_db= top_db)
    return yt

def batch_pad_audio(audio_input, audio_length, sampling_rate, direction="both"):
    """Pad the audio signal in batch.

    Padding is done with constant mode where zeros are padded to fill the audio 
    to the audio length specified.

    Args:
        audio_input: Input audio signal.
        audio length: Maximum duration of audio in seconds.
        sampling_rate: Sampling rate of the audio signal.

    Returns:
        padded_audio_batch: Zero-padded audio signal.
    """
    padded_audio_batch = []
    max_length = int(sampling_rate * audio_length)
    for single_audio in audio_input:
        if len(single_audio) < max_length:
            if direction not in ["start", "end", "both"]:
                print("Invalid padding mode")
                return
            elif direction == "start":
                padded_audio_batch.append(pad_audio_single(single_audio, 
                                    max_length, "start"))
            elif direction == "end":
                padded_audio_batch.append(pad_audio_single(single_audio, 
                                    max_length, "end"))
            else:
                padded_audio_batch.append(pad_audio_both(single_audio, 
                                    max_length=max_length))
        else:
            padded_audio_batch.append(single_audio[:max_length])
    return padded_audio_batch

def batch_mfcc(audio_batch, sampling_rate, hop_length, n):
    """Batch process to extract mfcc.

    MFCC is extracted from audio batch with sampling rate, hop length and 
    n_mfcc specified.

    Args:
        audio_batch: Input audio batch.
        sampling_rate: Sampling rate of the audio batch.
        hop_length: Number of samples between successive frames.
        n: Number of MFCCs to return.
        (See https://librosa.org/doc/latest/index.html)

    Returns:
        mfcc_batch: Batch of MFCC extracted.
    """
    mfcc_batch=[librosa.feature.mfcc(x, sr=sampling_rate, 
                hop_length=hop_length, n_mfcc=n) for x in audio_batch]
    return mfcc_batch

############################## Helper functions ##############################

def pad_audio_both(audio_input, max_length):
    """Pad audio signal on both side (start and end).

    Padding is done with constant mode where zeros are padded to fill the audio 
    to the max_length specified.

    Args:
        audio_input: Input audio signal.
        max_length: Maximum length of audio signal in terms of samples.

    Returns:
        padded_audio: Zero-padded audio signal.
    """
    audio_length = len(audio_input)
    pwidth_l = (max_length - audio_length) // 2
    pwidth_r = max_length - pwidth_l - audio_length
    padded_audio = np.pad(audio_input, (pwidth_l, pwidth_r), mode='constant')
    return padded_audio

def pad_audio_single(audio_input, max_length, side="end"):
    """Pad audio signal on one side (start or end).

    Padding is done with constant mode where zeros are padded to fill the audio 
    to the max_length specified.

    Args:
        audio_input: Input audio signal.
        max_length: Maximum length of audio signal in terms of samples.
        side: Default as end to pad the end. Specify "start" to pad the start.

    Returns:
        padded_audio: Zero-padded audio signal.
    """
    audio_length = len(audio_input)
    pwidth = max_length - audio_length
    padding = (0, pwidth)
    if side=="start":
        padding = (pwidth, 0)
    padded_audio = np.pad(audio_input, padding, mode='constant')
    return padded_audio

def pad_2D_square(audio_input, max_length):
    """Pad audio signal on both side (start and end).

    Padding is done with constant mode where zeros are padded to fill the audio 
    to the max_length specified.

    Args:
        audio_input: Input audio signal.
        max_length: Maximum length of audio signal in terms of samples.

    Returns:
        padded_audio: Zero-padded audio signal.
    """
    audio_height = audio_input.shape[0]
    audio_width = audio_input.shape[1]
    phelght_top = (max_length - audio_height) // 2
    pheight_btm = max_length - phelght_top - audio_height
    pwidth_l = (max_length - audio_width) // 2
    pwidth_r = max_length - pwidth_l - audio_width
    padded_audio = np.pad(audio_input, ((phelght_top, pheight_btm), (pwidth_l, pwidth_r)), mode='constant')
    return padded_audio

def normalize_audio(audio_signal, scale_factor=0.5):
    """Double-ended normalization of audio based on scale factor.

    Args:
        audio_signal: Audio signal in array.
        scale_factor: Default set to 0.5 such that the range will be kept to 
                      -0.5 to 0.5.

    Returns:
        noramalized audio signal
    """
    max_amp = max(abs(audio_signal))
    return scale_factor * audio_signal/max_amp
