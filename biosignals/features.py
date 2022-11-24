from typing import List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from random import Random

# Window and feature extraction


@dataclass(frozen=True)
class WindowConfig:
    # The length of the part of the window before the onset (in samples)
    pre_len: int
    # The length of the part of the window after the onset (in samples)
    post_len: int
    # Max jitter (in samples)
    max_jitter: int

    # Minimum distance (in samples) between "unrelated" onsets
    def exclude_len(self) -> int:
        return max(self.pre_len, self.post_len)

    # Total length of the window (in samples)
    def total_len(self) -> int:
        return self.pre_len + self.post_len

    # Start of the window (sample)
    def start(self, onset: int) -> int:
        return onset - self.pre_len

    # End of the window (sample)
    def end(self, onset: int) -> int:
        return onset + self.post_len

    # Return a random onset that has a valid window (not near ends of array)
    def random_onset(self, num_samps: int, rand: Random) -> int:
        return rand.randint(self.pre_len, num_samps - self.post_len)

    # Return true if the target is near any of the given onsets
    def is_near_positive(self, target: int, onsets: List[int]) -> bool:
        for onset in onsets:
            # Check for distance to positive example onsets
            if abs(onset - target) <= self.exclude_len():
                return True
        return False


# EEG_SAMPS_PER_MS = 1024 / 1000
# Since samps/ms is basically 1, we just use round numbers here
DEFAULT_WINDOW_CONFIG = WindowConfig(
    pre_len=500,
    post_len=250,
    max_jitter=50
)


# Extract all positive windows (marked by onsets) from eeg data
# Inputs:
# eeg: numpy array with shape (num_channels, num_trial_samples)
# onsets: list of onsets
# conf: window config
# rand: source of randomness
# Output:
# numpy array with shape (num_windows, num_channels, num_window_samples)
def positive_windows(
    eeg: np.ndarray,
    onsets: List[int],
    conf: WindowConfig = DEFAULT_WINDOW_CONFIG,
    rand: Optional[Random] = None
) -> np.ndarray:
    assert conf.max_jitter >= 0
    rand_onsets: List[int]
    if conf.max_jitter == 0:
        rand_onsets = onsets
    else:
        if rand is None:
            rand = Random()
        rand_onsets = [rand.randint(o - conf.max_jitter, o + conf.max_jitter) for o in onsets]
    windows = []
    for o in rand_onsets:
        windows.append(eeg[:, conf.start(o):conf.end(o)])
    return np.array(windows)


# Extract as many negative windows from eeg data as there are positive windows.
# These windows will be randomly chosen such that their onsets are a given
# distance from any positive onset.
# Inputs:
# eeg: numpy array with shape (num_channels, num_trial_samples)
# onsets: list of onsets
# conf: window config
# rand: source of randomness
# Output:
# numpy array with shape (num_windows, num_channels, num_window_samples)
def negative_windows(
    eeg: np.ndarray,
    onsets: List[int],
    conf: WindowConfig = DEFAULT_WINDOW_CONFIG,
    rand: Optional[Random] = None
) -> np.ndarray:
    if rand is None:
        rand = Random()
    rand_onsets: List[int] = []
    num_samps = eeg.shape[1]
    while len(rand_onsets) < len(onsets):
        target = conf.random_onset(num_samps, rand)
        if not conf.is_near_positive(target, onsets):
            rand_onsets.append(target)
    windows = []
    for o in rand_onsets:
        windows.append(eeg[:, conf.start(o):conf.end(o)])
    return np.array(windows)


# Extracts features from the given windows
# Inputs:
# eeg: numpy array with shape (num_windows, num_channels, num_window_samples)
# Output:
# dataframe with window_id, channel_id, and feature columns
def extract_features(eeg: np.ndarray) -> pd.DataFrame:
    assert len(eeg.shape) == 3
    fnames = ['window_id', 'channel_id']
    features = {n: list() for n in fnames}  # type: ignore
    for window_id in range(eeg.shape[0]):
        for channel_id in range(eeg.shape[1]):
            samps = eeg[window_id, channel_id, :]
            assert samps is not None
            # TODO actually add to features...
    return pd.DataFrame.from_dict(features)
