from typing import Any, Callable, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from random import Random
from functools import reduce
from biosignals.dataset import MarkedData

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
    exclude_len: int

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
            if abs(onset - target) <= self.exclude_len:
                return True
        return False


# EEG_SAMPS_PER_MS = 1024 / 1000
# Since samps/ms is basically 1, we just use round numbers here
DEFAULT_WINDOW_CONFIG = WindowConfig(
    pre_len=500,
    post_len=250,
    max_jitter=50,
    exclude_len=500
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
    marked: MarkedData,
    conf: WindowConfig = DEFAULT_WINDOW_CONFIG,
    rand: Optional[Random] = None
) -> np.ndarray:
    assert conf.max_jitter >= 0
    rand_onsets: List[int]
    if conf.max_jitter == 0:
        rand_onsets = marked.onsets
    else:
        if rand is None:
            rand = Random()
        rand_onsets = [rand.randint(o - conf.max_jitter, o + conf.max_jitter) for o in marked.onsets]
    windows = []
    for o in rand_onsets:
        windows.append(marked.eeg[:, conf.start(o):conf.end(o)])
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
    marked: MarkedData,
    conf: WindowConfig = DEFAULT_WINDOW_CONFIG,
    rand: Optional[Random] = None
) -> np.ndarray:
    if rand is None:
        rand = Random()
    rand_onsets: List[int] = []
    num_samps = marked.eeg.shape[1]
    while len(rand_onsets) < len(marked.onsets):
        target = conf.random_onset(num_samps, rand)
        if not conf.is_near_positive(target, marked.onsets):
            rand_onsets.append(target)
    windows = []
    for o in rand_onsets:
        windows.append(marked.eeg[:, conf.start(o):conf.end(o)])
    return np.array(windows)


# Abstractly, something that extracts features from data
class Extractor:
    # Inputs:
    # eeg: np array of shape (num_windows, num_channels, num_samples)
    # Output:
    # DataFrame with columns 'window_index', 'channel_id' and features
    def extract(self, eeg: np.ndarray) -> pd.DataFrame:
        raise NotImplementedError()


class ArrayExtractor(Extractor):
    def __init__(self, name: str, fn: Callable[[np.ndarray], np.ndarray], dtype: Any):
        self._name = name
        self._fn = fn
        self._dtype = dtype

    def extract(self, eeg: np.ndarray) -> pd.DataFrame:
        values = self._fn(eeg)
        assert len(values.shape) >= 2
        assert values.shape[0] == eeg.shape[0]
        assert values.shape[1] == eeg.shape[1]
        d = {'window_index': [], 'channel_id': [], self._name: []}  # type: ignore
        for w in range(values.shape[0]):
            for c in range(values.shape[1]):
                d['window_index'].append(w)
                d['channel_id'].append(c)
                d[self._name].append(values[w, c])
        e = {
            'window_index': pd.Series(d['window_index'], dtype=np.uint),
            'channel_id': pd.Series(d['channel_id'], dtype=np.uint),
            self._name: pd.Series(d[self._name], dtype=self._dtype)
        }
        return pd.DataFrame.from_dict(e)


def band_power_extractor(name: str, freq_low: float, freq_high: float) -> Extractor:
    def fn(eeg: np.ndarray) -> np.ndarray:
        raise Exception('TODO')
    return ArrayExtractor(name, fn, np.float64)


# This comes from https://mne.tools/dev/auto_examples/time_frequency/time_frequency_global_field_power.html
FREQ_BANDS = [
    ('theta', 4, 7),
    ('alpha', 8, 12),
    ('beta', 13, 25),
    ('gamma', 30, 45)
]

# Feature extractors for band power
FREQ_EXTRACTORS = [band_power_extractor(name, lo, hi) for (name, lo, hi) in FREQ_BANDS]
# Feature extractors to use by default
DEFAULT_EXTRACTORS = FREQ_EXTRACTORS


# Extracts features from the given windows
# Inputs:
# eeg: numpy array with shape (num_windows, num_channels, num_window_samples)
# Output:
# dataframe with window_index, channel_id, and feature columns
def extract_features(eeg: np.ndarray, extractors: List[Extractor] = DEFAULT_EXTRACTORS) -> pd.DataFrame:
    assert len(eeg.shape) == 3
    feat_frames = [ex.extract(eeg) for ex in extractors]
    feats = reduce(
        lambda df1, df2: pd.merge(df1, df2, on=['window_index', 'channel_id'], how='inner'),
        feat_frames
    )
    return feats
