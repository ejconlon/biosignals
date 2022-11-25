from enum import Enum
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from random import Random
from biosignals.dataset import MarkedData

# Windowing + dataset splitting


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
        start = conf.start(o)
        end = conf.end(o)
        if start >= 0 and end < marked.eeg.shape[1]:
            windows.append(marked.eeg[:, conf.start(o):conf.end(o)])
    return np.array(windows, dtype=marked.eeg.dtype)


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
    out = np.array(windows, dtype=marked.eeg.dtype)
    shape = (len(marked.onsets), marked.eeg.shape[1], conf.total_len())
    assert out.shape == shape
    return out


# Project a windowed array into a dataframe
# Inputs:
# data: numpy array with shape (num_windows, num_channels, num_window_samples)
# Output:
# dataframe with columns 'window_index', 'channel_id', 'eeg'
def project_df(part: str, data: np.ndarray) -> pd.DataFrame:
    assert len(data.shape) == 3
    d = {'part': [], 'window_index': [], 'channel_id': [], 'eeg': []}  # type: ignore
    for w in range(data.shape[0]):
        for c in range(data.shape[1]):
            d['part'].append(part)
            d['window_index'].append(w)
            d['channel_id'].append(c)
            d['eeg'].append(data[w, c])
    e = {
        'part': pd.Series(d['part'], dtype=str),
        'window_index': pd.Series(d['window_index'], dtype=np.uint),
        'channel_id': pd.Series(d['channel_id'], dtype=np.uint),
        'eeg': pd.Series(d['eeg'], dtype=data.dtype)
    }
    return pd.DataFrame.from_dict(e)


class Role(Enum):
    TRAIN = 0
    VALIDATE = 1
    TEST = 2


class Splitter:
    # Inputs:
    # role: train/validate/test
    # rand: optional random src
    # Output:
    # df with 'part', 'window_id', 'label', 'channel_id', ...
    def split(
        self,
        role: Role,
        conf: WindowConfig = DEFAULT_WINDOW_CONFIG,
        rand: Optional[Random] = None
    ) -> pd.DataFrame:
        raise NotImplementedError()


# Generate a random permutation of range(100) to select fairly
def generate_perm(rand: Optional[Random] = None) -> List[int]:
    if rand is None:
        rand = Random()
    xs = list(range(100))
    rand.shuffle(xs)
    return xs


# Splits data randomly by role according to the percentage breakdown and permutation
class RandomSplitter:
    def __init__(self, marked: Dict[str, MarkedData], pct: Dict[Role, int], perm: List[int]):
        self._marked = marked
        self._pct = pct
        total_pct = sum(pct.values())
        assert total_pct == 100
        self._last_window_id = 0
        self._perm = perm
        assert sorted(perm) == list(range(100))

    def _concat(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        assert len(dfs) > 0
        mod_dfs = []
        for df in dfs:
            max_index = df['window_index'].max()
            df['window_id'] = pd.Series(df['window_index'] + self._last_window_id, dtype=np.uint)
            self._last_window_id += max_index + 1
            del df['window_index']
            mod_dfs.append(df)
        return pd.concat(mod_dfs, ignore_index=True)

    def _min_pct(self, role: Role) -> int:
        if role == Role.TRAIN:
            return 0
        elif role == Role.VALIDATE:
            return self._pct[Role.TRAIN]
        else:
            return self._pct[Role.TRAIN] + self._pct[Role.VALIDATE]

    def _max_pct(self, role: Role) -> int:
        if role == Role.TRAIN:
            return self._pct[Role.TRAIN]
        elif role == Role.VALIDATE:
            return self._pct[Role.TRAIN] + self._pct[Role.VALIDATE]
        else:
            return 100

    def split(
        self,
        role: Role,
        conf: WindowConfig = DEFAULT_WINDOW_CONFIG,
        rand: Optional[Random] = None
    ) -> pd.DataFrame:
        pos_dfs = [project_df(k, positive_windows(v, conf, rand)) for (k, v) in self._marked.items()]
        all_pos_dfs = self._concat(pos_dfs)
        all_pos_dfs['label'] = True
        neg_dfs = [project_df(k, positive_windows(v, conf, rand)) for (k, v) in self._marked.items()]
        all_neg_dfs = self._concat(neg_dfs)
        all_neg_dfs['label'] = False
        assert len(all_pos_dfs) == len(all_neg_dfs)
        min_pct = self._min_pct(role)
        max_pct = self._max_pct(role)
        ixs = [
            i for i in range(len(all_pos_dfs))
            if self._perm[i % 100] >= min_pct and self._perm[i % 100] < max_pct
        ]
        sel_pos_dfs = all_pos_dfs.iloc[ixs]
        sel_neg_dfs = all_neg_dfs.iloc[ixs]
        return pd.concat((sel_pos_dfs, sel_neg_dfs), ignore_index=True)