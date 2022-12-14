from enum import Enum
from typing import Any, Dict, List, Set
import numpy as np
import pandas as pd
from dataclasses import dataclass
from random import Random
from biosignals.dataset import MarkedData
import functools

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


# Return true if onset can be used to construct an in-bounds window
def is_valid_onset(onset: int, num_samps: int, conf: WindowConfig) -> bool:
    start = conf.start(onset)
    end = conf.end(onset)
    return start >= 0 and end <= num_samps


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
    conf: WindowConfig,
    rand: Random
) -> np.ndarray:
    assert conf.max_jitter >= 0
    num_samps = marked.eeg.shape[1]
    windows: List[np.ndarray] = []
    for o in marked.onsets:
        if is_valid_onset(o, num_samps, conf):
            use_o: int
            if conf.max_jitter == 0:
                use_o = o
            else:
                while True:
                    use_o = rand.randint(o - conf.max_jitter, o + conf.max_jitter)
                    if is_valid_onset(use_o, num_samps, conf):
                        break
            start = conf.start(use_o)
            end = conf.end(use_o)
            windows.append(marked.eeg[:, start:end])
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
    conf: WindowConfig,
    rand: Random
) -> np.ndarray:
    num_samps = marked.eeg.shape[1]
    windows: List[np.ndarray] = []
    for o in marked.onsets:
        if is_valid_onset(o, num_samps, conf):
            while True:
                target = conf.random_onset(num_samps, rand)
                if not conf.is_near_positive(target, marked.onsets):
                    start = conf.start(target)
                    end = conf.end(target)
                    windows.append(marked.eeg[:, start:end])
                    break
    return np.array(windows, dtype=marked.eeg.dtype)


# Returns windows overlapping by step_ms
def overlap_windows(
    step_ms: int,
    marked: MarkedData,
    conf: WindowConfig,
    rand: Random
) -> np.ndarray:
    total_len = marked.eeg.shape[1]
    offsets = np.arange(start=conf.pre_len, stop=total_len - conf.post_len, step=step_ms, dtype=int)
    extents = [(conf.start(o), conf.end(o)) for o in offsets]
    return np.array([marked.eeg[:, s:e] for (s, e) in extents], dtype=marked.eeg.dtype)


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

    def pretty_name(self) -> str:
        return self.name.lower()


class Splitter:
    def __init__(self):
        self._last_window_id = 0

    # Inputs:
    # role: train/validate/test
    # conf: window config
    # rand: optional random src
    # Output:
    # df with 'part', 'window_id', 'label', 'channel_id', ...
    def split(
        self,
        marked: Dict[str, MarkedData],
        role: Role,
        conf: WindowConfig,
        rand: Random
    ) -> pd.DataFrame:
        pos_dfs = self._select(marked, role, conf, rand, positive_windows, 1)
        neg_dfs = self._select(marked, role, conf, rand, negative_windows, 0)
        assert len(pos_dfs) == len(neg_dfs)
        return pd.concat((pos_dfs, neg_dfs), ignore_index=True)

    def _concat(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        assert len(dfs) > 0
        mod_dfs = []
        for df in dfs:
            max_index = df['window_index'].max()
            df['window_id'] = pd.Series(df['window_index'] + self._last_window_id, dtype=np.uint)
            self._last_window_id += max_index + 1
            mod_dfs.append(df)
        return pd.concat(mod_dfs, ignore_index=True)

    def _select(
        self,
        marked: Dict[str, MarkedData],
        role: Role,
        conf: WindowConfig,
        rand: Random,
        windows: Any,
        label: int
    ) -> pd.DataFrame:
        raise NotImplementedError()


# Generate a random permutations of range(100) to select fairly per participant
def generate_perms(parts: Set[str], rand: Random) -> Dict[str, List[int]]:
    if rand is None:
        rand = Random()
    d = {}
    for part in parts:
        xs = list(range(100))
        rand.shuffle(xs)
        d[part] = xs
    return d


# Splits data randomly by role according to the percentage breakdown and permutation
class RandomSplitter(Splitter):
    def __init__(self, perms: Dict[str, List[int]], pct: Dict[Role, int]):
        super().__init__()
        self._perms = perms
        self._pct = pct
        total_pct = sum(pct.values())
        assert total_pct == 100

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

    def _select(
        self,
        marked: Dict[str, MarkedData],
        role: Role,
        conf: WindowConfig,
        rand: Random,
        windows: Any,
        label: int
    ) -> pd.DataFrame:
        dfs = [project_df(k, windows(v, conf, rand)) for (k, v) in marked.items()]
        all_dfs = self._concat(dfs)
        all_dfs['label'] = pd.Series(label, index=all_dfs.index, dtype=int)
        min_pct = self._min_pct(role)
        max_pct = self._max_pct(role)
        ixs: List[Any] = []
        for ix, row in all_dfs.iterrows():
            p = self._perms[row['part']]
            w = int(row['window_index'])
            x = p[w % 100]
            if x >= min_pct and x < max_pct:
                ixs.append(ix)
        del all_dfs['window_index']
        return all_dfs.iloc[ixs]


# Splits by participant
class PartSplitter(Splitter):
    def __init__(self, role_parts: Dict[Role, Set[str]]):
        super().__init__()
        self._role_parts = role_parts

    def _select(
        self,
        marked: Dict[str, MarkedData],
        role: Role,
        conf: WindowConfig,
        rand: Random,
        windows: Any,
        label: int
    ) -> pd.DataFrame:
        parts = self._role_parts[role]
        dfs = [
            project_df(k, windows(v, conf, rand))
            for (k, v) in marked.items() if k in parts
        ]
        all_dfs = self._concat(dfs)
        all_dfs['label'] = pd.Series(label, index=all_dfs.index, dtype=int)
        return all_dfs


# Splitter for online evaluation
class OnlineSplitter(PartSplitter):
    def __init__(self, step_ms: int, role_parts: Dict[Role, Set[str]]):
        super().__init__(role_parts)
        self._step_ms = step_ms

    def split(
        self,
        marked: Dict[str, MarkedData],
        role: Role,
        conf: WindowConfig,
        rand: Random
    ) -> pd.DataFrame:
        return self._select(
            marked,
            role,
            conf,
            rand,
            functools.partial(overlap_windows, self._step_ms),
            -1
        )
