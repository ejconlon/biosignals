from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import numpy as np
import pandas as pd
import scipy.signal as ss
import scipy.integrate as si
from biosignals.dataset import EEG_SAMPLE_RATE

# Feature extraction


# Abstractly, something that extracts features from data
class Extractor:
    # Inputs:
    # df: Dataframe with columns 'window_id', 'channel_id', 'eeg', etc
    # Side effects:
    # Mutates the given dataframe to add columns
    def extract(self, df: pd.DataFrame):
        raise NotImplementedError()


class ArrayExtractor(Extractor):
    def __init__(self, name: str, fn: Callable[[np.ndarray], Any], dtype: Any):
        self._name = name
        self._fn = fn
        self._dtype = dtype

    def extract(self, df: pd.DataFrame):
        values = []
        for _, row in df.iterrows():
            value = self._fn(row['eeg'])
            values.append(value)
        series = pd.Series(values, dtype=self._dtype)
        df.insert(0, self._name, series)


@dataclass(frozen=True)
class Band:
    name: str
    freq_low: float
    freq_high: float


# # Compute the average power in the frequency band
# # From: https://stackoverflow.com/questions/44547669/python-numpy-equivalent-of-bandpower-from-matlab
# class BandPowerExtractor(Extractor):
#     def __init__(self, bands: List[Band]):
#         self._bands = bands

#     def _power(self, f: np.ndarray, Pxx: np.ndarray, freq_low: float, freq_high: float) -> float:
#         ind_min = int(np.argmax(f > freq_low) - 1)
#         ind_max = int(np.argmax(f > freq_high) - 1)
#         return si.trapz(Pxx[ind_min:ind_max], f[ind_min:ind_max])

#     def extract(self, df: pd.DataFrame):
#         values: Dict[str, List[float]] = {b.name: [] for b in self._bands}
#         for _, row in df.iterrows():
#             eeg = row['eeg']
#             f, Pxx = ss.periodogram(eeg, fs=EEG_SAMPLE_RATE)
#             for b in self._bands:
#                 power = self._power(f, Pxx, b.freq_low, b.freq_high)
#                 values[b.name].append(power)
#         for name, vals in values.items():
#             series = pd.Series(vals, dtype=float)
#             df.insert(0, name, series)


class SegmentBandPowerExtractor(Extractor):
    def __init__(self, n_segments: int, bands: List[Band]):
        self._n_segments = n_segments
        self._bands = bands

    def _power(self, f: np.ndarray, Pxx: np.ndarray, freq_low: float, freq_high: float) -> float:
        ind_min = int(np.argmax(f > freq_low) - 1)
        ind_max = int(np.argmax(f > freq_high) - 1)
        return si.trapz(Pxx[ind_min:ind_max], f[ind_min:ind_max])

    def extract(self, df: pd.DataFrame):
        values: Dict[str, List[float]] = {}
        for _, row in df.iterrows():
            eeg = np.array(row['eeg'])
            assert len(eeg.shape) == 1
            assert eeg.shape[0] % self._n_segments == 0,\
                f'n_segments {self._n_segments} does not evenly divide eeg len {eeg.shape[0]}'
            seg_len = eeg.shape[0] // self._n_segments
            for seg_num in range(self._n_segments):
                start = seg_len * seg_num
                end = start + seg_len
                seg_eeg = eeg[start:end]
                f, Pxx = ss.periodogram(seg_eeg, fs=EEG_SAMPLE_RATE)
                for b in self._bands:
                    seg_name: str
                    if self._n_segments == 1:
                        seg_name = b.name
                    else:
                        seg_name = f'{b.name}_s{seg_num}'
                    power = self._power(f, Pxx, b.freq_low, b.freq_high + 1)
                    if seg_name not in values:
                        values[seg_name] = []
                    values[seg_name].append(power)
        for name, vals in values.items():
            series = pd.Series(vals, dtype=float)
            df.insert(0, name, series)


# Extracts features from the given windows
# Inputs:
# df: dataframe with window_id, channel_id, eeg
# extractors: list of features extractors
# Side effect:
# Adds feature columns to the input dataframe
def extract_features(df: pd.DataFrame, extractors: List[Extractor]):
    for ex in extractors:
        ex.extract(df)
