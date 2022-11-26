from typing import Any, Callable, List
import numpy as np
import pandas as pd
import scipy.signal as ss
import scipy.integrate as si
import functools
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


# Compute the average power in the frequency band
# From: https://stackoverflow.com/questions/44547669/python-numpy-equivalent-of-bandpower-from-matlab
def bandpower(freq_low: float, freq_high: float, eeg: np.ndarray) -> float:
    f, Pxx = ss.periodogram(eeg, fs=EEG_SAMPLE_RATE)
    ind_min = int(np.argmax(f > freq_low) - 1)
    ind_max = int(np.argmax(f > freq_high) - 1)
    return si.trapz(Pxx[ind_min:ind_max], f[ind_min:ind_max])


def band_power_extractor(name: str, freq_low: float, freq_high: float) -> Extractor:
    fn = functools.partial(bandpower, freq_low, freq_high)
    return ArrayExtractor(name, fn, np.float64)


# Extracts features from the given windows
# Inputs:
# df: dataframe with window_id, channel_id, eeg
# extractors: list of features extractors
# Side effect:
# Adds feature columns to the input dataframe
def extract_features(df: pd.DataFrame, extractors: List[Extractor]):
    for ex in extractors:
        ex.extract(df)
