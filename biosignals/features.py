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
    # Output:
    # The same input dataframe with feature columns added
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


class ArrayExtractor(Extractor):
    def __init__(self, name: str, fn: Callable[[np.ndarray], Any], dtype: Any):
        self._name = name
        self._fn = fn
        self._dtype = dtype

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        values = []
        for _, row in df.iterrows():
            value = self._fn(row['eeg'])
            values.append(value)
        series = pd.Series(values, dtype=self._dtype)
        return df.assign(**{self._name: series})


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


# This comes from https://mne.tools/dev/auto_examples/time_frequency/time_frequency_global_field_power.html
FREQ_BANDS = [
    ('theta_power', 4, 7),
    ('alpha_power', 8, 12),
    ('beta_power', 13, 25),
    ('gamma_power', 30, 45)
]


# Feature extractors for band power
FREQ_EXTRACTORS = [band_power_extractor(name, lo, hi) for (name, lo, hi) in FREQ_BANDS]


# Feature extractors to use by default
def default_extractors() -> List[Extractor]:
    return list(FREQ_EXTRACTORS)


# Extracts features from the given windows
# Inputs:
# df: dataframe with window_id, channel_id, eeg
# extractors: list of features extractors
# Output:
# dataframe with window_id, channel_id, and feature columns
def extract_features(df: pd.DataFrame, extractors: List[Extractor]) -> pd.DataFrame:
    for ex in extractors:
        df = ex.extract(df)
    return df
