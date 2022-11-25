from typing import Any, Callable, List
import numpy as np
import pandas as pd

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
# df: dataframe with window_id, channel_id, eeg
# extractors: list of features extractors
# Output:
# dataframe with window_id, channel_id, and feature columns
def extract_features(df: pd.DataFrame, extractors: List[Extractor] = DEFAULT_EXTRACTORS) -> pd.DataFrame:
    for ex in extractors:
        df = ex.extract(df)
    return df
