from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

# Feature scaling


# Scales a dataframe column in-place
# Assumes the column is in the dataframe.
class Scaler:
    def scale(self, col: str, df: pd.DataFrame):
        raise NotImplementedError()


# Scales all ints in range (inclusive) to float 0.0-1.0
# Everything out of range goes to -1.0
@dataclass(frozen=True)
class IntRangeScaler(Scaler):
    min_val: int
    max_val: int

    @classmethod
    def range(cls, val: int) -> 'IntRangeScaler':
        assert val > 0
        return cls(min_val=0, max_val=val - 1)

    def scale(self, col: str, df: pd.DataFrame):
        raise Exception('TODO')


# Scales by shifting then applying a multiplicative factor
@dataclass(frozen=True)
class RealScaler(Scaler):
    bias: float
    factor: float

    def scale(self, col: str, df: pd.DataFrame):
        raise Exception('TODO')


# Scales by applying a multiplicative factor
@dataclass(frozen=True)
class CenteredScaler(Scaler):
    factor: float

    def scale(self, col: str, df: pd.DataFrame):
        raise Exception('TODO')


# Sends False to 0.0 and True to 1.0
class BoolScaler(Scaler):
    def scale(self, col: str, df: pd.DataFrame):
        raise Exception('TODO')


# Some scalers need to observe some data to come up with a reasonable range
# This will compute that range, scale, and return a scaler for future use.
# If the column is not present, nothing changed or returned.
class ScalerType:
    def construct(self, col: str, df: pd.DataFrame) -> Optional[Scaler]:
        raise Exception('TODO')


# A scaler type that requires no observations.
@dataclass(frozen=True)
class StaticScalerType(ScalerType):
    scaler: Scaler

    def construct(self, col: str, df: pd.DataFrame) -> Optional[Scaler]:
        self.scaler.scale(col, df)
        return self.scaler


# Constructs a ShiftScaler
class ShiftScalerType(ScalerType):
    def construct(self, col: str, df: pd.DataFrame) -> Optional[Scaler]:
        raise Exception('TODO')


# Constructs a CenteredScaler
class CenteredScalerType(ScalerType):
    def construct(self, col: str, df: pd.DataFrame) -> Optional[Scaler]:
        raise Exception('TODO')


# Constructs all scalers from observing and scaling the given dataframe (in-place).
def construct_all(types: Dict[str, ScalerType], df: pd.DataFrame) -> Dict[str, Scaler]:
    # TODO implement
    return {}
    # d = {}
    # for col, st in types.items():
    #     sc = st.construct(col, df)
    #     if sc is not None:
    #         d[col] = sc
    # return d


# Applies all scalers to the dataframe (in-place).
def scale_all(scalers: Dict[str, Scaler], df: pd.DataFrame):
    # TODO implement
    pass
