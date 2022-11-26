from dataclasses import dataclass
from typing import Any, Dict
import pandas as pd

# Feature scaling


# Scales a dataframe column in-place
# Assumes the column is in the dataframe.
class Scaler:
    def scale(self, col: str, df: pd.DataFrame):
        ws = []
        for _, row in df.iterrows():
            v = row[col]
            w = self._scale_val(v)
            ws.append(w)
        df[col] = pd.Series(ws, dtype=float)

    def _scale_val(self, v: Any) -> float:
        raise NotImplementedError()


# Scales all ints in range (inclusive) to float 0.0-1.0
# Everything out of range goes to -1.0
@dataclass(frozen=True)
class IntRangeScaler(Scaler):
    min_val: int
    max_val: int

    # Corresponds to range(v)
    @classmethod
    def range(cls, v: int) -> 'IntRangeScaler':
        assert v > 0
        return cls(min_val=0, max_val=v - 1)

    def _scale_val(self, v: Any) -> float:
        v = int(v)
        if v < self.min_val or v > self.max_val:
            return -1.0
        else:
            return float(v - self.min_val) / self.max_val


# Scales by shifting then applying a multiplicative factor
@dataclass(frozen=True)
class ShiftScaler(Scaler):
    mean: float
    width: float

    def _scale_val(self, v: Any) -> float:
        v = float(v)
        return float(v - self.mean) / self.width


# Scales by applying a multiplicative factor
@dataclass(frozen=True)
class CenteredScaler(Scaler):
    width: float

    def _scale_val(self, v: Any) -> float:
        return float(v) / self.width


# Sends False to 0.0 and True to 1.0
class BoolScaler(Scaler):
    def _scale_val(self, v: Any) -> float:
        return 1.0 if v else 0.0


# Some scalers need to observe some data to come up with a reasonable range
# This will compute that range, scale, and return a scaler for future use.
# It is assumed the column is in the dataframe.
class ScalerType:
    def construct(self, col: str, df: pd.DataFrame) -> Scaler:
        raise Exception('TODO')


# A scaler type that requires no observations.
@dataclass(frozen=True)
class StaticScalerType(ScalerType):
    scaler: Scaler

    def construct(self, col: str, df: pd.DataFrame) -> Scaler:
        self.scaler.scale(col, df)
        return self.scaler


# Constructs a ShiftScaler
class ShiftScalerType(ScalerType):
    def construct(self, col: str, df: pd.DataFrame) -> Scaler:
        mean = df[col].mean()
        width = df[col].max() - df[col].min()
        scaler = ShiftScaler(mean=mean, width=width)
        scaler.scale(col, df)
        return scaler


# Constructs a CenteredScaler
class CenteredScalerType(ScalerType):
    def construct(self, col: str, df: pd.DataFrame) -> Scaler:
        width = df[col].max() - df[col].min()
        scaler = CenteredScaler(width=width)
        scaler.scale(col, df)
        return scaler


# Constructs all scalers from observing and scaling the given dataframe (in-place).
def construct_all(types: Dict[str, ScalerType], df: pd.DataFrame) -> Dict[str, Scaler]:
    d = {}
    for col, st in types.items():
        if col in df:
            sc = st.construct(col, df)
            d[col] = sc
    return d


# Applies all scalers to the dataframe (in-place).
def scale_all(scalers: Dict[str, Scaler], df: pd.DataFrame):
    for col, sc in scalers.items():
        sc.scale(col, df)
