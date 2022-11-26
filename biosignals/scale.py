from dataclasses import dataclass
from typing import Dict
import pandas as pd

# Feature scaling


class Scaler:
    def scale(self, col: str, df: pd.DataFrame):
        raise NotImplementedError()


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


@dataclass(frozen=True)
class RealScaler(Scaler):
    bias: float
    factor: float

    def scale(self, col: str, df: pd.DataFrame):
        raise Exception('TODO')


@dataclass(frozen=True)
class CenteredScaler(Scaler):
    factor: float

    def scale(self, col: str, df: pd.DataFrame):
        raise Exception('TODO')


class BoolScaler(Scaler):
    def scale(self, col: str, df: pd.DataFrame):
        raise Exception('TODO')


class ScalerType:
    def construct(self, col: str, df: pd.DataFrame) -> Scaler:
        raise Exception('TODO')


@dataclass(frozen=True)
class StaticScalerType(ScalerType):
    scaler: Scaler

    def construct(self, col: str, df: pd.DataFrame) -> Scaler:
        self.scaler.scale(col, df)
        return self.scaler


class ShiftScalerType(ScalerType):
    def construct(self, col: str, df: pd.DataFrame) -> Scaler:
        raise Exception('TODO')


class CenteredScalerType(ScalerType):
    def construct(self, col: str, df: pd.DataFrame) -> Scaler:
        raise Exception('TODO')


def construct_all(types: Dict[str, ScalerType], df: pd.DataFrame) -> Dict[str, Scaler]:
    d = {}
    for col, st in types.items():
        sc = st.construct(col, df)
        d[col] = sc
    return d
