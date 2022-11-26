from dataclasses import dataclass
import pickle
from typing import Any, Dict, List, Optional, Tuple
from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import biosignals.prepare as bp
import biosignals.split as bs
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from functools import reduce
import operator
from numpy.random import RandomState
from enum import Enum


# Classification strategy
class Strategy(Enum):
    # Ignore channel ids and train/predict based on observing
    # and one channel. (1-channel-at-a-time samples, 1 classifier)
    # (This keeps unclustered channel info)
    COMBINED = 0
    # Use clustering to order channels and train/predict based
    # on observing all channels. (N-channel-at-a-time samples, 1 classifier)
    # (This removes unclustered channel info)
    MULTI = 1
    # Train one classifier for each channel and vote on the
    # final prediction (1-channel-at-a-time samples, N classifiers)
    # (This removes unclustered channel info)
    ENSEMBLE = 2


# Results (true/false negatives/positives)
@dataclass(frozen=True)
class Results:
    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def size(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    @property
    def accuracy(self) -> float:
        return float(self.tn + self.tp) / self.size

    @classmethod
    def from_pred(cls, y_true: np.ndarray, y_pred: np.ndarray) -> 'Results':
        assert y_pred.shape == y_true.shape
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        assert tn + fp + fn + tp == y_true.shape[0]
        return cls(tn=tn, fp=fp, fn=fn, tp=tp)

    def __add__(self, other: 'Results') -> 'Results':
        return Results(
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tp=self.tp + other.tp,
        )


# Abstract definition for a model
class Model:
    # Train on all train/validate datasets
    # Return final results for training set
    def train_all(
        self,
        train_loaders: List[bp.FrameLoader],
        validate_loaders: List[bp.FrameLoader],
        rand: Optional[RandomState]
    ) -> Results:
        raise NotImplementedError()

    # Test on all test datasets
    def test_all(self, test_loaders: List[bp.FrameLoader]) -> Results:
        raise NotImplementedError()

    # Shorthand for training and testing on a prepared set
    def execute(self, prep_name: str, rand: Optional[RandomState]) -> Tuple[Results, Results]:
        lds = bp.read_prepared(prep_name)
        train_res = self.train_all(lds[bs.Role.TRAIN], lds[bs.Role.VALIDATE], rand)
        test_res = self.test_all(lds[bs.Role.TEST])
        return (train_res, test_res)

    # Save model to the given path
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    # Load model from the given path
    @classmethod
    def load(cls, path: str) -> 'Model':
        with open(path, 'rb') as f:
            return pickle.load(f)


SK_FEATURES = [
    'x', 'y', 'z',
    'theta_power', 'alpha_power', 'beta_power', 'gamma_power'
]


# Load/transform features for single-channel classification
def sk_load_single(loader: bp.FrameLoader) -> Tuple[List[str], str, pd.DataFrame]:
    columns = list(SK_FEATURES)
    columns.extend(['label'])
    df = loader.load(columns)
    return (SK_FEATURES, 'label', df)


# Load/transform features for multi-channel classification
def sk_load_multi(loader: bp.FrameLoader, n_channels: int) -> Tuple[List[str], str, pd.DataFrame]:
    columns = list(SK_FEATURES)
    columns.extend(['cluster_id', 'label'])
    df = loader.load(columns)
    # TODO filter clusters, generate feature columns etc
    raise Exception('TODO')


# Split the given dataframe and shuffle the numpy arrays
def split_df(
    feat_cols: List[str],
    lab_col: str,
    df: pd.DataFrame,
    rand: Optional[RandomState]
) -> Tuple[np.ndarray, np.ndarray]:
    x = df[feat_cols].to_numpy()
    y = df[[lab_col]].to_numpy().ravel()
    assert x.shape == (len(df), len(feat_cols))
    assert y.shape == (len(df),)
    if rand is None:
        return (x, y)
    else:
        x_shuf, y_shuf = shuffle(x, y, random_state=rand)
        assert x_shuf.shape == x.shape
        assert y_shuf.shape == y.shape
        return (x_shuf, y_shuf)


# An sklearn model
class SkModel(Model):
    # NOTE(ejconlon) I don't have a good type for model
    # but it should be an sklearn model instance (i.e. has fit, predict)
    def __init__(self, model_class: Any, model_args: Dict[str, Any], strategy: Strategy):
        assert strategy == Strategy.COMBINED, 'TODO support more strategies'
        self._model = model_class(**model_args)
        self._strategy = strategy

    # Overload this to extract features and labels
    def _load_frame(self, ld: bp.FrameLoader, rand: Optional[RandomState]) -> Tuple[np.ndarray, np.ndarray]:
        assert self._strategy == Strategy.COMBINED, 'TODO support more strategies'
        return split_df(*sk_load_single(ld), rand=rand)

    def _train_one(self, x: np.ndarray, y_true: np.ndarray):
        self._model.fit(x, y_true)

    def _test_one(self, x: np.ndarray, y_true: np.ndarray) -> Results:
        y_pred = self._model.predict(x)
        return Results.from_pred(y_true, y_pred)

    def train_all(
        self,
        train_loaders: List[bp.FrameLoader],
        validate_loaders: List[bp.FrameLoader],
        rand: Optional[RandomState]
    ) -> Results:
        for ld in train_loaders:
            self._train_one(*self._load_frame(ld, rand))
        return self.test_all(train_loaders)

    def test_all(self, test_loaders: List[bp.FrameLoader]) -> Results:
        return reduce(
            operator.add,
            (self._test_one(*self._load_frame(ld, None)) for ld in test_loaders)
        )


# Test training with some sklearn models
def test_models():
    rand = RandomState(42)
    skmodels = [
        (GaussianNB, {}, Strategy.COMBINED),
        (RandomForestClassifier, {}, Strategy.COMBINED),
        # (SVC, {'kernel': 'rbf'}, Strategy.COMBINED),
    ]
    for klass, args, strat in skmodels:
        print(f'Training model {klass} {args} {strat}')
        model = SkModel(klass, args, strat)
        _, tres = model.execute('rand', rand)
        print(tres)
        print('accuracy', tres.accuracy)
