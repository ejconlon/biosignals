from dataclasses import dataclass
import pickle
from typing import Any, Dict, List, Optional, Tuple
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import biosignals.prepare as bp
import biosignals.split as bs
import biosignals.evaluation as be
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
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
# Row-wise is a horrible way to do it but I don't know a way to do the groupby correctly.
def sk_load_multi(loader: bp.FrameLoader, n_clusters: int) -> Tuple[List[str], str, pd.DataFrame]:
    assert n_clusters > 0
    columns = list(SK_FEATURES)
    columns.extend(['cluster_id', 'window_id', 'part', 'label'])
    df = loader.load(columns)
    # Find all unique (window_id, part) in the dataframe (bad way to do it)
    part_windows: Dict[Tuple[int, str], int] = {}
    for _, row in df.iterrows():
        i = int(row['cluster_id'])
        if i >= 0 and i < n_clusters:
            p = row['part']
            w = int(row['window_id'])
            t = (w, p)
            if t not in part_windows:
                part_windows[t] = 0
            part_windows[t] += 1
    for t, n in part_windows.items():
        assert n == n_clusters, f'not full cluster for {t}: {n} (expected {n_clusters})'
    # Group and return the new df
    pairs = [(k, i) for i in range(n_clusters) for k in SK_FEATURES]
    new_feats = [f'{k}_{i}' for (k, i) in pairs]
    new_cols: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    new_labels: Dict[Tuple[str, int, int], int] = {}
    # Fill in all the features
    for _, row in df.iterrows():
        i = int(row['cluster_id'])
        if i >= 0 and i < n_clusters:
            p = row['part']
            w = int(row['window_id'])
            g = (p, i, w)
            assert g not in new_cols
            new_cols[g] = {k: row[k] for k in SK_FEATURES}
            if i == 0:
                assert g not in new_labels
                new_labels[g] = int(row['label'])
    conc_cols: Dict[str, List[Any]] = {f: [] for f in new_feats}
    conc_labels: List[int] = []
    for (w, p) in part_windows.keys():
        for i in range(n_clusters):
            g = (p, i, w)
            for k in SK_FEATURES:
                f = f'{k}_{i}'
                conc_cols[f].append(new_cols[g][k])
            if i == 0:
                conc_labels.append(new_labels[g])
    new_series = {f: pd.Series(conc_cols[f], dtype=float) for f in new_feats}
    new_series['label'] = pd.Series(conc_labels, dtype=int)
    new_df = pd.DataFrame.from_dict(new_series)
    return (new_feats, 'label', new_df)


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
        if strategy == Strategy.ENSEMBLE:
            raise Exception('ensemble not supported in this implementation')
        self._model = model_class(**model_args)
        self._strategy = strategy

    def _load_one(self, ld: bp.FrameLoader, rand: Optional[RandomState]) -> Tuple[np.ndarray, np.ndarray]:
        if self._strategy == Strategy.COMBINED:
            return split_df(*sk_load_single(ld), rand=rand)
        else:
            assert self._strategy == Strategy.MULTI
            return split_df(*sk_load_multi(ld, bp.NUM_CLUSTERS), rand=rand)

    def _load_all(self, lds: List[bp.FrameLoader], rand: Optional[RandomState]) -> Tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for ld in lds:
            x, y = self._load_one(ld, rand)
            xs.append(x)
            ys.append(y)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        return (x, y)

    def _train_one(self, x: np.ndarray, y_true: np.ndarray):
        self._model.fit(x, y_true)

    def _test_one(self, x: np.ndarray, y_true: np.ndarray) -> Results:
        y_pred = self._model.predict(x)
        # NOTE: Don't want to pop up window when running on command line!
        # Need to return something and evaluate it later.
        # be.evaluate_model(y_pred, y_true)
        return Results.from_pred(y_true, y_pred)

    def train_all(
        self,
        train_loaders: List[bp.FrameLoader],
        validate_loaders: List[bp.FrameLoader],
        rand: Optional[RandomState]
    ) -> Results:
        # Very few models are incremental, so we have to fit all at once,
        # which means we have to load and concat all training data now.
        x, y = self._load_all(train_loaders, rand)
        # Not every model has fit_predict, so we have to fit and predict
        # separately if we want to see perf on the training set.
        self._train_one(x, y)
        return self._test_one(x, y)

    def test_all(self, test_loaders: List[bp.FrameLoader]) -> Results:
        x, y = self._load_all(test_loaders, None)
        return self._test_one(x, y)


# Test training with some sklearn models
def test_models():
    rand = RandomState(42)
    skmodels = [
        # (GaussianNB, {}, Strategy.COMBINED),
        # (GaussianNB, {}, Strategy.MULTI),
        (RandomForestClassifier, {}, Strategy.COMBINED),
        (RandomForestClassifier, {}, Strategy.MULTI),
        # (SVC, {'kernel': 'rbf'}, Strategy.COMBINED),
    ]
    for klass, args, strat in skmodels:
        print(f'Training model {klass} {args} {strat}')
        model = SkModel(klass, args, strat)
        _, tres = model.execute('rand', rand)
        print(tres)
        print('accuracy', tres.accuracy)
