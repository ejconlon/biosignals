from dataclasses import dataclass, replace
import pickle
from typing import Any, Dict, List, Optional, Tuple
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import biosignals.evaluation as be
import biosignals.prepare as bp
import biosignals.split as bs
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from numpy.random import RandomState
from enum import Enum


FEATURES = [
    'x', 'y', 'z',
    'theta_power', 'alpha_power', 'beta_power', 'gamma_power'
]


# Load/transform features for single-channel classification
def load_single_features(
    loader: bp.FrameLoader,
    use_eeg: bool
) -> Tuple[List[str], str, pd.DataFrame]:
    feat_columns = list(FEATURES)
    if use_eeg:
        feat_columns.append('eeg')
    columns = list(feat_columns)
    columns.extend(['label'])
    df = loader.load(columns)
    return (feat_columns, 'label', df)


# Load/transform features for multi-channel classification
# Row-wise is a horrible way to do it but I don't know a way to do the groupby correctly.
def load_multi_features(
    loader: bp.FrameLoader,
    n_clusters: int,
    use_eeg: bool
) -> Tuple[List[str], str, pd.DataFrame]:
    assert n_clusters > 0
    feat_columns = list(FEATURES)
    if use_eeg:
        feat_columns.append('eeg')
    columns = list(feat_columns)
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
    pairs = [(k, i) for i in range(n_clusters) for k in feat_columns]
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
            new_cols[g] = {k: row[k] for k in feat_columns}
            if i == 0:
                assert g not in new_labels
                new_labels[g] = int(row['label'])
    conc_cols: Dict[str, List[Any]] = {f: [] for f in new_feats}
    conc_labels: List[int] = []
    for (w, p) in part_windows.keys():
        for i in range(n_clusters):
            g = (p, i, w)
            for k in feat_columns:
                f = f'{k}_{i}'
                conc_cols[f].append(new_cols[g][k])
            if i == 0:
                conc_labels.append(new_labels[g])
    new_series = {f: pd.Series(conc_cols[f], dtype=float) for f in new_feats}
    new_series['label'] = pd.Series(conc_labels, dtype=int)
    new_df = pd.DataFrame.from_dict(new_series)
    return (new_feats, 'label', new_df)


# Split dataframe into (normal features, extra features, label) arrays
# Normal features may be ['x', 'y', ...] (combined) or ['x_1', 'y_1', ...] (multichannel)
def split_df(
    feat_cols: List[str],
    lab_col: str,
    df: pd.DataFrame,
    rand: Optional[RandomState]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eeg_cols = [f for f in feat_cols if f.startswith('eeg')]
    normal_cols = [f for f in feat_cols if f not in eeg_cols]
    x = df[normal_cols].to_numpy()
    # Hack - get eeg window len
    eeg_len = 0
    wf = df[eeg_cols]
    if len(wf) > 0:
        for c in wf:
            eeg_len = len(wf[c][0])
            break
    # Hack - get np arrays out of dataframe
    wx = []
    for _, row in wf.iterrows():
        vs = [row[c] for c in eeg_cols]
        v = np.array(vs)
        wx.append(v)
    w = np.array(wx).reshape((len(df), len(eeg_cols), eeg_len))
    y = df[[lab_col]].to_numpy().ravel()
    assert x.shape == (len(df), len(normal_cols))
    assert y.shape == (len(df),)
    if rand is None:
        return (x, w, y)
    else:
        x_shuf, w_shuf, y_shuf = shuffle(x, w, y, random_state=rand)
        assert x_shuf.shape == x.shape
        assert w_shuf.shape == w.shape
        assert y_shuf.shape == y.shape
        return (x_shuf, w_shuf, y_shuf)


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
    # ENSEMBLE = 2


# Abstract definition for a model
class Model:
    # Train on all train/validate datasets
    # Return final results for training set
    def train_dataframe(
        self,
        train_loaders: List[bp.FrameLoader],
        validate_loaders: List[bp.FrameLoader],
        rand: Optional[RandomState]
    ) -> be.Results:
        raise NotImplementedError()

    # Test on all test datasets
    def test_dataframe(self, test_loaders: List[bp.FrameLoader]) -> be.Results:
        raise NotImplementedError()

    # Shorthand for training and testing on a prepared set
    def execute(self, prep_name: str, rand: Optional[RandomState]) -> Tuple[be.Results, be.Results]:
        lds = bp.read_prepared(prep_name)
        train_res = self.train_dataframe(lds[bs.Role.TRAIN], lds[bs.Role.VALIDATE], rand)
        test_res = self.test_dataframe(lds[bs.Role.TEST])
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


# Various options to control feature loading
@dataclass(frozen=True)
class FeatureConfig:
    # Single or multi channel features
    strategy: Strategy
    # Use PCA for feature preprocessing?
    use_pca: bool = False
    # If using PCA, how many components
    pca_components: int = 64
    # Load eeg timeseries?
    use_eeg: bool = False


# A model that does some basic feature loading and preprocessing
class FeatureModel(Model):
    def __init__(self, feat_config: FeatureConfig):
        assert feat_config.strategy == Strategy.COMBINED or feat_config.strategy == Strategy.MULTI
        self._feat_config = feat_config
        self._std_scaler = StandardScaler()
        self._max_scaler = MaxAbsScaler()
        self._pca = PCA(n_components=feat_config.pca_components)

    # Load raw features
    def _load_raw(
        self,
        ld: bp.FrameLoader,
        rand: Optional[RandomState]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        feat_cols: List[str]
        lab_col: str
        df: pd.DataFrame
        if self._feat_config.strategy == Strategy.COMBINED:
            feat_cols, lab_col, df = load_single_features(ld, self._feat_config.use_eeg)
        else:
            assert self._feat_config.strategy == Strategy.MULTI
            feat_cols, lab_col, df = load_multi_features(ld, bp.NUM_CLUSTERS, self._feat_config.use_eeg)
        return split_df(feat_cols, lab_col, df, rand)

    # Load processed features
    # Returns x - normal features, w - eeg feature, y - label
    def _load_proc(
        self,
        lds: List[bp.FrameLoader],
        rand: Optional[RandomState],
        is_train: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = []
        ws = []
        ys = []
        for ld in lds:
            x, w, y = self._load_raw(ld, rand)
            xs.append(x)
            ws.append(w)
            ys.append(y)
        x = np.concatenate(xs)
        w = np.concatenate(ws)
        y = np.concatenate(ys)
        if is_train:
            x = self._std_scaler.fit_transform(x)
            if w.shape[1] > 0:
                s = (w.shape[0] * w.shape[1] * w.shape[2], 1)
                w = self._max_scaler.fit_transform(w.reshape(s)).reshape(w.shape)
        else:
            x = self._std_scaler.transform(x)
            if w.shape[1] > 0:
                s = (w.shape[0] * w.shape[1] * w.shape[2], 1)
                w = self._max_scaler.transform(w.reshape(s)).reshape(w.shape)
        if self._feat_config.use_pca:
            if is_train:
                x = self._pca.fit_transform(x)
            else:
                x = self._pca.transform(x)
        # Final sanity check that everything is the right size
        n = x.shape[0]
        assert w.shape[0] == n
        assert y.shape == (n,)
        return (x, w, y)

    # Implement this for training
    # Takes x - normal features, w - eeg feature, y - label
    # x - shaped (num_rows, num_features))
    # eeg data (w - shaped (num_rows, num_eeg_features, eeg_len)))
    # and true labels (y - shaped (num_rows,))
    def train_numpy(self, x: np.ndarray, w: np.ndarray, y_true: np.ndarray) -> be.Results:
        raise NotImplementedError()

    # Implement this for testing
    # Takes x - normal features, w - eeg feature, y - label
    def test_numpy(self, x: np.ndarray, w: np.ndarray, y_true: np.ndarray) -> be.Results:
        raise NotImplementedError()

    def train_dataframe(
        self,
        train_loaders: List[bp.FrameLoader],
        validate_loaders: List[bp.FrameLoader],
        rand: Optional[RandomState]
    ) -> be.Results:
        # Very few models are incremental, so we have to fit all at once,
        # which means we have to load and concat all training data now.
        x, w, y = self._load_proc(train_loaders, rand, is_train=True)
        return self.train_numpy(x, w, y)

    def test_dataframe(self, test_loaders: List[bp.FrameLoader]) -> be.Results:
        x, w, y = self._load_proc(test_loaders, None, is_train=False)
        return self.test_numpy(x, w, y)


# An sklearn model
class SkModel(FeatureModel):
    # NOTE(ejconlon) I don't have a good type for model
    # but it should be an sklearn model instance (i.e. has fit, predict)
    def __init__(self, model_class: Any, model_args: Dict[str, Any], feat_config: FeatureConfig):
        super().__init__(feat_config)
        self._model = model_class(**model_args)

    def train_numpy(self, x: np.ndarray, w: np.ndarray, y_true: np.ndarray) -> be.Results:
        self._model.fit(x, y_true)
        return self.test_numpy(x, w, y_true)

    def test_numpy(self, x: np.ndarray, w: np.ndarray, y_true: np.ndarray) -> be.Results:
        y_pred = self._model.predict(x)
        return be.Results(y_true=y_true, y_pred=y_pred)


# Test training with some sklearn models
def test_models():
    bp.ensure_rand()
    rand = RandomState(42)
    combined_config = FeatureConfig(Strategy.COMBINED)
    eeg_config = replace(combined_config, use_eeg=True)
    multi_config = FeatureConfig(Strategy.MULTI)
    multi_eeg_config = replace(multi_config, use_eeg=True)
    multi_pca_config = replace(multi_config, use_pca=True)
    skmodels = [
        (RandomForestClassifier, {}, combined_config),
        # (RandomForestClassifier, {}, multi_config),
        # (RandomForestClassifier, {}, multi_pca_config),
        # (RandomForestClassifier, {}, eeg_config),
        # (RandomForestClassifier, {}, multi_eeg_config),
    ]
    for klass, args, feat_config in skmodels:
        print(f'Training model {klass} {args} {feat_config}')
        model = SkModel(klass, args, feat_config)
        train_res, test_res = model.execute('rand', rand)
        # print(train_res)
        print('train accuracy', train_res.accuracy())
        # print(test_res)
        print('test accuracy', test_res.accuracy())
        # be.plot_results('train', train_res)
        # be.plot_results('test', test_res)
