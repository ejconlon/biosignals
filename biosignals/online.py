from dataclasses import dataclass, replace
from typing import Any, List, Tuple
import biosignals.dataset as bd
import biosignals.evaluation as be
import biosignals.split as bs
import biosignals.features as bf
import biosignals.prepare as bp
import biosignals.models as bm
import numpy as np
import pandas as pd

# "Online" prediction


# Returns tuple of (offsets, truth)
def extract_onset_truth(
    step_ms: int,
    conf: bs.WindowConfig,
    total_len: int,
    onsets: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(start=conf.pre_len, stop=total_len - conf.post_len, step=step_ms, dtype=int)
    truth = np.zeros(shape=offsets.shape, dtype=int)
    for onset in onsets:
        if onset >= total_len:
            break
        rounded_onset = (onset // step_ms) * step_ms
        index = (rounded_onset - conf.pre_len) // step_ms
        truth[index] = 1
    return (offsets, truth)


@dataclass(frozen=True)
class OnlineConfig:
    part: str
    n_clusters: int
    cluster_df: pd.DataFrame
    strategy: bm.Strategy
    core_features: List[str]
    extractors: List[bf.Extractor]
    use_eeg: bool


def feature_windows(
    windows: np.ndarray,
    conf: OnlineConfig
) -> Tuple[np.ndarray, np.ndarray]:
    print('Projecting to dataframe')
    data_df = bs.project_df(conf.part, windows)
    data_df.rename(columns={'window_index': 'window_id'}, inplace=True)
    print('Adding cluster info and fake label')
    join_df = bd.add_cluster_info(data_df, conf.cluster_df)
    join_df.insert(0, 'label', pd.Series(-1, index=join_df.index, dtype=int))
    feat_columns = list(conf.core_features)
    if conf.use_eeg:
        feat_columns.append('eeg')
    if conf.strategy == bm.Strategy.MULTI:
        # If using multi strategy, throw away unused channels
        # BEFORE we do feature extraction (to save time)
        join_df = join_df.loc[join_df.cluster_id >= 0].reset_index()
    print('Extracting features')
    bf.extract_features(join_df, conf.extractors)
    if conf.strategy == bm.Strategy.MULTI:
        new_feat_cols, _, join_df = bm.process_multi_features(feat_columns, conf.n_clusters, join_df)
        feat_columns = new_feat_cols
    print('Splitting dataframe')
    x, w, _ = bm.split_df(feat_columns, 'label', join_df, None)
    return (x, w)


# The type of online predictor
class Predictor:
    def predict_online(self, windows: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


# An online predictor that uses featurization
class SkPredictor(Predictor):
    def __init__(self, model: Any, conf: OnlineConfig):
        self._model = model
        self._conf = conf

    def predict_online(self, windows: np.ndarray) -> np.ndarray:
        assert len(windows.shape) == 3
        print('Processing windows')
        x, _ = feature_windows(windows, self._conf)
        print('Predicting labels')
        return self._model._model.predict(x)


# Predict onsets as sliding window of eeg
def predict_onsets(
    predictor: Predictor,
    win_conf: bs.WindowConfig,
    offsets: np.ndarray,
    eeg: np.ndarray
) -> np.ndarray:
    extents = [(win_conf.start(o), win_conf.end(o)) for o in offsets]
    windows = np.array([eeg[:, s:e] for (s, e) in extents])
    return predictor.predict_online(windows)


def test_online():
    marked = bd.read_marked_data()
    cluster_df = bp.ensure_clusters()
    part = '01'
    md = marked[part]
    # HACK - testing on a smaller recording
    md = replace(md, eeg=md.eeg[:, 0:10000])
    step_ms = 50
    extractors = bp.default_extractors()
    win_conf = bp.DEFAULT_WINDOW_CONFIG
    print('Calculating truth')
    offsets, y_true = extract_onset_truth(step_ms, win_conf, md.eeg.shape[1], md.onsets)
    online_conf = OnlineConfig(
        part='01',
        n_clusters=bp.NUM_CLUSTERS,
        cluster_df=cluster_df,
        strategy=bm.Strategy.MULTI,
        core_features=bm.FEATURES,
        extractors=extractors,
        use_eeg=False,
    )
    print('Loading model')
    model = bm.Model.load('models/rf_multi')
    predictor = SkPredictor(model, online_conf)
    y_pred = predict_onsets(predictor, win_conf, offsets, md.eeg)
    res = be.Results(y_true=y_true, y_pred=y_pred)
    print(res.accuracy())
