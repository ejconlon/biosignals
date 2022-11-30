import os
import shutil
from dataclasses import dataclass, replace
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, Activation  # Dropout, BatchNormalization
from typing import Any, Dict
import biosignals.models as bm
import biosignals.evaluation as be
from numpy.random import RandomState
import numpy as np
import biosignals.prepare as bp
import tensorflow as tf

# Deep learning models - they are here because simply imporing some of the dependencies
# takes a little extra time...

# The model designs follow the SS classifier paper.
# They used keras so I've just kept it that way for now... We can switch to torch later if we want...


@dataclass(frozen=True)
class SequentialConfig:
    num_epochs: int = 30
    batch_size: int = 64
    verbose: bool = True


# # Example interface for filling in neural network structure
# class ModelBuilder:
#     def build(self, model: Sequential, shape: List[int]):
#         raise NotImplementedError()

# # Example
# class LstmModelBuilder:
#     def build(self, model: Sequential, shape: List[int]):
#        model.add(
#             LSTM(
#                 512,
#                 input_shape=tuple(shape),
#                 return_sequences=True
#             )
#         )
#         model.add(Activation("relu"))
#         model.add(LSTM(256))
#         model.add(Dense(1, activation='sigmoid'))
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


class SequentialModel(bm.FeatureModel):
    def __init__(
        self,
        model: Sequential,
        model_args: Dict[str, Any],
        feat_config: bm.FeatureConfig,
        seq_config: SequentialConfig
    ):
        super().__init__(feat_config)
        self._seq_config = seq_config
        self._model = model

    # Takes x - normal features, w - eeg features, y - label
    def train_numpy(self, x: np.ndarray, w: np.ndarray, y_true: np.ndarray) -> be.Results:
        w_T = np.swapaxes(w, 1, 2)
        print(w_T.shape)
        w_tf = tf.convert_to_tensor(w_T, dtype=tf.float64)
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.int32)
        self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if self._seq_config.verbose:
            print(self._model.summary())
        self._model.fit(
            w_tf,
            y_true_tf,
            epochs=self._seq_config.num_epochs,
            batch_size=self._seq_config.batch_size,
            verbose=self._seq_config.verbose
        )
        # Now predict
        y_pred_tf = self._model.predict(w_tf)
        return be.Results(y_true=y_true, y_pred=y_pred_tf)

    # Takes x - normal features, w - eeg features, y - label
    def test_numpy(self, x: np.ndarray, w: np.ndarray, y_true: np.ndarray) -> be.Results:
        w_T = np.swapaxes(w, 1, 2)
        w_tf = tf.convert_to_tensor(w_T, dtype=tf.float64)
        y_pred_tf = self._model.predict(w_tf)
        return be.Results(y_true=y_true, y_pred=y_pred_tf)


# Test training with some deep learning models
def test_models():
    bp.ensure_rand()
    rand = RandomState(42)
    multi_config = bm.FeatureConfig(bm.Strategy.MULTI)
    multi_eeg_config = bm.FeatureConfig(strategy=bm.Strategy.MULTI, use_eeg=True)
    multi_pca_config = replace(multi_config, use_pca=True)
    seq_config = SequentialConfig(num_epochs=30, batch_size=64, verbose=True)

    # Create LSTM model
    lstmModel = Sequential()
    lstmModel.add(LSTM(512, input_shape=(750, 32), return_sequences=True))
    lstmModel.add(Activation("relu"))
    lstmModel.add(LSTM(256))
    lstmModel.add(Dense(1, activation='sigmoid'))

    # Create GRU model
    gruModel = Sequential()
    gruModel.add(GRU(512, input_shape=(750, 32), return_sequences=True))
    gruModel.add(Activation("relu"))
    gruModel.add(GRU(256))
    gruModel.add(Dense(1, activation='sigmoid'))

    deepmodels = [
        ('lstm', lstmModel, {}, multi_eeg_config, seq_config),
        ('gru', gruModel, {}, multi_eeg_config, seq_config),
    ]
    os.makedirs('models', exist_ok=True)
    for name, klass, args, feat_config, seq_config in deepmodels:
        print(f'Training model {name} {klass} {args} {feat_config} {seq_config}')
        dest_dir = f'models/{name}'
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)
        model = SequentialModel(klass, args, feat_config, seq_config)
        train_res, test_res = model.execute('rand', rand)
        be.eval_performance(name, 'train', train_res, dest_dir)
        be.eval_performance(name, 'test', test_res, dest_dir)
        be.plot_results(name, 'train', train_res, dest_dir)
        be.plot_results(name, 'test', test_res, dest_dir)
