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


class SequentialModel(bm.FeatureModel):
    def __init__(
        self,
        model_class: Any,
        model_args: Dict[str, Any],
        feat_config: bm.FeatureConfig,
        seq_config: SequentialConfig
    ):
        super().__init__(feat_config)
        self._seq_config = seq_config
        self._model = Sequential()
        self._model_class = model_class

    def test_one(self, x: np.ndarray, y_true: np.ndarray) -> be.Results:
        x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
        x_tf = tf.expand_dims(x_tf, axis=1)
        # y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.int32)
        y_pred_tf = self._model.predict(x_tf)
        y_pred = tf.squeeze(y_pred_tf, axis=1).numpy()
        return be.Results(y_true=y_true, y_pred=y_pred)

    def train_one(self, x: np.ndarray, y_true: np.ndarray) -> be.Results:
        x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
        x_tf = tf.expand_dims(x_tf, axis=1)
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.int32)
        self._model = Sequential()
        self._model.add(
            self._model_class(
                512,
                input_shape=(x_tf.shape[1], x_tf.shape[2]),
                return_sequences=True
            )
        )
        self._model.add(Activation("relu"))
        self._model.add(self._model_class(256))
        self._model.add(Dense(1, activation='sigmoid'))
        self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if self._seq_config.verbose:
            print(self._model.summary())
        self._model.fit(
            x_tf,
            y_true_tf,
            epochs=self._seq_config.num_epochs,
            batch_size=self._seq_config.batch_size,
            verbose=self._seq_config.verbose
        )
        # Now predict
        y_pred_tf = self._model.predict(x_tf)
        y_pred = tf.squeeze(y_pred_tf, axis=1).numpy()
        return be.Results(y_true=y_true, y_pred=y_pred)


# Test training with some deep learning models
def test_models():
    bp.ensure_rand()
    rand = RandomState(42)
    multi_config = bm.FeatureConfig(bm.Strategy.MULTI)
    multi_pca_config = replace(multi_config, use_pca=True)
    seq_config = SequentialConfig()
    deepmodels = [
        (LSTM, {}, multi_config, seq_config),
        (LSTM, {}, multi_pca_config, seq_config),
        (GRU, {}, multi_config, seq_config),
        (GRU, {}, multi_pca_config, seq_config),
    ]
    for klass, args, feat_config, seq_config in deepmodels:
        print(f'Training model {klass} {args} {feat_config} {seq_config}')
        model = SequentialModel(klass, args, feat_config, seq_config)
        train_res, test_res = model.execute('rand', rand)
        print(train_res)
        print('train accuracy', train_res.accuracy())
        print(test_res)
        print('test accuracy', test_res.accuracy())
        be.plot_results('train', train_res)
        be.plot_results('test', test_res)
