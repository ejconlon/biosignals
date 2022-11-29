from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, Activation  # Dropout, BatchNormalization
from typing import Any, Dict, List, Optional, Tuple
from biosignals.models import Strategy, Model, Results, split_df, load_single_features, load_multi_features
from numpy.random import RandomState
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import biosignals.prepare as bp
import biosignals.evaluation as be
import tensorflow as tf

# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence

# Deep learning models - they are here because simply imporing some of the dependencies
# takes a little extra time...

# The model designs follow the SS classifier paper.
# They used keras so I've just kept it that way for now... We can switch to torch later if we want...


class SequentialModel(Model):

    def __init__(self, model_class: Any, model_args: Dict[str, Any], strategy: Strategy):
        assert strategy == Strategy.COMBINED or strategy == Strategy.MULTI
        self._usePCA = model_args.get("usePCA", None)
        self._model = Sequential()
        self._scaler = StandardScaler()
        self._model_class = model_class
        self._pca = PCA(n_components=64)
        self._strategy = strategy

    def _load_one(self, ld: bp.FrameLoader, rand: Optional[RandomState]) -> Tuple[np.ndarray, np.ndarray]:
        if self._strategy == Strategy.COMBINED:
            return split_df(*load_single_features(ld), rand=rand)
        else:
            assert self._strategy == Strategy.MULTI
            return split_df(*load_multi_features(ld, bp.NUM_CLUSTERS), rand=rand)

    def _load_all(self, lds: List[bp.FrameLoader],
                  isTraining: bool,
                  rand: Optional[RandomState]) -> Tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for ld in lds:
            x, y = self._load_one(ld, rand)
            xs.append(x)
            ys.append(y)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        self._scaler.fit(x)
        x = self._scaler.transform(x)
        if self._usePCA:
            if isTraining:
                self._pca.fit(x)
            x = self._pca.transform(x)
        x = tf.expand_dims(x, axis=1)
        y = tf.convert_to_tensor(y, dtype=tf.int32)
        return (x, y)

    def _test_one(self, x: np.ndarray, y_true: np.ndarray) -> Results:
        y_pred = self._model.predict(x)
        be.evaluate_model(y_pred, y_true)
        y_pred = tf.squeeze(y_pred, axis=1)
        return Results.from_pred(y_true, y_pred)

    # Train on all train/validate datasets
    # Return final results for training set
    def train_all(
        self,
        train_loaders: List[bp.FrameLoader],
        validate_loaders: List[bp.FrameLoader],
        rand: Optional[RandomState],
        num_epochs=30,
        batch_size=64,
        verbose=1,
    ) -> Results:
        train_data, train_labels = self._load_all(train_loaders, True, rand)
        self._model = Sequential()
        self._model.add(self._model_class(512, input_shape=(train_data.shape[1], train_data.shape[2]),
                        return_sequences=True))
        self._model.add(Activation("relu"))
        self._model.add(self._model_class(256))
        self._model.add(Dense(1, activation='sigmoid'))
        self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self._model.summary())
        self._model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, verbose=verbose)
        return self._test_one(train_data, train_labels)

    # Test on all test datasets
    def test_all(self, test_loaders: List[bp.FrameLoader]) -> Results:
        x, y = self._load_all(test_loaders, False, None)
        return self._test_one(x, y)


# Constructs and returns a GRU model. Call predict(<data>) on the returned model to make predictions on <data>.
def GRU_Model(train_data, train_labels, num_epochs=10, batch_size=64, verbose=1):
    model = Sequential()
    model.add(GRU(512, input_shape=(train_data.shape[1], train_data.shape[2]), return_sequences=True))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(GRU(256))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.build()
    print(model.summary())
    model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, verbose=verbose)
    return model


# Constructs and returns a LSTM model. Call predict(<data>) on the returned model to make predictions on <data>.
def LSTM_Model(train_data, train_labels, num_epochs=10, batch_size=64, verbose=1):
    model = Sequential()
    model.add(LSTM(512, input_shape=(train_data.shape[1], train_data.shape[2]), return_sequences=True))
    # model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(LSTM(256))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.build()
    print(model.summary())
    model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size, verbose=verbose)
    return model


# Test training with some deep learning models
def test_models():
    rand = RandomState(42)
    deepmodels = [
        (LSTM, {"usePCA": False}, Strategy.MULTI),
        (LSTM, {"usePCA": True}, Strategy.MULTI),
        (GRU, {"usePCA": False}, Strategy.MULTI),
        (GRU, {"usePCA": True}, Strategy.MULTI),
    ]
    for klass, args, strat in deepmodels:
        print(f'Training model {klass} {args} {strat}')
        model = SequentialModel(klass, args, strat)
        _, tres = model.execute('rand', rand)
        print(tres)
        print('accuracy', tres.accuracy)
