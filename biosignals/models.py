# The model designs follow the SS classifier paper.
# They used keras so I've just kept it that way for now... We can switch to torch later if we want...
from dataclasses import dataclass
from typing import Any, List, Tuple
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, Activation  # Dropout, BatchNormalization
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence
from sklearn.naive_bayes import GaussianNB
import biosignals.prepare as bp
import numpy as np
from sklearn.metrics import confusion_matrix
from functools import reduce
import operator


# Results (true/false negatives/positives)
@dataclass(frozen=True)
class Results:
    size: int
    tn: int
    fp: int
    fn: int
    tp: int

    @classmethod
    def from_pred(cls, y_true: np.ndarray, y_pred: np.ndarray) -> 'Results':
        assert len(y_true.shape) == 1
        size = y_true.shape[0]
        assert y_pred.shape == size
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return cls(size=size, tn=tn, fp=fp, fn=fn, tp=tp)

    def __add__(self, other: 'Results') -> 'Results':
        return Results(
            size=self.size + other.size,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tp=self.tp + other.tp,
        )


# Abstract definition for a model
class Model:
    # Train on all train/validate datasets
    # Return final results for validation
    def train_all(
        self,
        train_loaders: List[bp.FrameLoader],
        validate_loaders: List[bp.FrameLoader]
    ) -> Results:
        raise NotImplementedError()

    # Test on all test datasets
    def test_all(self, test_loaders: List[bp.FrameLoader]) -> Results:
        raise NotImplementedError()


# An sklearn model
class SkModel(Model):
    # NOTE(ejconlon) I don't have a good type for model
    # but it should be an sklearn model instance (i.e. has fit, predict)
    def __init__(self, model: Any):
        self._model = model

    # Overload this to extract features and labels
    def load(self, frame_loader: bp.FrameLoader) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def train_one(self, x: np.ndarray, y_true: np.ndarray):
        self._model.fit(x, y_true)

    def test_one(self, x: np.ndarray, y_true: np.ndarray) -> Results:
        y_pred = self._model.predict(x)
        return Results.from_pred(y_true, y_pred)

    def train_all(
        self,
        train_loaders: List[bp.FrameLoader],
        validate_loaders: List[bp.FrameLoader]
    ) -> Results:
        for ld in train_loaders:
            self.train_one(*self.load(ld))
        return self.test_all(validate_loaders)

    def test_all(self, test_loaders: List[bp.FrameLoader]) -> Results:
        return reduce(operator.add, (self.test_one(*self.load(ld)) for ld in test_loaders))


# Constructs and returns a GRU model. Call predict(<data>) on the returned model to make predictions on <data>.
def GRU_Model(train_data, train_labels):
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
    model.fit(train_data, train_labels, epochs=10, batch_size=64, verbose=1)
    return model


# Constructs and returns a LSTM model. Call predict(<data>) on the returned model to make predictions on <data>.
def LSTM_Model(train_data, train_labels):
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
    model.fit(train_data, train_labels, epochs=10, batch_size=64, verbose=1)
    return model


# Constructs and returns a Naive Bayes model. Call predict(<data>) on the returned model to make predictions on <data>.
def Naive_Bayes(train_data, train_labels):
    gnb = GaussianNB()
    return gnb.fit(train_data, train_labels)
