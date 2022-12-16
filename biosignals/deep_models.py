from dataclasses import dataclass, replace
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, GRU, Activation, Conv1D, Flatten, Input, MaxPooling1D, Dropout  # Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import GlorotNormal, HeNormal
from keras import regularizers
from typing import Any, Dict, List, Optional, cast
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
    clip_norm: Optional[float] = None
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
        model,
        model_args: Dict[str, Any],
        feat_config: bm.FeatureConfig,
        seq_config: SequentialConfig
    ):
        super().__init__(feat_config)
        self._seq_config = seq_config
        if type(model) is Sequential:
            self._model = model  # We have a Sequantial model
            self._isSequentialModel = True
        elif issubclass(model, tf.keras.Model):
            self._model = model()  # We have a custom class model
            self._isSequentialModel = False
        else:
            print(type(model))
            raise TypeError("Only Sequential or custom model classes are supported.")

    # Takes x - normal features, w - eeg features, y - label. Call this for Sequential models.
    def train_numpy(self, x: np.ndarray, w: np.ndarray, y_true: np.ndarray) -> be.Results:
        if not self._isSequentialModel:
            return self.train_numpy_custom(x, w, y_true)
        w_T = np.swapaxes(w, 1, 2)
        print(w_T.shape)
        w_tf = tf.convert_to_tensor(w_T, dtype=tf.float64)
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.int32)
        self._model.compile(
            optimizer=Adam(clipnorm=self._seq_config.clip_norm),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
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

    # Takes x - normal features, w - eeg features, y - label. Call this for Sequential models.
    def test_numpy(self, x: np.ndarray, w: np.ndarray, y_true: np.ndarray) -> be.Results:
        if not self._isSequentialModel:
            return self.test_numpy_custom(x, w, y_true)
        w_T = np.swapaxes(w, 1, 2)
        w_tf = tf.convert_to_tensor(w_T, dtype=tf.float64)
        y_pred_tf = self._model.predict(w_tf)
        return be.Results(y_true=y_true, y_pred=y_pred_tf)

    # Overridden: Save model to the given directory (must exist)
    def save(self, model_dir: str):
        # Need to save model weights first
        self._model.save(f'{model_dir}/model.tf')
        # Null out the model so it won't be pickled
        # But save it to restore right after
        saved_model = self._model
        self._model = None
        bm.pickle_save(self, model_dir)
        # Restore the model
        self._model = saved_model

    # Overridden: Load model from the given directory (must exist)
    @staticmethod
    def load(model_dir: str) -> 'SequentialModel':
        seq = cast(SequentialModel, bm.pickle_load(model_dir))
        # Now load the model
        seq._model = load_model(f'{model_dir}/model.tf')
        return seq

    # Takes x - normal features, w - eeg features, y - label. Call this for custom model classes (i.e. not Sequential)
    def train_numpy_custom(self, x: np.ndarray, w: np.ndarray, y_true: np.ndarray) -> be.Results:
        w_T = np.swapaxes(w, 1, 2)
        w_tf = tf.convert_to_tensor(w_T, dtype=tf.float64)
        x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
        y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.int32)
        model_inputs = [x_tf, w_tf]

        self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # if self._seq_config.verbose:
        #  print(self._model.summary())
        self._model.fit(
            [x_tf, w_tf],
            y_true_tf,
            epochs=self._seq_config.num_epochs,
            batch_size=self._seq_config.batch_size,
            verbose=self._seq_config.verbose
        )
        # Now predict
        y_pred_tf = self._model.predict([x_tf, w_tf])
        return be.Results(y_true=y_true, y_pred=y_pred_tf)

    # Takes x - normal features, w - eeg features, y - label. Call this for custom model classes (i.e. not Sequential)
    def test_numpy_custom(self, x: np.ndarray, w: np.ndarray, y_true: np.ndarray) -> be.Results:
        w_T = np.swapaxes(w, 1, 2)
        x_tf = tf.convert_to_tensor(x, dtype=tf.float64)
        w_tf = tf.convert_to_tensor(w_T, dtype=tf.float64)
        y_pred_tf = self._model.predict([x_tf, w_tf])
        return be.Results(y_true=y_true, y_pred=y_pred_tf)


class GRUFeatureModel(tf.keras.Model):
    def __init__(self):
        super(GRUFeatureModel, self).__init__()
        self.gru1 = GRU(512, input_shape=(750, 32), return_sequences=True)
        self.activation = Activation("relu")
        self.gru2 = GRU(256)
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.dense3 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        [x, w] = inputs
        out1 = self.gru1(w)
        out1 = self.activation(out1)
        out1 = self.gru2(out1)
        out1 = self.dense1(out1)
        out2 = self.dense2(x)
        out = tf.keras.layers.Add()([out1, out2])
        return self.dense3(out)


class LSTMFeatureModel(tf.keras.Model):
    def __init__(self):
        super(LSTMFeatureModel, self).__init__()
        self.lstm1 = LSTM(512, input_shape=(750, 32), return_sequences=True)
        self.activation = Activation("relu")
        self.lstm2 = LSTM(256)
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.dense3 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        [x, w] = inputs
        out1 = self.lstm1(w)
        out1 = self.activation(out1)
        out1 = self.lstm2(out1)
        out1 = self.dense1(out1)
        out2 = self.dense2(x)
        out = tf.keras.layers.Add()([out1, out2])
        return self.dense3(out)


FEAT_CONFIG = bm.FeatureConfig(strategy=bm.Strategy.MULTI, use_eeg=True)
SEQ_CONFIG = SequentialConfig(num_epochs=30, batch_size=64, verbose=True)


# Create dummy model (for testing only)
def mk_dummy_model():
    dummyModel = Sequential()
    dummyModel.add(LSTM(1, input_shape=(750, 32)))
    dummyModel.add(Dense(1, activation='sigmoid'))
    return SequentialModel(
        dummyModel, {}, FEAT_CONFIG, replace(SEQ_CONFIG, num_epochs=1))


# Create LSTM model
def mk_lstm_model():
    lstmModel = Sequential()
    lstmModel.add(LSTM(512, input_shape=(750, 32), return_sequences=True))
    lstmModel.add(Activation("relu"))
    lstmModel.add(LSTM(256))
    lstmModel.add(Dense(1, activation='sigmoid'))
    return SequentialModel(
        lstmModel, {}, FEAT_CONFIG, SEQ_CONFIG)


# Create GRU model
def mk_gru_model():
    gruModel = Sequential()
    gruModel.add(GRU(512, input_shape=(750, 32), return_sequences=True))
    gruModel.add(Activation("relu"))
    gruModel.add(GRU(256))
    gruModel.add(Dense(1, activation='sigmoid'))
    return SequentialModel(
        gruModel, {}, FEAT_CONFIG, SEQ_CONFIG)


# Create LSTM-CNN model
def mk_lstm_cnn_model():
    clModel = Sequential()
    clModel.add(Input(shape=(750, 32)))     # (None, 750, 32))
    clModel.add(
        LSTM(
            256,
            return_sequences=True,
            activation="relu",
            dropout=0.1,
            kernel_initializer=HeNormal(),
            kernel_regularizer=regularizers.L2(0.001)
        )
    )
    clModel.add(
        LSTM(
            256,
            return_sequences=True,
            kernel_initializer=HeNormal(),
            activation="relu"
        )
    )
    clModel.add(
        Dense(128,
        # kernel_initializer=GlorotNormal()
        )
    )
    clModel.add(
        Conv1D(
            filters=64,
            kernel_size=1,
            strides=1,
            kernel_initializer=HeNormal(),
            activation="relu"
        )
    )
    clModel.add(Flatten())
    clModel.add(
        Dense(64,
        # kernel_initializer=GlorotNormal()
        )
    )
    clModel.add(Dense(1, activation='sigmoid'))
    return SequentialModel(
        clModel, {}, FEAT_CONFIG, replace(SEQ_CONFIG, num_epochs=10))

def mk_cnn_lstm_model():
    clModel = Sequential()
    clModel.add(Input(shape=(750, 32)))     # (None, 750, 32))
    # clModel.add(
    #     Conv1D(
    #         filters=64,
    #         kernel_size=3,
    #         strides=1,
    #         activation="relu",
    #         kernel_initializer=HeNormal()
    #     )
    # )
    clModel.add(
        Conv1D(
            filters=256,
            kernel_size=3,
            strides=1,
            activation="relu",
            kernel_initializer=HeNormal()
        )
    )
    clModel.add(
        MaxPooling1D(
            pool_size=12
        ))
    # clModel.add(
    #     Conv1D(
    #         filters=128,
    #         kernel_size=3,
    #         strides=1,
    #         activation="relu",
    #         kernel_initializer=HeNormal()
    #     )
    # )
    clModel.add(
        Dense(128,
        # kernel_initializer=GlorotNormal()
        )
    )
    clModel.add(
        LSTM(
            256,
            return_sequences=True,
            activation="relu",
            dropout=0.1,
            kernel_initializer=HeNormal(),
            kernel_regularizer=regularizers.L2(0.001)
        )
    )
    clModel.add(
        LSTM(
            256,
            return_sequences=True,
            kernel_initializer=HeNormal(),
            activation="relu"
        )
    )
    clModel.add(
        Dense(128,
        # kernel_initializer=GlorotNormal()
        )
    )
    clModel.add(Flatten())
    clModel.add(
        Dense(64,
        # kernel_initializer=GlorotNormal()
        )
    )
    clModel.add(Dense(1, activation='sigmoid'))
    return SequentialModel(
        clModel, {}, FEAT_CONFIG, replace(SEQ_CONFIG, num_epochs=20, batch_size=256, clip_norm=1))

def mk_gru_feature_model():
    return SequentialModel(GRUFeatureModel, {}, FEAT_CONFIG, SEQ_CONFIG)


def mk_lstm_feature_model():
    return SequentialModel(LSTMFeatureModel, {}, FEAT_CONFIG, SEQ_CONFIG)


MODELS = []


# NOTE: When ready to test with holdouts, change list to bp.STANDARD_PREP_NAMES
# for prep_name in ['rand']:
for prep_name in ['jit']:
    MODELS.extend([
        # bm.ModelCase('dummy', prep_name, mk_dummy_model),
        # bm.ModelCase('lstm', prep_name, mk_lstm_model),
        # bm.ModelCase('gru', prep_name, mk_gru_model),
        # bm.ModelCase('lstm-cnn', prep_name, mk_lstm_cnn_model),
        bm.ModelCase('cnn-lstm', prep_name, mk_cnn_lstm_model),
        # bm.ModelCase('gru-feature', prep_name, mk_gru_feature_model),
        # bm.ModelCase('lstm-feature', prep_name, mk_lstm_feature_model),
    ])


# Test training with some deep learning models
def test_models(models: Optional[List[bm.ModelCase]] = None):
    if models is None:
        models = MODELS
    bp.ensure_all()
    for case in models:
        print(f'Training model {case.model_name} {case.prep_name}')
        rand = RandomState(42)
        model = case.model_fn()
        model.execute(case.model_name, case.prep_name, rand)


def test_model_load():
    for name in ['cnn-lstm_jit']:
        model = SequentialModel.load(f'models/{name}')
        test_res = model.execute_test('jit')
        be.eval_performance(name, 'test', test_res, '/tmp')
