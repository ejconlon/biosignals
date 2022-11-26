from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, Activation  # Dropout, BatchNormalization
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence

# Deep learning models - they are here because simply imporing some of the dependencies
# takes a little extra time...

# The model designs follow the SS classifier paper.
# They used keras so I've just kept it that way for now... We can switch to torch later if we want...


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
