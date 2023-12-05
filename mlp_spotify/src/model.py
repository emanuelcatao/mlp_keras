from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='tanh', input_shape=(input_shape,)),
        Dense(64, activation='tanh'),
        Dense(16, activation='tanh'),
        Dense(8, activation='tanh'),
        Dense(4, activation='tanh'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
