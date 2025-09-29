from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Flatten, InputLayer, Rescaling
from keras.losses import Huber
from keras.optimizers import Adam

def build_atari_dqn(state_shape: tuple[int, ...], num_actions: int) -> Model:
    model = Sequential(
        [
            InputLayer(shape=state_shape, dtype="uint8"),
            Rescaling(1.0 / 255.0),
            Conv2D(32, 8, strides=4, activation="relu"),
            Conv2D(64, 4, strides=2, activation="relu"),
            Conv2D(64, 3, strides=1, activation="relu"),
            Flatten(),
            Dense(512, activation="relu"),
            Dense(num_actions, activation="linear", dtype="float32"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.00025),  # type: ignore
        loss=Huber(delta=1.0),
    )

    return model