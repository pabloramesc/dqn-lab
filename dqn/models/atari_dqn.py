from keras.layers import Conv2D, Dense, Flatten, Input, InputLayer, Rescaling
from keras.losses import Huber
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam

from dqn.layers import DuelingHead, NoisyDense


def build_atari_vanilla_dqn(state_shape: tuple[int, ...], num_actions: int) -> Model:
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


def build_atari_rainbow_dqn(
    state_shape: tuple[int, ...], num_actions: int, sigma0: float = 0.5
) -> Model:
    inputs = Input(shape=state_shape, dtype="uint8")

    # Normalize inputs
    x = Rescaling(1.0 / 255.0)(inputs)

    x = Conv2D(32, 8, strides=4, activation="relu")(x)
    x = Conv2D(64, 4, strides=2, activation="relu")(x)
    x = Conv2D(64, 3, strides=1, activation="relu")(x)

    x = Flatten()(x)

    # Value head
    v = NoisyDense(512, activation="relu", sigma_init=sigma0)(x)
    v = NoisyDense(1, activation="linear", sigma_init=sigma0)(v)

    # Advantage head
    a = NoisyDense(512, activation="relu", sigma_init=sigma0)(x)
    a = NoisyDense(num_actions, activation="linear", sigma_init=sigma0)(a)

    # Combine value and advantage: V(s) + A(s, a) -> Q(s, a)
    q = DuelingHead(dtype="float32")([v, a])

    model = Model(inputs=inputs, outputs=q)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=Huber(delta=1.0))  # type: ignore
    return model


def load_atari_rainbow_dqn(model_path: str) -> Model:
    model = load_model(
        model_path,
        custom_objects={"DuelingHead": DuelingHead, "NoisyDense": NoisyDense},
        compile=True,
    )
    return model
