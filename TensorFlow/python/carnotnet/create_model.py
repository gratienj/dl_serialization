"""
Create, train and save a carnot net
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

from pathlib import Path


class ScaleLayer(keras.layers.Layer):
    """
    A custom layer used to normalize the input.
    """

    def __init__(self, scale, mean, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.scale = tf.constant(scale, dtype="float32", name="scale")
        self.mean = tf.constant(mean, dtype="float32", name="mean")

    def call(self, inputs):
        return (inputs - self.mean) / self.scale

    def get_config(self):
        config = {"scale": self.scale.numpy(), "mean": self.mean.numpy()}
        base_config = super(ScaleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PSwish(keras.layers.Layer):
    """
    Parametric swish activation function
    It follows: f(x) = x * sigmoid(beta * x),
    where beta is a learned array with the same shape as x.
    """

    def __init__(self, beta_initializer="ones", **kwargs):
        super(PSwish, self).__init__(**kwargs)
        self.beta_initializer = keras.initializers.get(beta_initializer)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.beta = self.add_weight(
            shape=param_shape, name="beta", initializer=self.beta_initializer
        )
        self.built = True

    def call(self, inputs):
        return inputs * tf.nn.sigmoid(self.beta * inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "beta_initializer": keras.initializers.serialize(self.beta_initializer)
        }
        base_config = super(PSwish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    cwd = Path.cwd()
    prj_path = None
    if str(cwd).endswith("dl_serialization"):
        prj_path = cwd
    else:
        for parent in cwd.parents:
            if str(parent).endswith("dl_serialization"):
                prj_path = parent
    data_path = prj_path.joinpath("data", "carnotnet", "data.npy")
    # Load data which has 100 rows and 6 columns
    # The data are the results of flash caluclations for the mixture of water and methane
    with open(data_path, "rb") as f:
        data = np.load(f)
    # Use the first 80 rows for the training and the last 20 for the test
    # The first 4 columns are the inputs, that is, pressure, temperature and composition
    # The last 2 columns are the targets, that is, fugacity coefficients of water and methane
    x_train, y_train = data[:80, :4], data[:80, 4:]
    x_test, y_test = data[80:, :4], data[80:, 4:]

    scale = np.array([5.74463518e07, 1.52131796e02, 2.88556360e-01, 2.88556360e-01])
    mean = np.array([1.00500000e08, 5.36500000e02, 5.00071078e-01, 4.99928922e-01])

    # Create a model using tensorflow functional API
    keras.backend.clear_session()
    input_ = keras.layers.Input(shape=(4,), name="input")
    scale_layer = ScaleLayer(scale=scale, mean=mean, name="scale_layer")(input_)
    hidden = scale_layer
    for i in range(1, 3):
        hidden = Dense(units=16, kernel_initializer="he_normal", name=f"hidden{i}")(hidden)
        hidden = PSwish(name=f"pswish{i}")(hidden)
    concat = keras.layers.Concatenate()([scale_layer, hidden])
    output_ = Dense(units=2, kernel_initializer="glorot_normal")(concat)
    model = keras.Model(inputs=input_, outputs=output_, name="carnotnet")
    print(model.summary())

    # Compile, train and evaluate the model
    model.compile(optimizer="adam", loss="mae", metrics=["mse"])
    model.fit(x_train, y_train, batch_size=80, epochs=400, verbose=0)
    print("Evaluation on the test set:")
    model.evaluate(x_test, y_test, batch_size=20)

    # Save the model
    saved_model_dir = prj_path.joinpath("model", "carnotnet")
    model.save(saved_model_dir, save_format="tf")

    # Load the model
    model2 = keras.models.load_model(saved_model_dir)

    # Use the loaded model to predict 2 test examples
    pred = model2.predict(x_test[:2, :])

    print("Everything is OK!")
