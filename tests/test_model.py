from flass.model import conv_model
import tensorflow as tf


def test_conv_model():
    model = conv_model()
    assert isinstance(model, tf.keras.models.Sequential)
