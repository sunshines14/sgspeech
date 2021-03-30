import abc
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, name, **kwargs):
        super(Model, self).__init__(name=name, **kwargs)

    def _build(self, *args, **kwargs):
        raise NotImplementedError()

    def call(self, inputs, training=False, **kwargs):
        raise NotImplementedError()

    def recognize(self, features, input_lengths, **kwargs):
        pass

    def recognize_beam(self, features, input_lenghts, **kwargs):
        pass