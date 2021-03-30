import tensorflow as tf

class GradientAccumulation:
    def __init__(self, trainable_variables):
        self.gradients = [
            tf.Variable(
                tf.zeros_like(g),
                trainable=False,
                synchronization=tf.VariableSynchronization.ON_READ,
            ) for g in trainable_variables
        ]

    def reset(self):
        for i, g in enumerate(self.gradients):
            self.gradients[i].assign(tf.zeros_like(g), read_value=False)

    def accumulate(self, step_gradients):
        for i, g in enumerate(step_gradients):
            if g is None: continue
            self.gradients[i].assign_add(g, read_value=False)
