import tensorflow as tf

class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, constraint=None, regularizer = None, initializer = None, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.constraint = tf.keras.constraints.get(constraint)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings", dtype=tf.float32,
            shape=[self.vocab_size, self.embed_dim],
            initializer=self.initializer,
            trainable=True, regularizer=self.regularizer,
            constraint=self.constraint
        )
        self.built = True

    def call(self, inputs):
        output = tf.cast(tf.expand_dims(inputs, axis=-1), dtype=tf.int32)
        return tf.gather_nd(self.embeddings, outputs)

    def get_config(self):
        conf=super(Embedding, self).get_config()
        conf.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "constraint": self.constraint,
            "regularizer": self.regularizer,
            "initializer": self.initializer
        })
        return conf