from typing import Optional

import numpy as np
import tensorflow as tf

from . import Model
from ..featurizers.speech_featurizer import NumpySpeechFeaturizer
from ..featurizers.text_featurizer import TextFeaturizer


class CTCModel(Model):
    def __init__(self, **kwargs):
        super(CTCModel, self).__init__(**kwargs)
        self.time_reduction_factor=1

    def _build(self, input_shape):
        features = tf.keras.Input(input_shape, dtype=tf.float32)
        self(features, training=False)

    def add_featurizers(self, speech_featurizer: NumpySpeechFeaturizer, text_featurizer: TextFeaturizer):
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    def call(self, inputs, training=False, **kwargs):
        raise NotImplementedError()


    @tf.function
    def recognize(self, features: tf.Tensor, input_lengths: Optional[tf.Tensor]):
        logits=self(features, training=False)
        probs = tf.nn.softmax(logits)

        def map_fn(prob): return tf.numpy_function(self._perform_greedy, inp=[prob], Tout=tf.string)

        return tf.map_fn(map_fn, probs, fn_output_signature=tf.TensorSpec([], dtype=tf.string))
        # 마무리해야댐

    @tf.function
    def compute_prob(self, features):
        logits=self(features, training=False)
        probs = tf.nn.softmax(logits)

        return probs

    def _perform_greedy(self, probs: np.ndarray):
        # This module was from https://github.com/huseinzol05/malaya-speech/tree/master/ctc-decoders
        from ctc_decoders import ctc_greedy_decoder
        decoded = ctc_greedy_decoder(probs, vocabulary=self.text_featurizer.vocab_array)
        return tf.convert_to_tensor(decoded, dtype=tf.string)

    @tf.function
    def recognize_beam(self, features: tf.Tensor, input_lengths: Optional[tf.Tensor], lm: bool=False):
        logits = self(features, training=False)
        probs = tf.nn.softmax(logits)

        def map_fn(prob): return tf.numpy_function(self._perform_beam_search, inp=[prob, lm], Tout=tf.string)

        return tf.map_fn(map_fn, probs, dtype=tf.string)

    def _perform_beam_search(self, probs: np.ndarray, lm: bool = False):
        from ctc_decoders import ctc_beam_search_decoder
        decoded = ctc_beam_search_decoder(
            probs_seq=probs,
            vocabulary=self.text_featurizer.vocab_array,
            beam_size=self.text_featurizer.decoder_config.beam_width,
            ext_scoring_func=self.text_featurizer.scorer if lm else None
        )
        #tf.print(decoded)
        decoded=decoded[0][-1]

        return tf.convert_to_tensor(decoded, dtype=tf.string)




