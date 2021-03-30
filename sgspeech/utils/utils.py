
import math
import os
from typing import Union, List

import tensorflow as tf
import numpy as np

def get_num_batches(samples, batch_size, drop_remainders=True):
    if samples is None or batch_size is None: return None
    if drop_remainders: return math.floor(float(samples) / float(batch_size))
    return math.ceil(float(samples) / float(batch_size))

def preprocess_paths(paths: Union[List, str]):
    if isinstance(paths, list):
        return [os.path.abspath(os.path.expanduser(path)) for path in paths]
    elif isinstance(paths, str):
        return os.path.abspath(os.path.expanduser(paths))
    else:
        return None

def shape_list(x, out_type=tf.int32):
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=out_type)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def get_reduced_length(length, reduction_factor):
    return tf.cast(tf.math.ceil(tf.divide(length, tf.cast(reduction_factor, dtype=length.dtype))), dtype=tf.int32)

def get_rnn(rnn_type:str):
    assert rnn_type in ["lstm", "gru", "rnn"]
    if rnn_type.lower() == "lstm": return tf.keras.layers.LSTM
    if rnn_type.lower() == "gru": return tf.keras.layers.GRU
    return tf.keras.layers.SimpleRNN

def get_conv(conv_type):
    assert conv_type in ["conv1d", "conv2d"]

    if conv_type == "conv1d":
        return tf.keras.layers.Conv1D

    return tf.keras.layers.Conv2D

def merge_two_last_dims(x):
    b, _, f, c = shape_list(x)
    return tf.reshape(x, shape=[b, -1, f*c])


def bytes_to_string(array: np.ndarray, encoding: str = "utf-8"):
    if array is None: return None
    return [transcript.decode(encoding) for transcript in array]

def count_non_blank(tensor: tf.Tensor, blank: int or tf.Tensor = 0, axis=None):
    return tf.reduce_sum(tf.where(tf.not_equal(tensor, blank), x=tf.ones_like(tensor), y=tf.zeros_like(tensor)), axis=axis)

def find_max_length_prediction_tfarray(tfarray: tf.TensorArray) -> tf.Tensor:
    with tf.name_scope("find_max_length_prediction_tfarray"):
        index = tf.constant(0, dtype=tf.int32)
        total = tfarray.size()
        max_length = tf.constant(0, dtype=tf.int32)

        def condition(index, _): return tf.less(index, total)

        def body(index, max_length):
            prediction = tfarray.read(index)
            length = tf.shape(prediction)[0]
            max_length = tf.where(tf.greater(length, max_length), length, max_length)
            return index + 1, max_length

        index, max_length = tf.while_loop(condition, body, loop_vars=[index, max_length], swap_memory=False)
        return max_length


def pad_prediction_tfarray(tfarray: tf.TensorArray, blank: int or tf.Tensor) -> tf.TensorArray:
    with tf.name_scope("pad_prediction_tfarray"):
        index = tf.constant(0, dtype=tf.int32)
        total = tfarray.size()
        max_length = find_max_length_prediction_tfarray(tfarray)

        def condition(index, _): return tf.less(index, total)

        def body(index, tfarray):
            prediction = tfarray.read(index)
            prediction = tf.pad(
                prediction, paddings=[[0, max_length - tf.shape(prediction)[0]]], mode="CONSTANT", constant_values=blank
            )
            tfarray = tfarray.write(index, prediction)
            return index +1, tfarray
        index, tfarray = tf.while_loop(condition, body, loop_vars=[index, tfarray], swap_memory=False)
        return tfarray


