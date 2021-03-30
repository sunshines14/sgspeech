import tensorflow as tf

@tf.function
def ctc_loss(y_true, y_pred, input_length, label_length, blank):
    return tf.nn.ctc_loss(
        labels = tf.cast(y_true, tf.int32),
        logit_length=tf.cast(input_length, tf.int32),
        logits=tf.cast(y_pred, tf.float32),
        label_length=tf.cast(label_length, tf.int32),
        logits_time_major=False,
        blank_index=blank
    )