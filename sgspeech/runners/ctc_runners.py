import os
import tensorflow as tf

from ..configs.config import RunningConfig
from ..featurizers.text_featurizer import TextFeaturizer
from .base_runners import BaseTrainer
from ..losses.ctc_losses import ctc_loss
from ..utils.utils import get_reduced_length

class CTCTrainer(BaseTrainer):

    def __init__(self,
                 text_featurizer: TextFeaturizer,
                 config: RunningConfig,
                 strategy: tf.distribute.Strategy = None):
        self.text_featurizer = text_featurizer
        super(CTCTrainer, self).__init__(config=config, strategy=strategy)

    def set_train_metrics(self):
        self.train_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("train_ctc_loss", dtype=tf.float32)
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "ctc_loss": tf.keras.metrics.Mean("eval_ctc_loss", dtype=tf.float32)
        }

    def save_model_weights(self):
        with self.strategy.scope():
            self.model.save_weights(os.path.join(self.config.outdir, "latest.h5"))

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        _, features, input_length, labels, label_length, _, _ = batch

        with tf.GradientTape() as tape:
            y_pred = self.model(features, training=True)
            tape.watch(y_pred)
            per_train_loss = ctc_loss(
                y_true = labels, y_pred=y_pred,
                input_length=get_reduced_length(input_length, self.model.time_reduction_factor),
                label_length=label_length,
                blank=self.text_featurizer.blank
            )
            train_loss = tf.nn.compute_average_loss(per_train_loss, global_batch_size=self.global_batch_size)

        gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics["ctc_loss"].update_state(per_train_loss)

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        _, features, input_length, labels, label_length, _, _ = batch

        logits = self.model(features, training=False)

        per_eval_loss = ctc_loss(
            y_true=labels, y_pred=logits,
            input_length=get_reduced_length(input_length, self.model.time_reduction_factor),
            label_length=label_length,
            blank=self.text_featurizer.blank
        )

        self.eval_metrics["ctc_loss"].update_state(per_eval_loss)

    def compile(self, model: tf.keras.Model, optimizer: any, max_to_keep: int=10):
        with self.strategy.scope():
            self.model = model
            self.optimizer = tf.keras.optimizers.get(optimizer)
        self.create_checkpoint_manager(max_to_keep, model=self.model, optimizer=self.optimizer)


