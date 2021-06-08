import os
import math
import argparse
from sgspeech.utils import setup_environment, setup_strategy

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Conformer Training")

parser.add_argument("--config", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--max_ckpts", type=int, default=10, help="Max number of checkpoints to keep")

parser.add_argument("--tfrecords", default=False, action="store_true", help="Whether to use tfrecords")

parser.add_argument("--tbs", type=int, default=None, help="Train batch size per replica")

parser.add_argument("--ebs", type=int, default=None, help="Evaluation batch size per replica")

parser.add_argument("--devices", type=int, nargs="*", default=[0], help="Devices' ids to apply distributed training")

parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

strategy = setup_strategy(args.devices)

from sgspeech.configs.config import Config
from sgspeech.datasets.speech_dataset import SpeechSliceDataset
from sgspeech.featurizers.speech_featurizer import NumpySpeechFeaturizer
from sgspeech.featurizers.text_featurizer import PhoneFeaturizer
from sgspeech.runners.transducer_runners import TransducerTrainer
from sgspeech.models.conformer import Conformer
from sgspeech.optimizers.schedules import TransformerSchedule

config = Config(args.config)
speech_featurizer = NumpySpeechFeaturizer(config.speech_config)
text_featurizer = PhoneFeaturizer(config.decoder_config)

train_dataset = SpeechSliceDataset(
    speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
    **vars(config.learning_config.train_dataset_config)
)
eval_dataset = SpeechSliceDataset(
    speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,
    **vars(config.learning_config.eval_dataset_config)
)

conformer_trainer = TransducerTrainer(
    config=config.learning_config.running_config,
    text_featurizer=text_featurizer, strategy=strategy
)

with conformer_trainer.strategy.scope():
    # build model
    conformer = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    conformer._build(speech_featurizer.shape)
    conformer.summary(line_length=120)

    optimizer_config = config.learning_config.optimizer_config
    optimizer = tf.keras.optimizers.Adam(
        TransformerSchedule(
            d_model=conformer.dmodel,
            warmup_steps=optimizer_config["warmup_steps"],
            max_lr=(0.05 / math.sqrt(conformer.dmodel))
        ),
        beta_1=optimizer_config["beta1"],
        beta_2=optimizer_config["beta2"],
        epsilon=optimizer_config["epsilon"]
    )

conformer_trainer.compile(model=conformer, optimizer=optimizer,
                          max_to_keep=args.max_ckpts)

conformer_trainer.fit(train_dataset, eval_dataset, train_bs=args.tbs, eval_bs=args.ebs)