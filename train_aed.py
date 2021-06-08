import os
import argparse
from sgspeech.utils import setup_environment, setup_strategy

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config_audio.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="AED DS2 Training")

parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--max_ckpts", type=int, default=10, help="Max number of checkpoints to keep")

parser.add_argument("--tbs", type=int, default=None, help="Train batch size per replicas")

parser.add_argument("--ebs", type=int, default=None, help="Evaluation batch size per replicas")

parser.add_argument("--tfrecords", default=False, action="store_true", help="Whether to use tfrecords dataset")

parser.add_argument("--devices", type=int, nargs="*", default=[0], help="Devices' ids to apply distributed training")

parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

strategy = setup_strategy(args.devices)

from sgspeech.configs.config import Config
from sgspeech.datasets.speech_dataset import SpeechSliceDataset
from sgspeech.featurizers.speech_featurizer import NumpySpeechFeaturizer
from sgspeech.featurizers.text_featurizer import LabelFeaturizer
from sgspeech.runners.ctc_runners import CTCTrainer
from sgspeech.models.ds2 import DeepSpeech2

config = Config(args.config)
speech_featurizer = NumpySpeechFeaturizer(config.speech_config)
text_featurizer = LabelFeaturizer(config.decoder_config)

train_dataset = SpeechSliceDataset(speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,**vars(config.learning_config.train_dataset_config))
eval_dataset = SpeechSliceDataset(speech_featurizer=speech_featurizer, text_featurizer=text_featurizer,**vars(config.learning_config.eval_dataset_config))

ctc_trainer = CTCTrainer(text_featurizer, config.learning_config.running_config)

# Build DS2 model

with ctc_trainer.strategy.scope():
    ds2_model = DeepSpeech2(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    #tf.print(f"dddd!: {speech_featurizer.shape}")
    ds2_model._build(speech_featurizer.shape)
    ds2_model.summary(line_length=120)
# Compile
ctc_trainer.compile(ds2_model, config.learning_config.optimizer_config,
                    max_to_keep=args.max_ckpts)

ctc_trainer.fit(train_dataset, eval_dataset, train_bs=args.tbs, eval_bs=args.ebs)