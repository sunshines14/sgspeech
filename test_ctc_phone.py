import os
import argparse
from sgspeech.utils import setup_environment, setup_devices

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(prog="Deep Speech 2 Tester")

parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")

parser.add_argument("--saved", type=str, default=None, help="Path to the model file to be exported")

parser.add_argument("--tfrecords", default=False, action="store_true", help="Whether to use tfrecords dataset")

parser.add_argument("--mxp", default=False, action="store_true", help="Enable mixed precision")

parser.add_argument("--device", type=int, default=0, help="Device's id to run test on")

parser.add_argument("--output_name", type=str, default="test", help="Result filename name prefix")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

setup_devices([args.device])

from sgspeech.configs.config import Config
from sgspeech.datasets.speech_dataset import SpeechSliceDataset
from sgspeech.featurizers.speech_featurizer import NumpySpeechFeaturizer
from sgspeech.featurizers.text_featurizer import PhoneFeaturizer
from sgspeech.runners.base_runners import BaseTester
from sgspeech.models.ds2 import DeepSpeech2

tf.random.set_seed(0)
#assert args.export

config = Config(args.config)
speech_featurizer = NumpySpeechFeaturizer(config.speech_config)
text_featurizer = PhoneFeaturizer(config.decoder_config)

# Build DS2 model
ds2_model = DeepSpeech2(**config.model_config, vocabulary_size=text_featurizer.num_classes)
ds2_model._build(speech_featurizer.shape)
ds2_model.load_weights(args.saved)
ds2_model.summary(line_length=120)
ds2_model.add_featurizers(speech_featurizer, text_featurizer)

#tf.print(config.learning_config.test_dataset_config)
#tf.print(config.learning_config.eval_dataset_config)
#tf.print(type(config.learning_config.test_dataset_config))
#tf.print(type(config.learning_config.eval_dataset_config))
test_dataset = SpeechSliceDataset(speech_featurizer=speech_featurizer, text_featurizer=text_featurizer, **vars(config.learning_config.test_dataset_config))

ctc_tester = BaseTester(
    config=config.learning_config.running_config,
    output_name=args.output_name
)
ctc_tester.compile(ds2_model)
ctc_tester.run(test_dataset)