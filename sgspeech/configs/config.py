
from ..utils.utils import preprocess_paths
from . import load_yaml

class DecoderConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.beam_width = config.pop("beam_width", 0)
        self.blank_at_zero = config.pop("blank_at_zero", True)
        self.norm_score = config.pop("norm_score", True)
        self.lm_config = config.pop("lm_config", {})

        self.vocabulary = preprocess_paths(config.pop("vocabulary", None))
        self.target_vocab_size = config.pop("target_vocab_size", 1024)
        self.max_subword_length = config.pop("max_subword_length", 4)
        self.output_path_prefix = preprocess_paths(config.pop("output_path_prefix", None))
        self.model_type = config.pop("model_type", None)
        self.corpus_files = preprocess_paths(config.pop("corpus_files", []))
        self.max_corpus_chars = config.pop("max_corpus_chars", None)
        self.reserved_tokens = config.pop("reserved_tokens", None)

        for k, v in config.items(): setattr(self, k, v)


class RunningConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.batch_size = config.pop("batch_size", 1)
        self.accumulation_steps = config.pop("accumulation_steps", 1)
        self.num_epochs = config.pop("num_epochs", 20)
        self.outdir = preprocess_paths(config.pop("outdir", None))
        self.log_interval_steps = config.pop("log_interval_steps", 500)
        self.save_interval_steps = config.pop("save_interval_steps", 500)
        self.eval_interval_steps = config.pop("eval_interval_steps", 1000)
        for k, v in config.items(): setattr(self, k, v)


class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.stage = config.pop("stage", None)
        self.data_paths = preprocess_paths(config.pop("data_paths", None))
        self.shuffle = config.pop("shuffle", False)
        self.cache = config.pop("cache", False)
        self.drop_remainder = config.pop("drop_remainder", True)
        self.buffer_size = config.pop("buffer_size", 100)
        self.use_tf = config.pop("use_tf", False)
        #self.augmentations = Augmenta



class LearningConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.train_dataset_config = DatasetConfig(config.pop("train_dataset_config", {}))
        self.eval_dataset_config = DatasetConfig(config.pop("eval_dataset_config", {}))
        self.test_dataset_config = DatasetConfig(config.pop("test_dataset_config", {}))
        self.optimizer_config = config.pop("optimizer_config", {})
        self.running_config = RunningConfig(config.pop("running_config", {}))
        for k, v in config.items(): setattr(self, k, v)


class Config:

    def __init__(self, path: str):
        config=load_yaml(preprocess_paths(path))
        self.decoder_config = config.pop("decoder_config", {})
        self.speech_config = config.pop("speech_config", {})
        self.model_config = config.pop("model_config", {})
        self.learning_config = LearningConfig(config.pop("learning_config", {}))
        for k, v in config.items(): setattr(self, k, v)