from abc import ABCMeta, abstractmethod
import tensorflow as tf

BUFFER_SIZE = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE

from ..augmentation.augments import Augmentation

class BaseDataset(metaclass=ABCMeta):
    def __init__(self,
                 data_paths: list,
                 augmentations: Augmentation = Augmentation(None),
                 cache: bool = False,
                 shuffle: bool = False,
                 buffer_size: int = BUFFER_SIZE,
                 drop_remainder: bool = True,
                 use_tf: bool = False,
                 stage: str = "train",
                 **kwargs):
        self.data_paths = data_paths
        self.augmentations = augmentations
        self.cache = cache
        self.shuffle = shuffle
        if buffer_size <= 0 and shuffle: raise ValueError("buffer_size must be positive when shuffle is on")
        self.buffer_size = buffer_size
        self.stage = stage
        self.use_tf = use_tf
        self.drop_remainder = drop_remainder
        self.total_steps = None

        @abstractmethod
        def parse(self, *args, **kargs):
            raise NotImplementedError()

        @abstractmethod
        def create(self, batch_size):
            raise NotImplementedError()
