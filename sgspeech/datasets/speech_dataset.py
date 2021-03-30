from .base_dataset import BaseDataset, BUFFER_SIZE, AUTOTUNE

import tensorflow as tf
import numpy as np

from ..featurizers.speech_featurizer import load_and_convert_to_wav, read_raw_audio, SpeechFeaturizer
from ..featurizers.text_featurizer import TextFeaturizer
from ..utils.utils import get_num_batches
from ..augmentation.augments import Augmentation


class SpeechDataset(BaseDataset):

    # speech, text feature extractor 추가해야됨
    def __init__(self,
                 stage: str,
                 speech_featurizer: SpeechFeaturizer,
                 text_featurizer: TextFeaturizer,
                 data_paths: list,
                 augmentations: Augmentation = Augmentation(None),
                 cache: bool = False,
                 shuffle: bool= False,
                 indefinite: bool = False,
                 drop_remainder: bool = True,
                 use_tf: bool = False,
                 buffer_size: int = BUFFER_SIZE,
                 **kwargs):
        super(SpeechDataset, self).__init__(
            data_paths=data_paths, cache=cache, augmentations=augmentations, shuffle=shuffle, buffer_size=buffer_size, drop_remainder=drop_remainder,
            use_tf=use_tf)

        # speech, text feature extractor 추가해야됨
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    def read_entries(self):

        self.entries = []

        for file_path in self.data_paths:
            print("Read files")
            # 설명 https://stackoverflow.com/questions/42256938/what-does-tf-gfile-do-in-tensorflow
            with tf.io.gfile.GFile(file_path, "r") as f:
                temp_lines = f.read().splitlines()
                # 맨 위에 헤더 제거
                self.entries += temp_lines[1:]

            self.entries = [line.split("\t", 2) for line in self.entries]

            for i, line in enumerate(self.entries):
                #tf.print("Speech dataset part")
                #tf.print(line)
                #tf.print(line[-1])
                self.entries[i][-1] = " ".join([str(x) for x in self.text_featurizer.extract(line[-1]).numpy()])
            self.entries = np.array(self.entries)
            #tf.print(self.entries)
            if self.shuffle: np.random.shuffle(self.entries)  # shuffle
            self.total_steps = len(self.entries)

    def generator(self):
        for path, _, indices in self.entries:
            audio = load_and_convert_to_wav(path).numpy()  # speech feature 구현 필요
            yield bytes(path, "utf-8"), audio, bytes(indices, "utf-8")

    def preprocess(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):
        with tf.device("/CPU:0"):
            def fn(_path: bytes, _audio: bytes, _indices: bytes):
                signal = read_raw_audio(_audio, sample_rate=self.speech_featurizer.sample_rate)

                signal = self.augmentations.before.augment(signal)
                features = self.speech_featurizer.extract(signal)

                features = self.augmentations.after.augment(features)

                label = tf.strings.to_number(tf.strings.split(_indices), out_type=tf.int32)
                label_len = tf.cast(tf.shape(label)[0], tf.int32)
                prediction = self.text_featurizer.prepand_blank(label)
                prediction_len = tf.cast(tf.shape(prediction)[0], tf.int32)
                features = tf.convert_to_tensor(features, tf.float32)
                input_len = tf.cast(tf.shape(features)[0], tf.int32)

                return _path, features, input_len, label, label_len, prediction, prediction_len

            return tf.numpy_function(
                fn, inp=[path, audio, indices],
                Tout=[tf.string, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]
            )

    # def tf_preprocess(self, path: tf.tensor, audio: tf.tensor, indices: tf.tensor):
    #     with tf.device("/cpu:0"):
    #         signal = tf_read_raw_audio(audio, self.speech_featurizer.sample_rate)
    #         # aumentation 추가해야 함
    #         features = self.speech_featurizer.tf_extract(signal)
    #         # aumentation 추가해야 함
    #
    #         label = tf.strings.to_number(tf.strings.split(indices), out_type=tf.int32)
    #         label_len = tf.cast(tf.shape(label)[0], tf.int32)
    #         prediction = self.text_featurizer.prepand_blank(label)
    #         prediction_len = tf.cast(tf.shape(prediction)[0], tf.int32)
    #         features = tf.convert_to_tensor(features, tf.float32)
    #         input_len = tf.cast(tf.shape(features)[0], tf.int32)
    #
    #         return path, features, input_len, label, label_len, prediction, prediction_len

    @tf.function
    def parse(self, path: tf.Tensor, audio: tf.Tensor, indices: tf.Tensor):

        #if self.use_tf: return self.tf_preprocess(path, audio, indices)
        return self.preprocess(path, audio, indices)

    def process(self, dataset: tf.data.Dataset, batch_size: int):
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(self.buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                tf.TensorShape([]),
                tf.TensorShape(self.speech_featurizer.shape),
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([]),

            ),
            padding_values=("", 0., 0, self.text_featurizer.blank, 0, self.text_featurizer.blank, 0),
            drop_remainder=self.drop_remainder
        )

        dataset = dataset.prefetch(AUTOTUNE)
        self.total_steps = get_num_batches(self.total_steps, batch_size)
        return dataset

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return None
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types=(tf.string, tf.string, tf.string),
            output_shapes=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))
        )
        return self.process(dataset, batch_size)


class SpeechSliceDataset(SpeechDataset):

    @staticmethod
    def load(record: tf.Tensor):
        def fn(path: bytes): return load_and_convert_to_wav(path.decode("utf-8")).numpy()
        audio = tf.numpy_function(fn, inp=[record[0]], Tout=tf.string)
        return record[0], audio, record[2]

    def create(self, batch_size: int):
        self.read_entries()
        if not self.total_steps or self.total_steps == 0: return None
        dataset = tf.data.Dataset.from_tensor_slices(self.entries)
        dataset = dataset.map(self.load, num_parallel_calls=AUTOTUNE)
        return self.process(dataset, batch_size)
