
from abc import ABCMeta, abstractmethod
from ..configs.config import DecoderConfig

import unicodedata
import tensorflow as tf
import codecs
import numpy as np
import os

import tensorflow_datasets as tds

ENGLISH_CHARACTERS = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                      "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

KOREAN_CHARACTERS = [" ","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄸ","ㄹ","ㄺ","ㄻ","ㄼ","ㄾ","ㅀ","ㅁ","ㅂ",
                     "ㅃ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ","ㅏ","ㅐ","ㅑ","ㅒ",
                     "ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]

"""
PHONE_CHARACTERS = [" ", "/AA", "/AA0", "/AA1/", "/AA2/", "/AE/", "/AE0/", "/AE1/", "/AE2/", "/AH/",
                       "/AH0", "/AH1", "/AH2/", "/AO/", "/AO0/", "/AO1/", "/AO2/", "/AW/", "/AW0/", "/AW1/",
                        "/AW2", "/AY", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH",
                        "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY", "EY0", "EY1",
                        "EY2", "F", "G", "HH", "IH", "IH0", "IH1", "IH2", "IY", "IY0",
                        "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW", "OW0", "OW1",
                        "OW2", "OY", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T",
                        "TH", "UH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V",
                        "W", "Y", "Z", "ZH"]
"""


PHONE_CHARACTERS = [ 'p0', 'ph', 'pp', 't0', 'th', 'tt', 'k0', 'kh', 'kk', 's0', 'ss',
                     'h0', 'c0', 'ch', 'cc', 'mm', 'nn', 'rr', 'pf', 'tf', 'kf', 'mf',
                     'nf', 'ng', 'll', 'ks', 'nc', 'nh', 'lk', 'lm', 'lb', 'ls', 'lt',
                     'lp', 'lh', 'ps', 'ii', 'ee', 'qq', 'aa', 'xx', 'vv', 'uu', 'oo',
                     'ye', 'yq', 'ya', 'yv', 'yu', 'yo', 'wi', 'wo', 'wq', 'we', 'wa',
                     'wv', 'xi', '@@']

LABEL_AUDIOS = ["baby","bicycle","boiling","car","carpassing", "clock","dog",
                "door","fire","glass","jackhammer","kettle","scream","siren",
                "speech","unknown","whistle"]



class TextFeaturizer(metaclass=ABCMeta):
    def __init__(self, decoder_config: dict):
        self.scorer = None
        self.decoder_config = DecoderConfig(decoder_config)
        self.blank = None
        self.tokens2indices = {}
        self.tokens = []
        self.num_classes = None

    @property
    def shape(self) -> list:
        return [self.max_length if self.max_length >0 else None]

    @property
    def prepand_shape(self) -> list:
        return [self.max_length+1 if self.max_length >0 else None]

    def preprocess_text(self, text):
        text = unicodedata.normalize("NFC", text.lower())
        return text.strip("\n")

    def update_length(self, length: int):
        self.max_length = max(self.max_length, length)

    def reset_length(self):
        self.max_length = 0

    def add_scorer(self, scorer: any = None):
        self.scorer = scorer

    def normalize_indices(self, indices: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("normalize_indices"):
            minus_one = -1*tf.ones_like(indices, dtype=tf.int32)
            blank_like = self.blank * tf.ones_like(indices, dtype=tf.int32)
            return tf.where(indices == minus_one, blank_like, indices)

    def prepand_blank(self, text:tf.Tensor) -> tf.Tensor:
        return tf.concat([[self.blank], text], axis=0)

    @abstractmethod
    def extract(self, text):
        raise NotImplementedError()

    @abstractmethod
    def iextract(self, indices):
        raise NotImplementedError()

    @abstractmethod
    def indice2upoints(self, indices):
        raise NotImplementedError()



class CharFeaturizer(TextFeaturizer):
    def __init__(self, decoder_config: dict):

        super(CharFeaturizer, self).__init__(decoder_config)
        self.__init_vocabulary()

    def __init_vocabulary(self):
        lines = []
        if self.decoder_config.vocabulary is not None:
            with codecs.open(self.decoder_config.vocabulary, "r", "utf-8") as fin:
                lines.extend(fin.readlines())
        else:
            lines = ENGLISH_CHARACTERS
            #lines = KOREAN_CHARACTERS

        self.blank = 0 if self.decoder_config.blank_at_zero else None
        self.tokens2indices = {}
        self.tokens = []
        index = 1 if self.blank == 0 else 0
        for line in lines:
            line = self.preprocess_text(line)
            if line.startswith("#") or not line: continue
            self.tokens2indices[line[0]] = index
            self.tokens.append(line[0])
            index += 1
        if self.blank is None: self.blank = len(self.tokens)
        self.vocab_array = self.tokens.copy()
        self.tokens.insert(self.blank, "")
        self.num_classes = len(self.tokens)
        self.tokens = tf.convert_to_tensor(self.tokens, dtype=tf.string)
        self.upoints = tf.strings.unicode_decode(self.tokens, "UTF-8").to_tensor(shape=[None, 1])

    def extract(self, text: str) -> tf.Tensor:
        text = self.preprocess_text(text)
        text = list(text.strip())
        indices = [self.tokens2indices[token] for token in text]
        return tf.convert_to_tensor(indices, dtype=tf.int32)

    def iextract(self, indices: tf.Tensor) -> tf.Tensor:
        indices = self.normalize_indices(indices)
        token = tf.gather_nd(self.tokens, tf.expand_dims(indices, axis = -1))
        with tf.device("/CPU:0"):
            tokens = tf.strings.reduce_join(token, axis=-1)
        return tokens

    @tf.function(
        input_signature=[
            tf.TensorSpec([None], dtype=tf.int32)
        ]
    )
    def indice2upoints(self, indices: tf.Tensor) -> tf.Tensor:

        with tf.name_scope("indices2upoints"):
            indices = self.normalize_indices(indices)
            upoints = tf.gather_nd(self.upoints, tf.expand_dims(indices, axis=-1))
            return tf.gather_nd(upoints, tf.where(tf.not_equal(upoints, 0)))