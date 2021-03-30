

from abc import ABCMeta, abstractmethod

import tensorflow as tf
#import tensorflow_io as tfio
import soundfile as sf
import numpy as np


import librosa
import os
import io

def load_and_convert_to_wav(path: str) -> tf.Tensor:
    wave, rate = librosa.load(os.path.expanduser(path), sr=None, mono=True)
    return tf.audio.encode_wav(tf.expand_dims(wave, axis=-1), sample_rate=rate)

def read_raw_audio(audio, sample_rate=16000):
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate, mono=True)

    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        if wave.ndim > 1: wave = np.mean(wave, axis=-1)
        wave = np.asfortranarray(wave)
        if sr != sample_rate: wave = librosa.resample(wave, sr, sample_rate)

    elif isinstance(audio, np.ndarray):
        if audio.ndim > 1: ValueError("input audio must be single channel")
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")

    return wave

def tf_read_raw_audio(audio: tf.Tensor, sample_rate=16000):
    wave, rate = tf.audio.decode_wav(audio, desired_channels=1, desired_samples=-1)
    # 나중에 하자
    return tf.reshape()

def preemphasis(signal: np.ndarray, coeff=0.97):
    if not coeff or coeff <= 0.00:
        return signal
    return np.append(signal[0], signal[1:] - coeff*signal[:-1])

def normalize_signal(signal: np.ndarray):
    gain = 1.0 / (np.max(np.abs(signal))+ 1e-9)
    return signal*gain

def apply_mvn(audio_feature: np.ndarray, per_feature=False):
    axis = 0 if per_feature else None
    mean = np.mean(audio_feature, axis=axis)
    std_dev = np.std(audio_feature, axis=axis) + 1e-9
    normalized = (audio_feature - mean) / std_dev
    return normalized


class SpeechFeaturizer(metaclass=ABCMeta):
    def __init__(self, speech_config: dict):
        """
        speech_config = {
            "sample_rate": int,
            "frame_ms": int,
            "stride_ms":int,
            "num_feature_bins": int,
            "feature_type": str,
            "delta": bool,
            "delta_delta": bool,
            "pitch": bool,
            "normalize_signal": bool,
            "normalize_feature": bool,
            "normalize_per_feature": bool,
        }
        """

        self.sample_rate = speech_config.get("sample_rate", 16000)
        self.frame_length = int(self.sample_rate * (speech_config.get("frame_ms", 25) / 1000))
        self.frame_step = int(self.sample_rate * (speech_config.get("stride_ms", 10) / 1000))

        self.num_feature_bins = speech_config.get("num_feature_bins", 80)
        self.feature_type = speech_config.get("feature_type", "log_mel_spectrogram")
        self.preemphasis = speech_config.get("preemphasis", None)

        self.normalize_signal = speech_config.get("normalize_signal", True)
        self.normalize_feature = speech_config.get("normalize_feature", True)
        self.normalize_per_feature = speech_config.get("normalize_per_feature", False)

    @property
    def nfft(self) -> int:
        return 2**(self.frame_length - 1).bit_length()

    @property
    def shape(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def stft(self, signal):
        raise NotImplementedError()

    @abstractmethod
    def power_to_db(self, S, ref=1.0, amin=1e-10, top_db=80.0):
        raise NotImplementedError()

    @abstractmethod
    def extract(self, signal):
        raise NotImplementedError()

class NumpySpeechFeaturizer(SpeechFeaturizer):
    def __init__(self, speech_config: dict):
        super(NumpySpeechFeaturizer, self).__init__(speech_config)
        self.delta = speech_config.get("delta", False)
        self.delta_delta = speech_config.get("delta_delta", False)
        self.pitch = speech_config.get("pitch", False)

    @property
    def shape(self) -> list:
        channel_dim = 1

        if self.delta:
            channel_dim+=1

        if self.delta_delta:
            channel_dim+=1

        if self.pitch:
            channel_dim+=1

        return [None, self.num_feature_bins, channel_dim]

    def stft(self, signal):
        return np.square(np.abs(librosa.core.stft(signal, n_fft=self.nfft, hop_length=self.frame_step, win_length=self.frame_length, center=False, window="hann")))

    def power_to_db(self, S, ref=1.0, amin=1e-10, top_db=80.0):
        return librosa.power_to_db(S, ref=ref, amin=amin, top_db=top_db)

    def extract(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asfortranarray(signal)
        if self.normalize_signal:
            signal = normalize_signal(signal)
        signal = preemphasis(signal, self.preemphasis)

        if self.feature_type == "mfcc":
            features = self.compute_mfcc(signal)
        elif self.feature_type == "log_mel_spectrogram":
            features = self.compute_log_mel_spectrogram(signal)
        elif self.feature_type == "log_mel_spectrogram":
            features = self.compute_spectrogram(signal)
        else:
            raise ValueError("mfcc or log_mel_spectrogram")

        original_features = features.copy()

        # apply mvn
        if self.normalize_feature:
            features = apply_mvn(features, per_feature=self.normalize_per_feature)

        features = np.expand_dims(features, axis=-1)

        if self.delta:
            delta = librosa.feature.delta(original_features.T).T
            if self.normalize_feature:
                delta=apply_mvn(delta, per_feature=self.normalize_per_feature)
            features=np.concatenate([features, np.expand_dims(delta, axis=-1)], axis=-1)

        if self.delta_delta:
            delta_delta = librosa.feature.delta(original_features.T, order=2).T
            if self.normalize_per_feature:
                delta_delta = apply_mvn(delta_delta, per_feature=self.normalize_per_feature)

            features = np.concatenate([features, np.expand_dims(delta_delta, axis=-1)], axis=-1)

        if self.pitch:
            pitches = self.compute_pitch(signal)
            if self.normalize_feature:
                pitches = apply_mvn(pitches, self.normalize_per_feature)
            features = np.concatenate([features, np.expand_dims(pitches, axis=-1)], axis=-1)

        return features


    def compute_pitch(self, signal: np.ndarray) -> np.ndarray:
        pitches, _ = librosa.core.piptrack(
            y=signal, sr=self.sample_rate,
            n_fft = self.nfft, hop_length=self.frame_step, fmin=0.0, fmax=int(self.sample_rate / 2), win_length=self.frame_length, center=False
        )


    def compute_mfcc(self, signal: np.ndarray) -> np.ndarray:
        S = self.stft(signal)

        mel = librosa.filters.mel(self.sample_rate, self.nfft, n_mels=self.num_feature_bins, fmin=0.0, fmax=int(self.sample_rate/2))

        mel_spectrogram = np.dot(S.T, mel.T)

        mfcc = librosa.feature.mfcc(sr=self.sample_rate, S=self.power_to_db(mel_spectrogram).T, n_mfcc=self.num_feature_bins)

        return mfcc.T

    def compute_log_mel_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        S = self.stft(signal)

        mel = librosa.filters.mel(self.sample_rate, self.nfft, n_mels=self.num_feature_bins, fmin=0.0, fmax=int(self.sample_rate/2))

        mel_spectrogram = np.dot(S.T, mel.T)

        return self.power_to_db(mel_spectrogram)


    def compute_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        powspec = self.stft(signal)
        features = self.power_to_db(powspec.T)

        assert self.num_feature_bins <= self.frame_length // 2 + 1, \
            "num_features for spectrogram should \
        be <= (sample_rate * window_size // 2 + 1)"

        # cut high frequency part, keep num_feature_bins features
        features = features[:, :self.num_feature_bins]

        return features

