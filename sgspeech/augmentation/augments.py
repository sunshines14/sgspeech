import tensorflow as tf
import nlpaug.flow as naf

from .signal_augment import SignalVtlp, SignalSpeed, SignalPitch, SignalNoise, SignalMask, SignalCropping, SignalLoudness, SignalShift
from .spec_augment import FreqMasking, TimeMasking

AUGMENTATIONS = {
    "freq_masking": FreqMasking,
    "time_masking": TimeMasking,
    "noise": SignalNoise,
    "masking": SignalMask,
    "cropping": SignalCropping,
    "loudness": SignalLoudness,
    "pitch": SignalPitch,
    "shift": SignalShift,
    "speed": SignalSpeed,
    "vtlp": SignalVtlp
}

class Augmentation:
    def __init__(self, config:dict = None, use_tf: bool = False):
        if not config: config = {}
        self.before = self.parse(config.pop("before", {}))
        self.after = self.parse(config.pop("after", {}))

    @staticmethod
    def parse(config: dict) -> list:
        augmentations = []
        for key, value in config.items():
            au = AUGMENTATIONS.get(key, None)
            if au is None:
                raise KeyError(f"No augmentation named: {key}\n"
                               f"Available augmentations: {AUGMENTATIONS.keys()}")
            aug = au(**value) if value is not None else au()
            augmentations.append(aug)
        return naf.Sometimes(augmentations)
