import numpy as np
import tensorflow as tf

from nlpaug.flow import Sequential
from nlpaug.util import Action
from nlpaug.model.spectrogram import Spectrogram
from nlpaug.augmenter.spectrogram import SpectrogramAugmenter

from ..utils.utils import shape_list

class FreqMaskingModel(Spectrogram):
    def __init__(self, mask_factor: int = 27):
        super(FreqMaskingModel, self).__init__()
        self.mask_factor = mask_factor

    def mask(self, data: np.ndarray) -> np.ndarray:
        spectrogram = data.copy()
        freq = np.random.randint(0, self.mask_factor + 1)
        freq = min(freq, spectrogram.shape[1])
        freq0 = np.random.randint(0, spectrogram.shape[1] - freq +1)
        spectrogram[:, freq0:freq, :] = 0
        return spectrogram


class FreqMaskingAugmenter(SpectrogramAugmenter):
    def __init__(self, mask_factor: float = 27, name: str="FreqMaskingAugmenter", verbose=0):
        super(FreqMaskingAugmenter, self).__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., factor=(40, 80), silence=False, stateless=True
        )
        self.model = FreqMaskingModel(mask_factor)

    def substitute(self, data):
        return self.model.mask(data)

class FreqMasking(SpectrogramAugmenter):
    def __init__(self,
                 num_masks: int = 1,
                 mask_factor: float = 27,
                 name: str = "FreqMasking",
                 verbose = 0):
        super(FreqMasking, self).__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., factor=(40, 80), silence=False, stateless=True
        )
        self.flow=Sequential([FreqMaskingAugmenter(mask_factor) for _ in range(num_masks)])

    def substitute(self, data):
        return self.flow.augment(data)


class TimeMaskingModel(Spectrogram):
    def __init__(self, mask_factor: float = 100, p_upperbound: float = 1.0):

        super(TimeMaskingModel, self).__init__()
        self.mask_factor = mask_factor
        self.p_upperbound = p_upperbound
        assert 0.0 <= self.p_upperbound <= 1.0, "0.0 <=p_upperbound <= 1.0"

    def mask(self, data:np.ndarray) -> np.ndarray:
        spectrogram = data.copy()
        time = np.random.randint(0, self.mask_factor +1)
        time = min(time, int(self.p_upperbound * spectrogram.shape[0]))
        time0 = np.random.randint(0, spectrogram.shape[0] - time +1)
        spectrogram[time0:time0+time, :, :] = 0
        return spectrogram

class TimeMaskingAugmenter(SpectrogramAugmenter):
    def __init__(self,
                 mask_factor: float = 100,
                 p_upperbound: float = 1,
                 name: str = "TimeMaskingAugmenter",
                 verbose = 0):
        super(TimeMaskingAugmenter, self).__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., silence=False, stateless=True
        )
        self.model = TimeMaskingModel(mask_factor, p_upperbound)

    def substitute(self, data):
        return self.model.mask(data)

class TimeMasking(SpectrogramAugmenter):
    def __init__(self,
                 num_masks: int = 1,
                 mask_factor: float = 100,
                 p_upperbound: float = 1,
                 name: str = "TimeMasking",
                 verbose = 0):
        super(TimeMasking, self).__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., silence=False, stateless=True
        )
        self.flow = Sequential([
            TimeMaskingAugmenter(mask_factor, p_upperbound) for _ in range(num_masks)
        ])

    def substitute(self, data):
        return self.flow.augment(data)








