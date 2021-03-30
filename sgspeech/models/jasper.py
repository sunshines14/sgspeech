import tensorflow as tf

from ..utils.utils import merge_two_last_dims
from .ctc import CTCModel


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs):
        return merge_two_last_dims(inputs)


class JasperSubBlock(tf.keras.layers.Layer):
    def __init__(self,
                 channels: int = 256,
                 kernels: int = 11,
                 strides: int = 1,
                 dropout: float = 0.1,
                 dilation: int = 1,
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 **kwargs):

        super(JasperSubBlock, self).__init__(**kwargs)

        self.conv1d = tf.keras.layers.Conv1D(
            filters=channels, kernel_size=kernels,
            strides=strides, dilation_rate=dilation, padding="same",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_conv1d"
        )

        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")
        self.relu = tf.keras.layers.ReLU(name=f"{self.name}_relu")
        self.do = tf.keras.layers.Dropout(dropout, name=f"{self.name}_dropout")
        self.reduction_factor = strides

    def call(self, inputs, training=False, **kwargs):
        outputs = inputs
        outputs = self.conv1d(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(JasperSubBlock, self).get_config()
        conf.update(self.conv1d.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.relu.get_config())
        conf.update(self.do.get_config())
        return conf


class JasperResidual(tf.keras.layers.Layer):
    def __init__(self,
                 channels: int = 256,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(JasperResidual, self).__init__(**kwargs)
        self.pointwise_conv1d = tf.keras.layers.Conv1D(
            filters=channels, kernel_size=1,
            strides=1, padding="same",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_pointwise_conv1d"
        )
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.pointwise_conv1d(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(JasperResidual, self).get_config()
        conf.update(self.pointwise_conv1d.get_config())
        conf.update(self.bn.get_config())
        return conf


class JasperSubBlockResidual(JasperSubBlock):
    def __init__(self,
                 channels: int = 256,
                 kernels: int = 11,
                 strides: int = 1,
                 dropout: float = 0.1,
                 dilation: int = 1,
                 nresiduals: int = 1,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(JasperSubBlockResidual, self).__init__(
            channels=channels, kernels=kernels,
            strides=strides, dropout=dropout,
            dilation=dilation, kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, **kwargs
        )

        self.residuals = [
            JasperResidual(
                channels=channels,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_residual_{i}"
            ) for i in range(nresiduals)
        ]

        self.add = tf.keras.layers.Add(name=f"{self.name}_add")

    def call(self, inputs, training=False, **kwargs):
        outputs, residuals = inputs
        outputs = self.conv1d(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        for i, res in enumerate(residuals):
            res = self.residuals[i](res, training=training, **kwargs)
            outputs = self.add([outputs, res], training=training)
        outputs = self.relu(outputs, training=training)
        outputs = self.do(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(JasperSubBlockResidual, self).get_config()
        conf.update(self.residual.get_config())
        conf.update(self.add.get_config)
        return conf

class JasperBlock(tf.keras.Model):
    def __init__(self,
                 nsubblocks: int =3,
                 channels: int = 256,
                 kernels: int = 11,
                 dropout: float = 0.1,
                 dense: bool = False,
                 nresiduals: int = 1,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(JasperBlock, self).__init__(**kwargs)

        self.dense = dense

        self.subblocks = [
            JasperSubBlock(
                channels=channels,
                kernels=kernels,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_subordinate_{i}"
            ) for i in range(nsubblocks - 1)
        ]

        self.subblock_residual = JasperSubBlockResidual(
            channels=channels,
            kernels=kernels,
            dropout=dropout,
            nresiduals=nresiduals,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_subordinate_{nsubblocks - 1}"
        )

        self.reduction_factor = 1

    def call(self, inputs, training=False, **kwargs):
        inputs, residuals = inputs
        outputs = inputs
        for subblock in self.subblocks:
            outputs = subblock(outputs, training=training, **kwargs)
        if self.dense:
            residuals.append(inputs)
            outputs = self.subblock_residual([outputs, residuals], training=training, **kwargs)
        else:
            outputs = self.subblock_residual([outputs, [inputs]], training=training, **kwargs)
        return outputs, residuals

    def get_config(self):
        conf = self.subblock_residual.get_config()
        conf.update({"dense": self.dense})
        for subblock in self.subblocks:
            conf.update(subblock.get_config())
        return conf


class Jasper(CTCModel):
    def __init__(self,
                 vocabulary_size: int,
                 dense: bool = False,
                 first_additional_block_channels: int = 256,
                 first_additional_block_kernels: int = 11,
                 first_additional_block_strides: int = 2,
                 first_additional_block_dilation: int = 1,
                 first_additional_block_dropout: int = 0.2,
                 nsubblocks: int = 5,
                 block_channels: list = [256, 384, 512, 640, 768],
                 block_kernels: list = [11, 13, 17, 21, 25],
                 block_dropout: list = [0.2, 0.2, 0.2, 0.3, 0.3],
                 second_additional_block_channels: int = 896,
                 second_additional_block_kernels: int = 1,
                 second_additional_block_strides: int = 1,
                 second_additional_block_dilation: int = 2,
                 second_additional_block_dropout: int = 0.4,
                 third_additional_block_channels: int = 1024,
                 third_additional_block_kernels: int = 1,
                 third_additional_block_strides: int = 1,
                 third_additional_block_dilation: int = 1,
                 third_additional_block_dropout: int = 0.4,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name: str = "jasper",
                 **kwargs):
        super(Jasper, self).__init__(name=name, **kwargs)

        assert len(block_channels) == len(block_kernels) == len(block_dropout)

        self.reshape = Reshape(name=f"{self.name}_reshape")

        self.first_additional_block = JasperSubBlock(
            channels=first_additional_block_channels,
            kernels=first_additional_block_kernels,
            strides=first_additional_block_strides,
            dropout=first_additional_block_dropout,
            dilation=first_additional_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_first_block"
        )

        self.blocks = [
            JasperBlock(
                nsubblocks=nsubblocks,
                channels=block_channels[i],
                kernels=block_kernels[i],
                dropout=block_dropout[i],
                dense=dense,
                nresiduals=(i+1) if dense else 1,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_block_{i}"
            ) for i in range(len(block_channels))
        ]

        self.second_additional_block = JasperSubBlock(
            channels=second_additional_block_channels,
            kernels=second_additional_block_kernels,
            strides=second_additional_block_strides,
            dropout=second_additional_block_dropout,
            dilation=second_additional_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_second_block"
        )

        self.third_additional_block = JasperSubBlock(
            channels=third_additional_block_channels,
            kernels=third_additional_block_kernels,
            strides=third_additional_block_strides,
            dropout=third_additional_block_dropout,
            dilation=third_additional_block_dilation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_third_block"
        )

        self.last_block = tf.keras.layers.Conv1D(
            filters=vocabulary_size, kernel_size=1,
            strides=1, padding="same",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{self.name}_last_block"
        )

        self.time_reduction_factor = self.first_additional_block.reduction_factor
        self.time_reduction_factor *= self.second_additional_block.reduction_factor
        self.time_reduction_factor *= self.third_additional_block.reduction_factor

    def call(self, inputs, training=False, **kwargs):
        outputs = self.reshape(inputs)
        outputs = self.first_additional_block(outputs, training=training, **kwargs)

        residuals = []
        for block in self.blocks:
            outputs, residuals = block([outputs, residuals], training=training, **kwargs)

        outputs = self.second_additional_block(outputs, training=training, **kwargs)
        outputs = self.third_additional_block(outputs, training=training, **kwargs)
        outputs = self.last_block(outputs, training=training, **kwargs)
        return outputs

    def summary(self, line_length=100, **kwargs):
        super(Jasper, self).summary(line_length=line_length, **kwargs)

    def get_config(self):
        conf = self.reshape.get_config()
        conf.update(self.first_additional_block.get_config())
        for block in self.blocks:
            conf.update(block.get_config())
        conf.update(self.second_additional_block.get_config())
        conf.update(self.third_additional_block.get_config())
        conf.update(self.last_block.get_config())
        return conf






