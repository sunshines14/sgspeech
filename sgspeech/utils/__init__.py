

def setup_environment():  # Set memory growth and only log ERRORs
    """ Setting tensorflow running environment """
    import warnings

    warnings.simplefilter("ignore")

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})


def setup_devices(devices, cpu=False):
    """Setting visible devices
    Args:
        devices (list): list of visible devices' indices
    """
    import tensorflow as tf

    if cpu:
        cpus = tf.config.list_physical_devices("CPU")
        tf.config.set_visible_devices(cpus, "CPU")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            visible_gpus = [gpus[i] for i in devices]
            tf.config.set_visible_devices(visible_gpus, "GPU")
            print("Run on", len(visible_gpus), "Physical GPUs")


def setup_strategy(devices):
    """Setting mirrored strategy for training
    Args:
        devices (list): list of visible devices' indices
    Returns:
        tf.distribute.Strategy: MirroredStrategy for training one or multiple gpus
    """
    import tensorflow as tf

    setup_devices(devices)

    return tf.distribute.MirroredStrategy()
