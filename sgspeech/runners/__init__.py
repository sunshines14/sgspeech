import os
import tensorflow as tf

def save_from_checkpoint(func, outdir: dir,
                         max_to_keep: int = 10,
                         **kwargs):
    steps = tf.Variable(0, trainable=False, dtype=tf.int64)
    epochs = tf.Variable(1, trainable=False)

    checkpoint_dir = os.path.join(outdir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"checkpoint directory not found: {checkpoint_dir}")

    ckpt = tf.train.Checkpoint(steps=steps, epochs=epochs, **kwargs)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=max_to_keep)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        func(**kwargs)
    else:
        raise ValueError("no latest checkpoint found")
