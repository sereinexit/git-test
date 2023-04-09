import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

Regularizer = {
    "Adam": tf.train.AdamOptimizer,
    "RMSProp": tf.train.RMSPropOptimizer,
    "SGD": tf.train.GradientDescentOptimizer,
    # "Momentum": tf.train.MomentumOptimizer,
    # "LAdam": tf.contrib.opt.LazyAdamOptimizer,
}