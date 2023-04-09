import tensorflow as tf

Regularizer = {
    "Adam": tf.compat.v1.train.AdamOptimizer,
    "RMSProp": tf.compat.v1.train.RMSPropOptimizer,
    "SGD": tf.compat.v1.train.GradientDescentOptimizer,#tf.train.GradientDescentOptimizer,
    # "Momentum": tf.train.MomentumOptimizer,
    # "LAdam": tf.contrib.opt.LazyAdamOptimizer,
}