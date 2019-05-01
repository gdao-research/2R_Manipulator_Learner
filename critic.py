import tensorflow as tf


class ForwardCritic(tf.keras.Model):
    def __init__(self, out_dim=1):
        super(ForwardCritic, self).__init__()
        self.conv2d1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.VarianceScaling(2.0))
        self.conv2d2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.VarianceScaling(2.0))
        self.conv2d3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.VarianceScaling(2.0))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense_action = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(out_dim, name='pred_Q')

    def call(self, state, action):
        x = tf.cast(state, tf.float32)/255.0
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        a = self.dense_action(action)
        x = tf.concat([a, x], axis=-1)  # shape = (?, 512)
        x = self.dense2(x)
        x = self.dense3(x)
        Q = self.dense4(x)
        return Q


class Critic(object):
    def __init__(self, out_dim=1, name='online'):
        self.name = name
        with tf.name_scope(f'{self.name}/critic'):
            self.forward_fn = ForwardCritic(out_dim)

    def __call__(self, state, action):
        with tf.name_scope(f'{self.name}/critic/'):
            pred_Q = self.forward_fn(state, action)
        return pred_Q

    def trainable_vars(self):
        return tf.trainable_variables(scope=f'{self.name}/critic')

    def global_vars(self):
        return tf.global_variables(scope=f'{self.name}/critic')
