import tensorflow as tf

class ForwardActor(tf.keras.Model):
    def __init__(self, out_dim=3):
        super(ForwardActor, self).__init__()
        self.conv2d1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation=tf.nn.relu, padding='same', kernel_initializer=tf.keras.initializers.VarianceScaling(2.0))
        self.conv2d2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, padding='same', kernel_initializer=tf.keras.initializers.VarianceScaling(2.0))
        self.conv2d3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation=tf.nn.relu, padding='same', kernel_initializer=tf.keras.initializers.VarianceScaling(2.0))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(out_dim, activation='tanh', name='pred_a')

    def call(self, state):
        x = tf.cast(state, tf.float32)/255.0
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        action = self.dense2(x)
        return action

class Actor(object):
    def __init__(self, out_dim=3, name='online'):
        self.name = name
        with tf.name_scope(f'{self.name}/actor'):
            self.forward_fn = ForwardActor(out_dim)

    def __call__(self, state):
        with tf.name_scope(f'{self.name}/actor/'):
            pred_a = self.forward_fn(state)
        return pred_a

    def trainable_vars(self):
        return tf.trainable_variables(scope=f'{self.name}/actor')

    def global_vars(self):
        return tf.global_variables(scope=f'{self.name}/actor')
