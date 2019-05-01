import tensorflow as tf
import tensorflow.contrib as tc
from actor import Actor
from critic import Critic
from config import CONFIG as C


class DDPG(object):
    def __init__(self, state_dim=(84, 84, 4), action_dim=3):
        self.s_ph = tf.placeholder(tf.uint8, shape=(None,) + (state_dim), name='s_ph')
        self.a_ph = tf.placeholder(tf.float32, shape=(None, action_dim), name='a_ph')
        self.r_ph = tf.placeholder(tf.float32, shape=(None, 1), name='r_ph')
        self.d_ph = tf.placeholder(tf.float32, shape=(None, 1), name='d_ph')
        self.ns_ph = tf.placeholder(tf.uint8, shape=(None,) + (state_dim), name='ns_ph')

        self.online_actor = Actor(action_dim, 'online')
        self.online_critic = Critic(1, 'online')
        self.target_actor = Actor(action_dim, 'target')
        self.target_critic = Critic(1, 'target')

        self.pred_a = self.online_actor(self.s_ph)
        self.pred_Q = self.online_critic(self.s_ph, self.a_ph)
        self.pred_Q_with_actor = self.online_critic(self.s_ph, self.pred_a)
        self.pred_next_a = self.target_actor(self.ns_ph)
        self.pred_next_Q = self.target_critic(self.ns_ph, self.pred_next_a)

        # Set up update target networks
        self.init_ops, self.update_ops = self._target_updates(C.tau)

        # Set up actor optimization
        actor_optimizer = tf.train.AdamOptimizer(0.0001)
        with tf.name_scope('actor_loss'):
            actor_loss = -tf.reduce_mean(self.pred_Q_with_actor)
            actor_grads = tf.gradients(actor_loss, self.online_actor.trainable_vars())
        if C.clip_norm is not None:
            with tf.name_scope('grads_clip'):
                with tf.name_scope('actor'):
                    actor_grads = [tf.clip_by_norm(grad, C.clip_norm) for grad in actor_grads]
            assert len(actor_grads) == len(self.online_actor.trainable_vars())
        self.actor_optimize_op = actor_optimizer.apply_gradients(zip(actor_grads, self.online_actor.trainable_vars()))

        # Set up critic optimization
        critic_optimizer = tf.train.AdamOptimizer(0.001)
        with tf.name_scope('critic_loss'):
            target_Q = self.r_ph + (1.0 - self.d_ph)*C.discount_factor*self.pred_next_Q
            critic_loss = tf.reduce_mean(tf.square(self.pred_Q - tf.stop_gradient(target_Q)))
            if C.critic_l2_reg > 0.0:
                critic_reg_vars = [v for v in self.online_critic.trainable_vars() if 'pred_Q' not in v.name]
                critic_reg = tc.layers.apply_regularization(tc.layers.l2_regularizer(C.critic_l2_reg), weights_list=critic_reg_vars)
                critic_loss += critic_reg
            critic_grads = tf.gradients(critic_loss, self.online_critic.trainable_vars())
        if C.clip_norm is not None:
            with tf.name_scope('grads_clip/'):
                with tf.name_scope('critic'):
                    critic_grads = [tf.clip_by_norm(grad, C.clip_norm) for grad in critic_grads]
            assert len(critic_grads) == len(self.online_critic.trainable_vars())
        self.critic_optimize_op = critic_optimizer.apply_gradients(zip(critic_grads, self.online_critic.trainable_vars()))

    def _target_updates(self, tau):
        online_actor_vars = self.online_actor.global_vars()
        online_critic_vars = self.online_critic.global_vars()
        target_actor_vars = self.target_actor.global_vars()
        target_critic_vars = self.target_critic.global_vars()

        assert len(online_actor_vars) == len(target_actor_vars)
        assert len(online_critic_vars) == len(target_critic_vars)

        soft_ops = []
        init_ops = []
        with tf.name_scope('updates'):
            with tf.name_scope('actor'):
                for o, t in zip(online_actor_vars, target_actor_vars):
                    init_ops.append(t.assign(o))
                    soft_ops.append(t.assign((1.0 - tau)*t + tau*o))
            with tf.name_scope('critic'):
                for o, t in zip(online_critic_vars, target_critic_vars):
                    init_ops.append(t.assign(o))
                    soft_ops.append(t.assign((1.0 - tau)*t + tau*o))
            with tf.name_scope('group_ops'):
                io = tf.group(*init_ops)
                so = tf.group(*soft_ops)
        assert len(init_ops) == len(soft_ops)
        return io, so

    def initialize_target(self):
        tf.get_default_session().run(self.init_ops)

    def update_target(self):
        tf.get_default_session().run(self.update_ops)

    def action(self, s):
        if s.ndim == 3:
            s = [s]
        return tf.get_default_session().run(self.pred_a, feed_dict={self.s_ph: s})

    def train_critic(self, sb, ab, rb, db, nsb):
        tf.get_default_session().run(self.critic_optimize_op, feed_dict={self.s_ph: sb, self.a_ph: ab, self.r_ph: rb, self.d_ph: db, self.ns_ph: nsb})

    def train_actor(self, sb):
        tf.get_default_session().run(self.actor_optimize_op, feed_dict={self.s_ph: sb})

    def train(self, sb, ab, rb, db, nsb):
        tf.get_default_session().run([self.critic_optimize_op, self.actor_optimize_op], feed_dict={self.s_ph: sb, self.a_ph: ab, self.r_ph: rb, self.d_ph: db, self.ns_ph: nsb})

    def save(self, directory, i):
        saver = tf.train.Saver()
        d = saver.save(tf.get_default_session(), f'{directory}/model_{i}.ckpt')
        return d

    def load(self, directory, i):
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), f'{directory}/model_{i}.ckpt')


if __name__ == '__main__':
    import os
    import numpy as np
    os.system('rm -rf test/')  # Clean up

    sess = tf.InteractiveSession()
    ddpg = DDPG()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./test', sess.graph)

    sb = (np.random.rand(32, 84, 84, 4)*255).astype('uint8')
    ab = np.random.rand(32, 3)
    rb = np.random.randn(32, 1)
    db = np.random.randint(2, size=(32, 1))
    nsb = (np.random.rand(32, 84, 84, 4)*255).astype('uint8')

    ddpg.initialize_target()
    for o, t in zip(ddpg.online_actor.global_vars(), ddpg.target_actor.global_vars()):
        assert np.allclose(o.eval(), t.eval())
    for o, t in zip(ddpg.online_critic.global_vars(), ddpg.target_critic.global_vars()):
        assert np.allclose(o.eval(), t.eval())

    print(ddpg.action(sb[0]))
    print(ddpg.action(sb[:3]))

    ddpg.train_critic(sb, ab, rb, db, nsb)
    t0a = []
    t0c = []
    for o, t in zip(ddpg.online_actor.trainable_vars(), ddpg.target_actor.trainable_vars()):
        t0a.append(t.eval())
        assert np.allclose(o.eval(), t.eval())
    for o, t in zip(ddpg.online_critic.trainable_vars(), ddpg.target_critic.trainable_vars()):
        t0c.append(t.eval())
        assert not np.allclose(o.eval(), t.eval())

    ddpg.train_actor(sb)
    for o, t in zip(ddpg.online_actor.trainable_vars(), ddpg.target_actor.trainable_vars()):
        assert not np.allclose(o.eval(), t.eval())
    # for o, t in zip(ddpg.online_critic.trainable_vars(), ddpg.target_critic.trainable_vars()):
    #     assert not np.allclose(o.eval(), t.eval())

    ddpg.update_target()
    for i, (o, t) in enumerate(zip(ddpg.online_actor.trainable_vars(), ddpg.target_actor.trainable_vars())):
        assert np.allclose(t0a[i], (t.eval() - C.tau*o.eval())/(1 - C.tau))
        assert not np.allclose(o.eval(), t.eval())
    for i, (o, t) in enumerate(zip(ddpg.online_critic.trainable_vars(), ddpg.target_critic.trainable_vars())):
        assert np.allclose(t0c[i], (t.eval() - C.tau*o.eval())/(1 - C.tau))
        assert not np.allclose(o.eval(), t.eval())
