import numpy as np
import cv2
from manipulator import ManipulatorEnvironment
from threads import ShareSessionThread, LoopThread
from noise import OUNoise

class ScaleEnv(object):
    def __init__(self, env, training=True, img_size=(84, 84), angle_scale=np.pi/18):
        self.env = env
        self.img_size = img_size
        self.is_training = training
        self.angle_scale = angle_scale

    def nb_actions(self):
        return len(self.env.get_links()) + 1

    def reset(self, current_angles=None):
        obs, info = self.env.reset(current_angles)
        # if self.is_training:
        obs = self.resize_obs(obs)
        return obs, info

    def step(self, action):
        if action.ndim == 2:
            action = action[0]
        action *= self.angle_scale
        obs, reward, done, info = self.env.step(action)
        # if self.is_training:
        obs = self.resize_obs(obs)
        return obs, reward, done, info

    def resize_obs(self, obs):
        return cv2.resize(obs, self.img_size)

    def draw_goal(self, goal):
        return self.env.draw_goal(goal)

    def compute_reward(self, end_effector, angle, goal):
        return self.env.compute_reward(end_effector, angle, goal)


def make_env(is_training):
    env = ManipulatorEnvironment()
    env = ScaleEnv(env, is_training)
    return env


class ThreadWorker(ShareSessionThread, LoopThread):
    def __init__(self, fn, queue, is_training):
        super(ThreadWorker, self).__init__()
        self.action_fn = fn
        self.q = queue
        self.is_training = is_training
        self.env = make_env(is_training)
        self.action_noise = OUNoise(mu=np.zeros(self.env.nb_actions()), sigma=0.2*np.ones(self.env.nb_actions()))

    def run(self):
        exp_buffer = []
        with self.default_sess():
            while not self.stopped():
                if not self.paused:
                    try:
                        if self.is_training:
                            exp_b, success = self.play_one_episode()
                            exp_buffer += exp_b
                            exp_buffer = self.hindsight_exp(exp_buffer, k=4)
                            self.queue_put_stopable(self.q, exp_buffer)
                        else:
                            _, success = self.play_one_episode()
                            self.queue_put_stopable(self.q, success)
                    except RuntimeError:
                        return

    def play_one_episode(self):
        buffer = []
        s, _ = self.env.reset()
        success = False
        for i in range(50):
            a = self.action_fn(s)
            if self.is_training:
                a = np.clip(a + self.action_noise(), -1, 1)
            ns, r, d, info = self.env.step(a)
            if r == 0:
                success = True
                self.action_noise.reset()
            buffer.append((s, a, r, d, ns, info))
            s = np.copy(ns)
        return buffer, success

    def hindsight_exp(self, buffer, k):
        g_idx = np.random.choice(np.arange(len(buffer)) - 5, size=k, replace=False) + 5  # 5 - 49
        for idx in g_idx:
            s, _, _, _, _, info = buffer[idx]
            goal_img = np.copy(s[:, :, 2])  # Hardcoded grip as blue channel
            end_effector = np.copy(info['end_points'][-1]).astype('float32')
            angle = sum(info['angles'])
            goal = np.concatenate([end_effector, [angle]], axis=0)
            for i in range(idx+1):
                s, a, r, d, ns, info = buffer[i]
                s[:, :, -1] = goal_img
                ns[:, :, -1] = goal_img
                end_effector = info['end_points'][-1]
                angle = sum(info['angles'])
                r = self.env.compute_reward(end_effector, angle, goal)
                d = True if r == 0 else False
                buffer.append((s, a, r, d, ns, info))
        return buffer


class SingleWorker(object):
    def __init__(self, fn, is_training):
        self.action_fn = fn
        self.is_training = is_training
        self.env = make_env(is_training)
        self.action_noise = OUNoise(mu=np.zeros(self.env.nb_actions()), sigma=0.2*np.ones(self.env.nb_actions()))

    def run(self):
        exp_buffer = []
        exp_buffer, success = self.play_one_episode()
        if self.is_training:
            exp_buffer = self.hindsight_exp(exp_buffer, k=4)
        return exp_buffer, success

    def play_one_episode(self):
        buffer = []
        s, _ = self.env.reset()
        d = False
        success = False
        while not d:
            a = self.action_fn(s)
            if self.is_training:
                a = np.clip(a + self.action_noise(), -1, 1)
            ns, r, d, info = self.env.step(a)
            if d and r == 0:
                success = True
            buffer.append((s, a, r, d, ns, info))
            s = np.copy(ns)
        return buffer, success

    def hindsight_exp(self, buffer, k):
        if len(buffer) > 8:
            g_idx = np.random.choice(np.arange(len(buffer)) - 5, size=k, replace=False) + 5  # 5 - 50
            # g_idx = [5, 10, 15, 20]  # Enable for testing
            for idx in g_idx:
                s, _, _, _, _, info = buffer[idx]
                goal_img = np.copy(s[:, :, 2])  # Hardcoded grip as blue channel
                end_effector = np.copy(info['end_points'][-1]).astype('float32')
                angle = sum(info['angles'])
                goal = np.concatenate([end_effector, [angle]], axis=0)
                for i in range(idx+1):
                    s, a, r, d, ns, info = buffer[i]
                    # _s = np.zeros(s.shape, dtype=s.dtype)
                    # _ns = np.zeros(s.shape, dtype=s.dtype)
                    _s = np.concatenate([s[:, :, :-1], goal_img[:, :, None]], axis=2)
                    _ns = np.concatenate([ns[:, :, :-1], goal_img[:, :, None]], axis=2)
                    end_effector = info['end_points'][-1]
                    angle = sum(info['angles'])
                    r = self.env.compute_reward(end_effector, angle, goal)
                    d = True if r == 0 else False
                    buffer.append((_s, a, r, d, _ns, info))
        return buffer

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def a_fn(s):
        # return np.random.uniform(-1, 1, size=3)
        return np.asarray([1, 0, 0], dtype='float32')

    def plot(s):
        fig = plt.figure()
        fig.add_subplot(121)
        plt.imshow(s[:, :, :-1])
        fig.add_subplot(122)
        plt.imshow(s[:, :, -1])
        plt.show()

    def plot_multiple(ss):
        fig = plt.figure()
        for i in range(len(ss)):
            fig.add_subplot(100 + len(ss)*10 + i + 1)
            plt.imshow(ss[i][:, :, :-1])
        plt.show()

    worker = SingleWorker(a_fn, True)
    buffer, _ = worker.run()
