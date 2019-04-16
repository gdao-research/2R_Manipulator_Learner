import numpy as np

class OUNoise(object):
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        self.x_prev += self.theta*(self.mu - self.x_prev)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        return self.x_prev

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
