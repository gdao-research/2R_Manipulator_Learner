from collections import deque
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size=1000000):
        self.buffer = deque([], maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, s, a, r, d):
        self.buffer.append((s, a, r, d))

    def sample(self, batch_size=128):
        idx = np.random.choice(len(self.buffer) - 2, size=batch_size)
        sb, ab, rb, db, nsb = [], [], [], [], []
        for i in idx:
            s, a, r, d, ns = self._decode_sample(i)
            sb.append(s)
            ab.append(a)
            rb.append(r)
            db.append(d)
            nsb.append(ns)
        return np.asarray(sb), np.asarray(ab), np.asarray(rb), np.asarray(db), np.asarray(nsb)

    def _decode_sample(self, i):
        s, a, r, d = self.buffer[i]
        if r != 0 and d:  # Ignore 50th step sample
            i += 1
            s, a, r, d = self.buffer[i]
        ns = self.buffer[i+1][0]  # next state is state of next experience
        return s, a, [r], [d], ns


if __name__ == '__main__':
    from wrappers import SingleWorker
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
    rbuffer = ReplayBuffer()
    buf, _ = worker.run()
    for s, a, r, d, ns, info in buf:
        rbuffer.append(s, a, r, d)
    
