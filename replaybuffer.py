from collections import deque
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size=1000000):
        self.buffer = deque([], maxlen= max_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, s, a, r, d, ns):
        exp = (s, a, r, d, ns) if r == 0 else (s, a, r, d, None)  # Reduce memory usage
        self.buffer.append(exp)

    def sample(self, batch_size=64):
        idx = np.random.choice(len(self.buffer) - 2, size=batch_size)
        sb, ab, rb, db, nsb = [], [], [], [], []
        for i in idx:
            s, a, r, d, ns = self._sample(i)
            sb.append(s)
            ab.append(a)
            rb.append(r)
            db.append(d)
            nsb.append(ns)
        return np.asarray(sb), np.asarray(ab), np.asarray(rb), np.asarray(db), np.asarray(nsb)

    def _sample(self, i):
        s, a, r, d, ns = self.buffer[i]
        if r != 0 and d:
            i += 1
            s, a, r, d, ns = self.buffer[i]
        if ns is None:
            ns = self.buffer[i+1][0]  # next state is state of next experience
        return s, a, [r], [d], ns
