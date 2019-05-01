import sys
import pickle
import numpy as np
import logger
from ddpg import DDPG
from replaybuffer import ReplayBuffer
from wrappers import SingleWorker


class Agent(object):
    def __init__(self):
        self.brain = DDPG()
        self.train_worker = SingleWorker(self.brain.action, True)
        # self.eval_worker = SingleWorker(self.brain.action, False)
        self.memory = ReplayBuffer()
        self.f_cnt = 0
        self.ep_cnt = 0
        self.train_rate = 0
        self.best_rate = 90

    def perceive(self):
        self.ep_cnt += 1
        _buffer, success = self.train_worker.run(self.ep_cnt, training=True)
        # episode_a = [_buffer[i][1] for i in range(min([50, len(_buffer)]))]
        # logger.log(f'Ep: {self.ep_cnt} | Mean: {np.mean(episode_a)} | Max: {np.max(episode_a)} | Min: {np.min(episode_a)}')
        self.train_rate += success
        if self.ep_cnt % 100 == 0:
            logger.log(f'Train current 100 episodes to {self.ep_cnt}: {self.train_rate}%')
            self.train_rate = 0
        for s, a, r, d, ns, inf in _buffer:
            self.memory.append(s, a, r, d)

        if len(self.memory) > 50000:
            for i in range(len(_buffer)//4):
                sb, ab, rb, db, nsb = self.memory.sample()
                self.brain.train_critic(sb, ab, rb, db, nsb)
                self.brain.train_actor(sb)
                # self.brain.train(sb, ab, rb, db, nsb)
                self.brain.update_target()

        if self.ep_cnt % 200 == 1:
            self.f_cnt = 0
            rate = 0
            for i in range(50):
                buf, sc = self.train_worker.run(self.ep_cnt, training=False)
                rate += sc
                # if sc:
                #     self.f_cnt += 1
                #     b = [buf[i][0] for i in range(len(buf))]
                #     with open(f'success{self.ep_cnt}_{self.f_cnt}.pkl', 'wb') as f:
                #         pickle.dump(b, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.log(f'Eval {self.ep_cnt}: {rate*2}%')
            if rate*2 >= self.best_rate:
                d = self.brain.save('./model', self.ep_cnt)
                logger.log(d)
                self.best_rate = rate*2
                # if rate*2 == 100 and self.ep_cnt > 1000:
                #     logger.log('Rate: 100%')
                #     sys.exit()
