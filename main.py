import os
import tensorflow as tf
from agent import Agent
import logger
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only run on GPU 1


if __name__ == '__main__':
    logger.configure('./log4')
    sess = tf.InteractiveSession()
    agent = Agent()
    sess.run(tf.global_variables_initializer())
    agent.brain.initialize_target()
    for ep in range(100001):
        agent.perceive()
