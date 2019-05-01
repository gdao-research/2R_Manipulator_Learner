import pickle
import numpy as np
import imageio
import glob

def toImg(state):
    temp = np.zeros((84, 175, 3), dtype=state.dtype)
    temp[:, :84, :] = state[:, :, :-1]
    temp[:, -84:, 0] = state[:, :, -1]
    temp[:, -84:, 1] = state[:, :, -1]
    temp[:, -84:, 2] = state[:, :, -1]
    temp[:, 84:-84, :] = 255
    return temp


files = glob.glob('success/success11401_*.pkl')
imgs = []

for filename in files:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    for d in data:
        imgs.append(toImg(d))
    imgs.append(toImg(d))
    imgs.append(toImg(d))

imageio.mimsave('movie.gif', imgs, 'GIF', duration=0.1)
