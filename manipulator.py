import numpy as np
import cv2


class Manipulator(object):
    def __init__(self, base, current_angles, links=(18, 13)):
        self._links = np.array(links, dtype='float32')
        self._base = base
        self._current_angles = np.array(current_angles, dtype='float32')

    def wrap_to_PI(self, angles):
        ret = np.remainder(angles, 2*np.pi)
        mask = np.abs(ret) > np.pi
        ret[mask] -= 2*np.pi*np.sign(ret[mask])
        return ret

    def set_current_angles(self, current_angles):
        self._current_angles = np.array(current_angles)

    def get_current_angles(self):
        return self._current_angles

    def compute_end_points(self, rot_angles):
        assert len(rot_angles) == len(self._links) + 1
        self._current_angles = self.wrap_to_PI(
            self._current_angles + np.array(rot_angles))
        x = 0
        y = 0
        end_points = [np.array([self._base[0], self._base[1]])]
        for i in range(len(self._links)):
            x += int(self._links[i]*np.cos(np.sum(self._current_angles[:i+1])))
            y += int(self._links[i]*np.sin(np.sum(self._current_angles[:i+1])))
            end_points.append(np.array([x + self._base[0], y + self._base[1]]))
        return end_points


class ManipulatorEnvironment(object):
    def __init__(self, max_movements=50, links=(160, 120), img_size=640):
        assert sum(links) < img_size/2 - 10
        self._size = img_size
        self.link_colors = [[0], [1]]
        self._base = np.array([int(round(self._size/2)), int(round(self._size/2))])
        self._goal = None
        self.goal_img = None
        self._links = links
        self.max_movements = max_movements
        self.available_moves = max_movements
        self.nb_actions = len(links) + 1
        self.manipulator = Manipulator(self._base, np.zeros(self.nb_actions), self._links)
        self.end_points = self.manipulator.compute_end_points(np.zeros(self.nb_actions))

    def set_goal(self, goal=None):
        while not self._goal_is_reachable(goal):
            goal = self._sample_new_goal()
        self._goal = np.array(goal)

    def get_goal(self):
        return self._goal

    def compute_reward(self, end_effector, angle, goal=None):
        if goal is None:
            goal = np.copy(self._goal)
        if self.is_goal(goal, end_effector, angle):
            return 0
        return -1

    def is_goal(self, goal, pos, angle):
        # Goal is defined as almost same location & orientation different is less than 10 degree
        # -1 index is orientation
        return np.linalg.norm(goal[:-1] - pos) < 10 and np.abs(goal[-1] - angle) < np.pi/18

    def _goal_is_reachable(self, goal):
        if goal is None:
            return False
        dist = np.linalg.norm(np.array(goal[:2]) - self._base)
        return (self._links[0] - sum(self._links[1:])) < dist < sum(self._links)

    def _sample_new_goal(self):
        xy = np.random.randint(0, self._size, size=2)
        orientation = np.random.uniform(-np.pi, np.pi)
        return np.asarray([xy[0], xy[1], orientation])

    def draw_goal(self, goal=None):
        if goal is None:
            goal = np.copy(self._goal)
        ret = np.ones((self._size, self._size), dtype='uint8')
        ret[int(goal[0]), int(goal[1])] = 255
        k = np.asarray([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype='uint8')
        ret = cv2.dilate(ret, k, iterations=20)
        M = cv2.getRotationMatrix2D((goal[1], goal[0]), goal[2]/np.pi*180, 1)
        ret = cv2.warpAffine(ret, M, ret.shape)
        return ret.reshape((self._size, self._size, 1))

    def _connect_end_points(self, end0, end1):
        d = end1 - end0
        j = np.argmax(np.abs(d))
        D = d[j]
        aD = np.abs(D)
        points = end0 + (np.outer(np.arange(aD+1), d) + (aD >> 1))//aD
        return (points[:, 0], points[:, 1])

    def _draw_links(self):
        ret = np.zeros((self._size, self._size, 3), dtype='uint8')
        for i in range(len(self.end_points)-1):
            points = self._connect_end_points(
                self.end_points[i], self.end_points[i+1])
            temp = np.zeros((self._size, self._size), dtype='uint8')
            temp[points] = 255
            k = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='uint8')
            temp = cv2.dilate(temp, k, iterations=4)
            for c in self.link_colors[i]:
                ret[:, :, c] += temp
        return np.clip(ret, 0, 255)

    def _draw_end_effector(self, angle):
        end_effector = self.end_points[-1]
        temp = np.zeros((self._size, self._size), dtype='uint8')
        temp[end_effector[0], end_effector[1]] = 255
        k = np.asarray([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype='uint8')
        temp = cv2.dilate(temp, k, iterations=22)
        M = cv2.getRotationMatrix2D(
            (end_effector[1], end_effector[0]), angle/np.pi*180, 1)
        temp = cv2.warpAffine(temp, M, temp.shape)
        ret = np.zeros((self._size, self._size, 3), dtype='uint8')
        ret[:, :, 2] = temp
        return ret

    def _draw(self, angle):
        links_img = self._draw_links()
        end_effector_img = self._draw_end_effector(angle)
        return np.clip(links_img + end_effector_img, 0, 255)

    def reset(self, current_angles=None):
        if current_angles is None:
            current_angles = np.random.uniform(-np.pi, np.pi, size=self.nb_actions)
        self.manipulator.set_current_angles(current_angles)
        self.available_moves = self.max_movements
        self.end_points = self.manipulator.compute_end_points(current_angles)
        while self._goal is None or self.is_goal(self._goal, self.end_points[-1], current_angles[-1]):
            self.set_goal()
        # self.set_goal(self.end_points[-1])  # Test goal is initial configuratuion
        self.goal_img = self.draw_goal()
        obs = self._draw(current_angles[-1])
        info = {'end_points': self.end_points,
                'angles': self.manipulator.get_current_angles(), 'links': self._links}
        return np.concatenate([obs, self.goal_img], axis=2), info

    def step(self, action):
        self.available_moves -= 1
        self.end_points = self.manipulator.compute_end_points(action)
        obs = self._draw(self.manipulator.get_current_angles()[-1])
        reward = self.compute_reward(self.end_points[-1], action[-1])
        done = True if not self.available_moves or reward == 0 else False
        info = {'end_points': self.end_points,
                'angles': self.manipulator.get_current_angles(), 'links': self._links}
        return np.concatenate([obs, self.goal_img], axis=2), reward, done, info


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def plot(ss):
        fig = plt.figure()
        nb = len(ss)
        for i, s in enumerate(ss):
            fig.add_subplot(nb*100 + 20 + i*nb+1)
            plt.imshow(s[:, :, :3])
            plt.title('state')
            plt.axis('off')
            fig.add_subplot(nb*100 + 20 + i*nb+2)
            plt.imshow(s[:, :, 3], cmap='gray')
            plt.title('goal')
            plt.axis('off')
        plt.show()

    env = ManipulatorEnvironment()
    s, info = env.reset()
    print(s.shape, info)
    # s = cv2.resize(s, (84, 84))
    # plot(s)

    ns, r, d, info = env.step(np.asarray([0, np.pi/2, np.pi/3]))
    print(ns.shape, r, d, info)
    plot([s, ns])
