import numpy as np
import cv2

class Manipulator2R(object):
    def __init__(self, links=(18, 13), total_steps=50, max_rot_angle=np.pi/18):
        assert sum(links) < 38  # Constraint length
        self.img_size = 84  # Constraint image size to draw

        self.links = np.asarray(links, dtype='float32')
        self._total_steps = total_steps
        self._rot_angle_range = np.asarray([-max_rot_angle, max_rot_angle], dtype='float32')
        self.nb_actions = len(links)
        self.desired_goal = np.zeros(len(links), dtype='float32')

    def _wrap_to_pi(self, angle):
        angle = (angle/np.pi*180 + 180) % 360
        if angle < 0:
            angle += 360
        angle -= 180
        angle = angle/180*np.pi
        return angle

    def reset(self):
        self.available_steps = self._total_steps
        self.current_angles = np.random.uniform(-np.pi, np.pi, size=self.links.shape).astype('float32')
        self.end_points = self._compute_end_effectors()
        assert len(self.end_points) == len(self.links) + 1
        self.current_pos = self.end_points[-1]

        self.desired_goal = self._sample_new_goal()
        while not self._is_reachable_goal(self.desired_goal) or self.is_goal(self.current_pos, self.desired_goal):  # Make sure new goal is reachable & initially not starting configuration
            self.desired_goal = self._sample_new_goal()
        # self.desired_goal = self.current_pos
        self.goal_draw = self._draw_goal()
        return self._update_canvas()

    def step(self, action):
        assert len(action) == self.nb_actions
        self.available_steps -= 1
        self.current_angles += action
        for i in range(len(self.current_angles)):
            if self.current_angles[i] > np.pi or self.current_angles[i] < -np.pi:
                self.current_angles[i] = self._wrap_to_pi(self.current_angles[i])
        self.end_points = self._compute_end_effectors()
        self.current_pos = self.end_points[-1]
        obs = self._update_canvas()
        reached = self.is_goal(self.current_pos, self.desired_goal)
        if reached:
            return obs, 0.0, True, {}
        else:
            if self.available_steps:
                return obs, -1.0, False, {}
            else:
                return obs, -1.0, True, {}

    def is_goal(self, current, desired):
        dist = np.sqrt(np.sum((current - desired)**2))
        return True if dist < 2 else False

    def _compute_end_effectors(self):
        x = 0
        y = 0
        end_points = [np.copy([self.img_size//2, self.img_size//2])]
        for i in range(len(self.links)):
            x += int(self.links[i]*np.cos(np.sum(self.current_angles[:i+1])))
            y += int(self.links[i]*np.sin(np.sum(self.current_angles[:i+1])))
            end_points.append(np.copy([x + self.img_size//2, y + self.img_size//2]))
        return end_points

    def _sample_new_goal(self):
        return np.random.randint(0, self.img_size, size=self.nb_actions)

    def _is_reachable_goal(self, goal):
        goal_to_O_dist = np.sqrt((goal[0] - self.img_size//2)**2 + (goal[1] - self.img_size//2)**2)
        return True if (self.links[0] - self.links[1]) < goal_to_O_dist < (self.links[0] + self.links[1]) else False

    def _connect_end_points(self, end0, end1):
        d = end1 - end0
        j = np.argmax(np.abs(d))
        D = d[j]
        aD = np.abs(D)
        points = end0 + (np.outer(np.arange(aD+1), d) + (aD>>1))//aD
        return (points[:, 0], points[:, 1])

    def _draw_link(self, end0, end1, colors, color_scale):
        points = self._connect_end_points(end0, end1)
        temp = np.zeros((self.img_size, self.img_size), dtype='uint8')
        temp[points] = 255
        strel = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype='uint8')
        temp = cv2.dilate(temp, strel, iterations=1)
        final = np.zeros((self.img_size, self.img_size, 3), dtype='uint8')
        for i, c in enumerate(colors):
            final[:, :, c] = (temp*color_scale[i]).astype('uint8')
        return final

    def _draw_end_effector(self):
        strel = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='uint8')
        temp = np.zeros((self.img_size, self.img_size), dtype='uint8')
        temp[self.current_pos[0], self.current_pos[1]] = 255
        temp = cv2.dilate(temp, strel, iterations=4)
        final = np.zeros((self.img_size, self.img_size, 3), dtype='uint8')
        final[:, :, 1] = temp  # end effector is green
        return final

    def _draw_goal(self):
        strel = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype='uint8')
        temp = np.zeros((self.img_size, self.img_size), dtype='uint8')
        temp[self.desired_goal[0], self.desired_goal[1]] = 255
        temp = cv2.dilate(temp, strel, iterations=3)
        final = np.zeros((self.img_size, self.img_size, 3), dtype='uint8')
        final[:, :, 0] = temp  # goal is red
        return final

    def _update_canvas(self):
        colors = [[2], [1, 2]]
        colors_scale = [[1], [0.3, 1]]
        canvas = [self.goal_draw]
        for i in range(len(self.end_points) - 1):
            canvas.append(self._draw_link(self.end_points[i], self.end_points[i+1], colors[i], colors_scale[i]))
        canvas.append(self._draw_end_effector())
        canvas = np.clip(np.sum(canvas, axis=0), 0, 255)
        return canvas


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = Manipulator2R()
    s = env.reset()
    ns, r, d, _ = env.step(np.asarray([np.pi/2, -np.pi/2]))
    fig = plt.figure()
    fig.add_subplot(121)
    plt.imshow(s)
    plt.title('s')
    fig.add_subplot(122)
    plt.imshow(ns)
    plt.title('ns')
    plt.show()
