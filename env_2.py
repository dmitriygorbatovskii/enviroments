import numpy as np
import random

"""
окружение представляет собой поле 10 на 15, в котором в случайных местах 
появляется агент 'A' и цель 'T'. целью агента является достижение местоположения
цели, за это он получает награду в 10 очков. observation - растояние между
целью и агентом на развенутом одномерном массиве. агент получает положительную
награду при приближении к цили и отрицательную в обратном случае.
"""

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

CONFIG = {
    'len_x': 15,
    'len_y': 10,
    'multiplier': 0.01
}


class test_env():

    def __init__(self, config=CONFIG):
        self.area = []
        self.reward = 0
        self.shape = (config['len_y'], config['len_x'])
        self.area_size = np.prod(self.shape)
        self.pos_agent = 0
        self.pos_target = 0

    def reset(self):
        self.done = False
        self.reward = 0

        while True:
            self.pos_agent  = random.choice(list(range(self.area_size)))
            self.pos_target = random.choice(list(range(self.area_size)))
            if self.pos_target!=self.pos_agent: break

        self.area = np.full(self.shape, '_')
        self.area = np.ravel(self.area)
        self.area[self.pos_agent] = 'A'
        self.area[self.pos_target] = 'T'
        self.area = np.reshape(self.area, self.shape)
        return self.get_observation()

    def step(self, action):
        self.done = False
        info = {}
        self.prev_pos = self.position(self.pos_agent)
        if self.movie(action)[0] in range(10) and self.movie(action)[1] in range(15):
            self.area[self.position(self.pos_agent)] = '_'
            self.area[self.movie(action)]  = 'A'
            self.pos_agent = list(np.ravel(self.area)).index('A')
            self.reward = self.calculate_reward()
        else:
            self.reward = -1
        return self.get_observation(), self.reward, self.done, info

    def get_observation(self):
        return self.pos_agent-self.pos_target

    def render(self):
        print(self.area)

    def calculate_reward(self, config=CONFIG):
        y = abs(self.position(self.pos_agent)[0] - self.position(self.pos_target)[0])
        x = abs(self.position(self.pos_agent)[1] - self.position(self.pos_target)[1])
        dy = abs(self.prev_pos[0] - self.position(self.pos_target)[0]) - y
        dx = abs(self.prev_pos[1] - self.position(self.pos_target)[1]) - x
        d = dy + dx
        if self.get_observation() == 0:
            self.reward += 10
        elif d == 1:
            self.reward = 1/abs(self.get_observation())
        else:
            self.reward = -abs(self.get_observation()*config['multiplier'])
        return self.reward

    def movie(self, action):
        a = self.position(self.pos_agent)
        if action == 0:
            return a[0], a[1]-1
        elif action == 1:
            return a[0]-1, a[1]
        elif action == 2:
            return a[0], a[1]+1
        elif action == 3:
            return a[0]+1, a[1]

    def position(self, pos):
        y = pos//15
        x = pos-15*y
        return y, x




