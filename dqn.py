import env_2
import math, random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import matplotlib.pyplot as plt

from collections import deque

import pickle

env = env_2.test_env()

model = nn.Sequential(
    nn.Linear(2, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 4)
)
optimizer = optim.Adam(model.parameters())

replay_buffer = deque(maxlen=1000)

num_frames = 1000
batch_size = 1
gamma = 0.99

losses = []
all_rewards = []
episode_reward = 0
a = []

for i in range(100):
    state = env.reset()
    for frame_idx in range(1, num_frames + 1):
        epsilon = 1 - ((1 / 100) * i)
        if not i % 10:
            if frame_idx > 900:
                epsilon = 0.0
                env.render()

        if random.random() > epsilon:
            st3 = torch.FloatTensor(state).unsqueeze(0)
            q_value = model(st3)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(4)

        next_state, reward, done, _ = env.step(int(action))

        st2 = np.expand_dims(state, 0)
        next_st2 = np.expand_dims(next_state, 0)
        replay_buffer.append((st2, action, reward, next_st2, done))

        state = next_state
        episode_reward += reward

        if not frame_idx % 100:
            print(i, frame_idx)

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            a.append(i)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            st, action, reward, next_st, done = zip(*random.sample(replay_buffer, batch_size))
            st = np.concatenate(st)
            next_st = np.concatenate(next_st)

            st = torch.FloatTensor(np.float32(st))
            next_st = torch.FloatTensor(np.float32(next_st))
            action = torch.LongTensor(action)
            reward = torch.FloatTensor(reward)
            done = torch.FloatTensor(done)

            q_values = model(st)
            next_q_values = model(next_st)

            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = reward + gamma * next_q_value * (1 - done)

            loss = (q_value - expected_q_value.data).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data)

plt.plot(a, all_rewards)
plt.show()