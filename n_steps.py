import env_2
import numpy as np
import matplotlib.pyplot as plt


Q = []
for y in range(-9, 10):
    for x in range(-14, 15):
        Q.append([y, x, 0, 0, 0, 0])

env = env_2.test_env()
a = []
b = []
alpha = 0.001
for i in range(5000):
    observation = env.reset()
    v = 0
    min_steps = env.min_steps()
    while True:
        result1 = [item for item in Q if item[:2] == observation]
        action = list(result1)[0][2:].index(max(list(result1)[0][2:]))
        obs, reward, done, info = env.step(action)
        pos = env.pos_agent
        result2 = [item for item in Q if item[:2] == obs]
        action2 = list(result2)[0][2:].index(max(list(result2)[0][2:]))
        obs2, reward1, done, info = env.step(action2)
        result3 = [item for item in Q if item[:2] == obs2]
        action3 = list(result3)[0][2:].index(max(list(result3)[0][2:]))
        obs3, reward2, done, info = env.step(action3)
        list(result1)[0][action+2] = list(result1)[0][action+2]+ alpha * list(result2)[0][action2+2]\
                                     + alpha * (reward + 1 * (list(result3)[0][action3+2]) - list(result1)[0][action+2])
        observation = obs
        env.set_position(pos)
        v += 1

        if done:
            list(result2)[0][action2 + 2] = list(result2)[0][action2 + 2] + alpha * (
                        reward1 + 1 * (list(result2)[0][action3 + 2]) - list(result2)[0][action2 + 2])
            list(result3)[0][action3 + 2] = alpha * reward2
            '''list(result2)[0][action2 + 2] += alpha * reward1
            list(result3)[0][action3 + 2] += alpha * reward2'''
            print(i, v+2-min_steps)
            a.append(v+2-min_steps)
            b.append(i)
            break


print(1)
poly = np.polyfit(b, a, 5)
poly_y = np.poly1d(poly)(b)
plt.plot(b, a)
plt.plot(b, poly_y)
plt.show()
