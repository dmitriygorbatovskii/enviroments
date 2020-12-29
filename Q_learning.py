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
for i in range(1000):
    observation = env.reset()
    v = 0
    min_steps = env.min_steps()
    while True:
        result1 = [item for item in Q if item[:2] == observation]
        action = list(result1)[0][2:].index(max(list(result1)[0][2:]))
        obs, reward, done, info = env.step(action)
        result2 = [item for item in Q if item[:2] == obs]
        best_next_action = list(result2)[0][2:].index(max(list(result2)[0][2:]))
        list(result1)[0][action+2] = list(result1)[0][action+2] + 0.6 * (reward + 1 * (list(result2)[0][best_next_action+2]) - list(result1)[0][action+2])
        observation = obs
        v += 1
        if done:
            a.append(v-min_steps)
            b.append(i)
            break


print(1)
poly = np.polyfit(b, a, 5)
poly_y = np.poly1d(poly)(b)
plt.plot(b, a)
plt.plot(b, poly_y)
plt.show()
#action = list(Q[observation][1:]).index(max(Q[observation][1:]))
#Q[observation][action+1] = Q[observation][action+1] + 0.6*(reward + 1*(Q[obs][best_next_action+1]) - Q[observation][action+1])
'''[6, -14]
[-5, -9]'''