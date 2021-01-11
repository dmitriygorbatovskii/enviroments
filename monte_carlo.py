import env_2
import numpy as np
import matplotlib.pyplot as plt


Q = []
for y in range(-9, 10):
    for x in range(-14, 15):
        Q.append([y, x, 0, 0, 0, 0])
Q2 = Q
qwe = []
ewq = []
env = env_2.test_env()
for i in range(1000):
    observation = env.reset()
    R = []
    v = 0
    min_steps = env.min_steps()
    while True:
        result = [item for item in Q if item[:2] == observation][0]
        action = result[2:].index(max(result[2:]))
        obs, reward, done, info = env.step(action)

        f = []
        for j in range(len(result)):
            if j == action+2:
                f.append(result[j]+reward)
            else:
                f.append(result[j])
        R.append(f)

        result[action+2] += reward
        observation = obs
        v+=1
        if done:

            a = [0, 0, 0, 0]
            for r in R:
                for p in range(4):
                    a[p] += r[p+2]
            for d in range(4):
                a[d] = (a[d]/v)*0.01

            for x in range(len(Q)):
                for y in range(len(R)):
                    if Q[x][:2] == R[y][:2]:
                        for h in range(4):
                            Q2[x][2+h] += a[h]

            Q = Q2
            qwe.append(v - min_steps)
            ewq.append(i)
            break


poly = np.polyfit(ewq, qwe, 5)
poly_y = np.poly1d(poly)(ewq)
plt.plot(ewq, qwe)
plt.plot(ewq, poly_y)
plt.show()
