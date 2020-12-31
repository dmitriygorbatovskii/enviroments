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
for i in range(5000):
    observation = env.reset()
    R = []
    v = 0
    min = env.min_steps()
    while True:
        for j in range(len(Q)):
            if Q[j][:2] == observation:
                result = Q[j]
        if type(result) is None:
            print(1)
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
            print(v-min)
            #print(v-min)

            a = [0, 0, 0, 0]
            for r in R:
                a[0] += r[2]
                a[1] += r[3]
                a[2] += r[4]
                a[3] += r[5]
            a[0] = (a[0]/v)*0.01
            a[1] = (a[1]/v)*0.01
            a[2] = (a[2]/v)*0.01
            a[3] = (a[3]/v)*0.01

            #print(a)

            for x in range(len(Q)):
                for y in range(len(R)):
                    if Q[x][:2] == R[y][:2]:
                        for h in range(4):
                            Q2[x][2+h] += a[h]
                        #Q2[x][2:] += a
            Q = Q2

            qwe.append(v - min)
            ewq.append(i)

            break


poly = np.polyfit(ewq, qwe, 5)
poly_y = np.poly1d(poly)(ewq)
plt.plot(ewq, qwe)
plt.plot(ewq, poly_y)
plt.show()
