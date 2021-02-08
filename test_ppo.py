import numpy as np
import random

from coinrun_env import CoinrunEnv


if __name__ == '__main__':
    env = CoinrunEnv()
    obs = env.reset()
    for i in range(10000):
        action = random.randint(0, 6)
        obs, reward, done, info = env.step(action)
        # env.render()
        if reward > 0:
            print(i, reward)
        if done:
            print('done', i)
            obs = env.reset()

    env.close()
