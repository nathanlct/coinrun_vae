from coinrun import setup_utils, make
import gym
import numpy as np


class CoinrunEnv(gym.Env):
    def __init__(self, num_envs=1):
        super(CoinrunEnv, self).__init__()

        self.num_envs = 1
        self.env_type = 'standard'  # standard, platform, maze

        self._init_coinrun()

    def _init_coinrun(self):
        setup_utils.setup_and_load(use_cmd_line_args=False)
        self.env = make(self.env_type, num_envs=self.num_envs) 

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        """
        Input: array of actions from Discrete(7) of size (self.num_envs,)
            Actions are:
            0 don't move - 1 go right - 2 go left
            3 jump straight - 4 right-jump - 5 left-jump
            6 go down (step down from a crate)
        Output: 
            observations (self.num_envs, 64, 64, 3)    (or 3 -> 1 if black and white)
            rewards (self.num_envs,)
            dones (self.num_envs,) 
            infos
        """
        obs, rwd, done, _ = self.env.step(np.array([action]))
        return obs[0], float(rwd[0]), bool(done[0]), {}

    def random_step(self):
        actions = np.array([self.action_space.sample() for _ in range(self.num_envs)])
        return self.step(actions)

    def render(self, mode='human'):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        return obs[0]

    def close(self):
        self.env.close()

    def __del__(self):
        self.env.close()