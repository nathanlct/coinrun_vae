import numpy as np
from coinrun import setup_utils, make
from PIL import Image
import time
import gym
import random


def show_img(np_array):
    img = Image.fromarray(np_array, 'RGB')
    img.show()

def save_img(np_array, path):
    img = Image.fromarray(np_array, 'RGB')
    img.save(path)


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

###

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList


# https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html


# need callbacks
# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html

# + make sure env is vectorized (ie parallelized) (+ is n_envs = n_cpus?)
# & how does it work to parallelize envs if we have 20 cpus but 1 gpu?
# and can i use coinrun which is parallelized by default? instead of stacking several hacked coinruns
from stable_baselines3.common.vec_env import SubprocVecEnv


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.rollout_length = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.rollout_length = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.rollout_length += 1
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print(f'rollout ends after {self.rollout_length} steps')

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

def make_env():
    def _init():
        env = CoinrunEnv()
        return env
    return _init

if __name__ == '__main__':

    if True:
        env = CoinrunEnv()
        eval_env = CoinrunEnv()
        check_env(env)

        num_cpu = 3

        env = SubprocVecEnv([make_env() for _ in range(num_cpu)])

        model = PPO('CnnPolicy', env, verbose=2, tensorboard_log="./tensorboard_test/")

        # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
        # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path='./logs/',
                                                name_prefix='rl_model')
        cc = CustomCallback()
        callbacks = CallbackList([checkpoint_callback, cc])

        model.learn(total_timesteps=1_000_000, tb_log_name="test", callback=callbacks)

        model.save("save_test")

    else:
        model = PPO.load("save_test")
        env = CoinrunEnv()
        obs = env.reset()
        for i in range(10000):
            action = random.randint(0, 6)
            # action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if reward > 0:
                print(i, action, reward, done)
            if done:
                obs = env.reset()

        env.close()
