import numpy as np
import random

# https://stable-baselines3.readthedocs.io/en/master/
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from coinrun_env import CoinrunEnv
from ppo import PPO


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
    def _on_training_start(self) -> None:
        pass
    def _on_rollout_start(self) -> None:
        pass
    def _on_step(self) -> bool:
        pass
    def _on_rollout_end(self) -> None:
        pass
    def _on_training_end(self) -> None:
        pass

def make_env():
    def _init():
        env = Monitor(CoinrunEnv())
        return env
    return _init


if __name__ == '__main__':
    eval_env = CoinrunEnv()
    check_env(eval_env)

    # vectorize env (1 per cpu)
    n_cpus = 4
    env = SubprocVecEnv([make_env() for _ in range(n_cpus)])
    # env = Monitor(env, filename='logs.txt')

    # init model
    model = PPO(
        policy='CnnPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=2048,  # 2048,  # true train batch size is n_steps * n_cpus (n_cpus being number of envs running in parallel)
        batch_size=64,  # minibatch size
        n_epochs=10, # 10, # number of passes to do over the whole rollout buffer (of size 2048*n_cpus) during one training iter
        create_eval_env=False,  # todo
        seed=None,
        verbose=2,
        tensorboard_log="./ppo_logs/")

    # evaluate
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    # save a checkpoint every n steps
    checkpoint_callback = CheckpointCallback(save_freq=5_000, save_path='./ppo_checkpoints/', name_prefix='debug_model')
    callbacks = CallbackList([checkpoint_callback, CustomCallback()])
    # cf /Users/nathan/opt/anaconda3/envs/vae/lib/python3.7/site-packages/stable_baselines3/common/callbacks.py
    # to make own checkpoints to have more control
    # the save_freq of cp_callback here doesnt take parallelism into account, so like 10k train steps with
    # 3 agents will only be 3333 steps for cp callback not enough to reach 5k (save freq)

    # train
    model.learn(total_timesteps=10_000, callback=callbacks)

    # evaluate
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


    # ------------------------------------------------------------------------------------------------------
    # model = PPO.load("model_save")
    # env = CoinrunEnv()
    # obs = env.reset()
    # for i in range(10000):
    #     action = random.randint(0, 6)
    #     # action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    #     if reward > 0:
    #         print(i, action, reward, done)
    #     if done:
    #         obs = env.reset()

    # env.close()
