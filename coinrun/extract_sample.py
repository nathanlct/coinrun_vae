"""
Train an agent using a PPO2 based on OpenAI Baselines.
"""

import time
import os
import numpy as np
import tensorflow as tf
import coinrun.main_utils as utils

from baselines.common import set_global_seeds
from mpi4py import MPI
from coinrun import setup_utils, policies, wrappers, ppo2
from coinrun.config import Config

def create_act_model(sess, env, nenvs):
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = policies.get_policy()
    act = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)

    return act

def main(sess):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000

    if Config.EXTRACT_SEED != -1:
        seed = Config.EXTRACT_SEED
    if Config.EXTRACT_RANK != -1:
        rank = Config.EXTRACT_RANK

    set_global_seeds(seed * 100 + rank)

    utils.setup_mpi_gpus()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    use_policy = (Config.RESTORE_ID != '')

    nenvs = Config.NUM_ENVS
    total_timesteps = int(502e6)
    env = utils.make_general_env(nenvs, seed=rank)

    if use_policy:
        agent = create_act_model(sess, env, nenvs)
        sess.run(tf.global_variables_initializer())
        loaded_params = utils.load_params_for_scope(sess, 'model')
        if not loaded_params:
            print('NO SAVED PARAMS LOADED')

    # make directory
    DIR_NAME = './VAE/records/'
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME, exist_ok=True)
    
    # set file name
    filename = DIR_NAME+"/"+Config.get_save_file()+"_"+str(seed * 100 + rank)+".npz"
    
    with tf.Session(config=config):
        env = wrappers.add_final_wrappers(env)
        nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        obs[:] = env.reset()
        dones = [False for _ in range(nenv)]
        
        # remove noisy inputs
        actions = [env.action_space.sample() for _ in range(nenv)]
        actions = np.array(actions)
        obs[:], rewards, dones, _ = env.step(actions)
        state = agent.initial_state
        
        mb_obs, mb_rewards, mb_actions, mb_next_obs, mb_dones = [],[],[],[],[]
        # For n in range number of steps
        for _ in range(400):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            if use_policy:
                actions, _, _, _ = agent.step(obs, state, dones)
            else:
                actions = [env.action_space.sample() for _ in range(nenv)]
            actions = np.array(actions)
            mb_obs.append(obs.copy())
            mb_actions.append(actions)
            mb_dones.append(dones)
            
            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            obs[:], rewards, dones, _ = env.step(actions)
            mb_next_obs.append(obs.copy())
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=obs.dtype)
        mb_next_obs = np.asarray(mb_next_obs, dtype=obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        
        #np.savez_compressed(filename, obs=mb_obs, action=mb_actions, next_obs=mb_next_obs, reward=mb_rewards, dones=mb_dones)
        np.savez_compressed(filename, obs=mb_obs)
        return filename
        
if __name__ == '__main__':
    utils.setup_mpi_gpus()
    setup_utils.setup_and_load()
    with tf.Session() as sess:
        main(sess)
