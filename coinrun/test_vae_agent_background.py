"""
Load an agent trained with train_agent.py and 
"""

import time
import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import coinrun.main_utils as utils

from coinrun import setup_utils
from coinrun.config import Config
from coinrun import policies, wrappers
from VAE.vae import ConvVAE
#from PIL import Image
import scipy.misc

mpi_print = utils.mpi_print

def create_act_model(sess, env, nenvs, z_size):
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = policies.get_policy()
    act = policy(sess, z_size, ac_space, nenvs, 1, reuse=False)

    return act

def enjoy_env_sess(sess, DIR_NAME):
    should_render = True
    should_eval = Config.TRAIN_EVAL or Config.TEST_EVAL
    rep_count = Config.REP

    file_name = '%s/%s.txt'%(DIR_NAME, Config.RESTORE_ID)
    f_io = open(file_name, 'a')
    
    if should_eval:
        if Config.TEST_NUM_EVAL > -1:
            env = utils.make_general_env(Config.TEST_NUM_EVAL)
        else:
            env = utils.make_general_env(Config.NUM_EVAL)
        should_render = False
    else:
        env = utils.make_general_env(1)

    env = wrappers.add_final_wrappers(env)

    if should_render:
        from gym.envs.classic_control import rendering

    nenvs = env.num_envs
    
    vae = ConvVAE(z_size=Config.VAE_Z_SIZE, batch_size=nenvs, is_training=False, reuse=False, gpu_mode=True, use_coord_conv=True)
    agent = create_act_model(sess, env, nenvs, Config.VAE_Z_SIZE)
    num_actions = env.action_space.n
    
    init_rand = tf.variables_initializer([v for v in tf.global_variables() if 'randcnn' in v.name])
    sess.run(tf.global_variables_initializer())
    
    soft_numpy = tf.placeholder(tf.float32, [nenvs, num_actions], name='soft_numpy')
    dist = tfp.distributions.Categorical(probs=soft_numpy)
    sampled_action = dist.sample()
    
    loaded_params = utils.load_params_for_scope(sess, 'model')
    vae.load_json_full(Config.VAE_PATH)
    
    if not loaded_params:
        print('NO SAVED PARAMS LOADED')

    obs = env.reset()
    t_step = 0

    if should_render:
        viewer = rendering.SimpleImageViewer()

    should_render_obs = not Config.IS_HIGH_RES

    def maybe_render(info=None):
        if should_render and not should_render_obs:
            env.render()

    maybe_render()

    scores = np.array([0] * nenvs)
    score_counts = np.array([0] * nenvs)
    curr_rews = np.zeros((nenvs, 3))

    def should_continue():
        if should_eval:
            return np.sum(score_counts) < rep_count * nenvs

        return True

    state = agent.initial_state
    done = np.zeros(nenvs)
    
    actions = [env.action_space.sample() for _ in range(nenvs)]
    actions = np.array(actions)
    obs, _, _, _ = env.step(actions)
    
    sess.run(init_rand)
    while should_continue():
        
        #scipy.misc.imsave('raw_inputs.png', obs[0])
        encoder_in = obs.astype(np.float32) / 255.0
        batch_z = vae.encode(encoder_in)
        #reconstruct = vae.decode(batch_z)
        #scipy.misc.imsave('recon.png', reconstruct[0])
        
        action, values, state, _ = agent.step(batch_z, state, done)  
        obs, rew, done, info = env.step(action)
        
        if should_render and should_render_obs:
            if np.shape(obs)[-1] % 3 == 0:
                ob_frame = obs[0,:,:,-3:]
            else:
                ob_frame = obs[0,:,:,-1]
                ob_frame = np.stack([ob_frame] * 3, axis=2)
            viewer.imshow(ob_frame)

        curr_rews[:,0] += rew

        for i, d in enumerate(done):
            if d:
                if score_counts[i] < rep_count:
                    score_counts[i] += 1

                    if 'episode' in info[i]:
                        scores[i] += info[i].get('episode')['r']

        maybe_render(info[0])

        t_step += 1

        if should_render:
            time.sleep(.02)

        if done[0]:
            if should_render:
                mpi_print('ep_rew', curr_rews)

            curr_rews[:] = 0

    result = 0

    if should_eval:
        mean_score = np.mean(scores) / rep_count
        max_idx = np.argmax(scores)

        result = mean_score
        
        f_io.write("{}\n".format(result))
        f_io.close()
        
    return result

def main():
    utils.setup_mpi_gpus()
    setup_utils.setup_and_load()
    DIR_NAME = Config.TEST_LOG_NAME
    
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        results = enjoy_env_sess(sess, DIR_NAME)
        print(results)

if __name__ == '__main__':
    main()