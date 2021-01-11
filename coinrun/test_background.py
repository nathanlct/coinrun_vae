"""
Load an agent trained with train_agent.py and 
"""

import time
import os
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import coinrun.main_utils as utils

import scipy.misc
from coinrun import setup_utils
from coinrun.config import Config
from coinrun import policies, wrappers
#from PIL import Image

def create_act_model(sess, env, nenvs):
    ob_space = env.observation_space
    ac_space = env.action_space

    policy = policies.get_policy()
    act = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)

    return act

def enjoy_env_sess(sess, DIR_NAME):
    should_render = True
    should_eval = Config.TRAIN_EVAL or Config.TEST_EVAL
    rep_count = Config.REP
    mpi_print = utils.mpi_print

    
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
    
    agent = create_act_model(sess, env, nenvs)
    num_actions = env.action_space.n
    
    init_rand = tf.variables_initializer([v for v in tf.global_variables() if 'randcnn' in v.name])
    sess.run(tf.compat.v1.global_variables_initializer())
    
    soft_numpy = tf.placeholder(tf.float32, [nenvs, num_actions], name='soft_numpy')
    dist = tfp.distributions.Categorical(probs=soft_numpy)
    sampled_action = dist.sample()
    
    loaded_params = utils.load_params_for_scope(sess, 'model')

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
    
    sess.run(init_rand)
    while should_continue():
        if Config.USE_LSTM == 8425 or Config.USE_LSTM == 1081:
            q_actions, values, state, _ = agent.step(obs, state, done)
            # e-greedy
            greedy_flag = np.random.rand(q_actions.shape[0])
            greedy_flag = greedy_flag < 0.1
            greedy_flag.astype(np.int)
            random_actions = np.random.randint(0, num_actions, size=q_actions.shape[0])
            action = random_actions*greedy_flag + (1-greedy_flag)*q_actions
        else:
            total_soft = agent.get_softmax(obs, state, done)
            action = sess.run([sampled_action], {soft_numpy:total_soft})
            action = action[0]
            #action, values, state, _ = agent.step(obs, state, done)
            
        obs, rew, done, info = env.step(action)
        #scipy.misc.imsave('raw_inputs.png', obs[0])
        #print(dd)
        
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
    
    with tf.compat.v1.Session(config=config) as sess:
        results = enjoy_env_sess(sess, DIR_NAME)
        print(results)

if __name__ == '__main__':
    main()
