import os
import argparse
import shutil
import subprocess
import time

import numpy as np
from mpi4py import MPI

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--restore_id', type=str, default='')
    parser.add_argument('--num_levels', type=int, required=True)
    parser.add_argument('--set_seed', type=int, default=-1)
    parser.add_argument('--bg_index', type=int, default=0)
    parser.add_argument('--ground_index', type=int, default=0)
    parser.add_argument('--agent_index', type=int, default=0)
    parser.add_argument('--num_envs', type=int, default=32)
    parser.add_argument('--skip_frame', type=int, default=1)
    parser.add_argument('--pvi', type=int, default=-1)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000

    # make directory
    DIR_NAME = './VAE/records/'
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME, exist_ok=True)
    
    # set file name
    filename = DIR_NAME + "/" + 'sav_' + args.run_id + '_' + str(rank) + '_' + str(seed * 100 + rank)+".npz"
    
    obs_list = []

    for level in range(args.num_levels):
        print('='*80)
        print('Extracting Samples {}/{}'.format(level + 1, args.num_levels))
        print('='*80)
        subprocess.check_output(
            ["python", "-m", "coinrun.extract_sample",
             "--run-id", args.run_id,
             "--restore-id", args.restore_id,
             "--num-levels", str(args.num_levels),
             "--set-seed", str(args.set_seed),
             "-bg_index", str(args.bg_index),
             "-ground_index", str(args.ground_index),
             "-agent_index", str(args.agent_index),
             "-yg_level", str(level),
             "-ne", str(args.num_envs),
             "-pvi", str(args.pvi),
             "-extract_seed", str(seed),
             "-extract_rank", str(rank)]
        )
        obs = np.load(filename)['obs']
        obs_list.append(obs.copy())
        os.remove(filename)

    
    mb_obs = np.concatenate(obs_list, axis=1)

    mb_obs = mb_obs[::args.skip_frame, ...]
    
    np.savez_compressed(filename, obs=mb_obs)


if __name__ == '__main__':
    main()