#!/bin/bash

conda create --name vae python=3.7 -y
conda activate vae
pip install -r requirements.txt

# add to .bashrc 
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # for training deterministicity
export PMIX_MCA_gds=^ds12  # for shutting off coinrun error

cat << EOF
Installation done in conda environment "vae".
Don't forget to setup your AWS credentials (usually in $HOME/.aws/credentials) if you will use S3 checkpoints.
EOF




# new
conda create --name vae python=3.7 -y
conda activate vae
pip install stable-baselines3 # gets 0.10.0, and torch 1.7.1
pip install tensorflow==2.3.1
pip install https://github.com/openai/baselines/archive/7139a66d333b94c2dafc4af35f6a8c7598361df6.zip
