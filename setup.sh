#!/bin/bash

conda create --name vae python=3.7 -y
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vae
pip install -r requirements.txt
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # for training deterministicity

cat << EOF
Installation done in conda environment "vae".
Don't forget to setup your AWS credentials (usually in $HOME/.aws/credentials) if you will use S3 checkpoints.
EOF