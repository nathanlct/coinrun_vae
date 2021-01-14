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