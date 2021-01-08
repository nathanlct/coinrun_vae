# Coinrun VAE

## Installation

```
git clone https://github.com/nathanlct/coinrun_vae
cd coinrun_vae
./setup.sh
```

## Usage

```
python main.py --expname test
```

See `main.py` for available command line arguments.

## Datasets

`dataset/data_initial_state.npz` was generated using:

```
python -m coinrun.extract_sample --num-levels 500 -large_random_flag 1
```