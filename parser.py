import argparse
from termcolor import colored as c

BOLD = lambda s: c(s, attrs=['bold'])
TINFO = '[' + c('INFO', 'cyan', attrs=['bold']) + '] '
TLOG = '[' + c('LOG', 'yellow', attrs=['bold']) + '] '
TERR = '[' + c('ERROR', 'red', attrs=['bold']) + '] '
TLINE = TINFO + '-' * 80


def _arg_parser():
    parser = argparse.ArgumentParser(
        description='Coinrun VAE training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--expname', type=str, default='coinrun_vae',
        help='Name of the experiment')
    parser.add_argument('--description', type=str, default='',
        help='Additional description for the experiment')
    parser.add_argument('--debug', action='store_true', default=False,
        help='Set this to true when debugging on CPU to train on small dataset')
    parser.add_argument('--seed', type=int, default=None,
        help='Seed for deterministic training')
    parser.add_argument('--data_path', type=str, default='dataset/data_initial_state.npz',
        help='Path of dataset to use for training and validation')

    parser.add_argument('--lr', type=float, default=3e-4,
        help='Training learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
        help='Training batch size')
    parser.add_argument('--latent_dim', type=int, default=32,
        help='Dimension of the VAE latent space (bottleneck)')

    parser.add_argument('--epochs', type=int, default=5,
        help='Number of training epochs')
    parser.add_argument('--validate_every', type=int, default=-1,
        help='Run a validation step every n epochs. Set to -1 to validate at the end only')
    parser.add_argument('--checkpoint_every', type=int, default=-1,
        help='Create a checkpoint every n epochs. Set to -1 for a checkpoint at the end only')
    parser.add_argument('--save_folder', type=str, default='vae_results',
        help='Local folders where experiment data and checkpoints are saved')

    parser.add_argument('--s3', action='store_true', default=False,
        help='Where to upload the local save folder to AWS S3')
    parser.add_argument('--s3_bucket', type=str, default='nathan.experiments',
        help='Name of the AWS S3 bucket where data should be saved (requires --s3)')
    parser.add_argument('--s3_path', type=str, default='adversarial/coinrun_vae',
        help='Path where data should be saved in the AWS S3 bucket (requires --s3)')

    parser.add_argument('--notify', action='store_true', default=False,
        help='Whether to send a Pushover notification when training ends or if it errors '
             '(requires to have APP_TOKEN and USER_TOKEN setup in utils.py)')

    parser.add_argument('--beta_min', type=float, default=1.0,
        help='Initial beta the KLD loss will be multiplied by')
    parser.add_argument('--beta_max', type=float, default=1.0,
        help='Final beta the KLD loss will be multiplied by '
             '(beta will increase linearly from beta_min to beta_max over the training)')

    return parser


def parse_args(log=True):
    parser = _arg_parser()
    args = parser.parse_args()

    if log:
        print(TLINE)
        print(TINFO + 'Using following parameters for the experiment')
        print(TLINE)
        max_len_arg = max([len(arg) for arg in vars(args)])
        for arg, value in vars(args).items():
            s = TINFO + BOLD(arg) + ' ' * (max_len_arg - len(arg) + 1) + str(value)
            print(s)
        print(TLINE)

    return args
    