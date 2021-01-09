from collections import defaultdict
import numpy as np
import boto3
import os
import os.path
from datetime import datetime
import argparse
import json

import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import CoinrunDataset
from vae import VAE


# hyperparameters
parser = argparse.ArgumentParser(
    description='Coinrun VAE training.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--expname', type=str, default='coinrun_vae',
    help='Name of the experiment')
parser.add_argument('--description', type=str, default='',
    help='Additional description for the experiment')
parser.add_argument('--local', action='store_true', default=False,
    help='Set this to true when running locally without GPU')
parser.add_argument('--seed', type=int, default=1234,
    help='Seed for deterministic training')

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

args = parser.parse_args()

# create save folder
today = datetime.now().strftime('%d-%m-%Y')
expname = args.expname.replace(' ', '_') + datetime.now().strftime('_%Hh%M')
save_folder = os.path.join(args.save_folder, today, expname)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f'Saving at {save_folder}')
    # os.makedirs(os.path.join(save_folder, 'reconstructions'))
    # os.makedirs(os.path.join(save_folder, 'samples'))
    os.makedirs(os.path.join(save_folder, 'checkpoints'))
else:
    print(f'Save folder {save_folder} already exists. Aborting')
    exit(0)

# s3
if args.s3:
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(args.s3_bucket)

# save args to file
with open(os.path.join(save_folder, 'params.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

# set device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device {device}')

# set seed for reproducibility (https://pytorch.org/docs/stable/notes/randomness.html)
if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False  # can reduce performance

# dataset
set_range = transforms.Lambda(lambda X: 2 * X - 1.)
transform = transforms.Compose([transforms.ToTensor(), set_range])

train_dataset = CoinrunDataset('dataset/data_initial_state.npz', split='train', transform=transform)
val_dataset = CoinrunDataset('dataset/data_initial_state.npz', split='test', transform=transform)

num_workers = 0 if args.local else 4
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    drop_last=True, num_workers=num_workers, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
    drop_last=True, num_workers=num_workers, pin_memory=True)

n_train, n_train_batches = len(train_dataset), len(train_dataloader)
n_val, n_val_batches = len(val_dataset), len(val_dataloader)

assert(n_train_batches == n_train // args.batch_size)
assert(n_val_batches == n_val // args.batch_size)

print(f'Initialized training dataset with {n_train} samples, {n_train_batches} batches')
print(f'Initialized validation dataset with {n_val} samples, {n_val_batches} batches')

# model
model = VAE(latent_dim=args.latent_dim).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# tensorboard logging
writer = SummaryWriter(log_dir=save_folder)

# training loop
print(f'** Starting experiment {args.expname} **')

for epoch in range(args.epochs):
    print(f'----- Epoch {epoch+1}/{args.epochs} -----')

    # train for one epoch
    print(f'> training over {n_train_batches} batches')
    model.train()
    train_losses = defaultdict(list)
    for batch_idx, batch in enumerate(train_dataloader):
        print('.', end='', flush=True)

        data = batch.to(device)
        optimizer.zero_grad()

        results = model(data)
        train_loss = model.loss(data, *results, epoch, args.epochs)
        for k, v in train_loss.items():
            train_losses[k].append(v.item())

        train_loss['loss'].backward()
        optimizer.step()

    # print epoch training data
    print()
    train_loss_data = []
    for k, v in train_losses.items():
        train_loss_data.append(f'{k}: {round(np.mean(v), 3)} (+- {round(np.std(v), 3)})')
        writer.add_scalar(f'Train/{k}', np.mean(v), epoch)
    print(', '.join(train_loss_data))

    # validate every args.validate_every epochs and at the last epoch
    if (args.validate_every > 0 and epoch % args.validate_every == 0) or epoch == args.epochs - 1:
        print(f'> validating over {n_val_batches} batches')
        with torch.no_grad():
            model.eval()
            val_losses = defaultdict(list)
            for batch_idx, batch in enumerate(val_dataloader):
                print('.', end='', flush=True)

                data = batch.to(device)
                results = model(data)
                val_loss = model.loss(data, *results, epoch, args.epochs)
                for k, v in val_loss.items():
                    val_losses[k].append(v.item())

            # print validation data
            print()
            val_loss_data = []
            for k, v in val_losses.items():
                val_loss_data.append(f'{k}: {round(np.mean(v), 3)} (+- {round(np.std(v), 3)})')
                writer.add_scalar(f'Val/{k}', np.mean(v), epoch)
            print(', '.join(val_loss_data))

            # upload to tensorboard some validation dataset reconstructions 
            # and reconstruction from random samples in the latent space
            print('> sampling images')
            test_input = next(iter(val_dataloader)).to(device)
            recons = model.generate(test_input)
            grid1 = torchvision.utils.make_grid(recons.data, normalize=True, nrow=8)
            samples = model.sample(args.batch_size, device)
            grid2 = torchvision.utils.make_grid(samples.cpu().data, normalize=True, nrow=8)
            writer.add_image('reconstructions', grid1, epoch)
            writer.add_image('samples', grid2, epoch)    
            del test_input, recons, samples

    if (args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0) or epoch == args.epochs - 1:
        print('> saving checkpoint')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(save_folder, f'checkpoints/epoch_{epoch}.checkpoint'))

        if args.s3:
            for path, subdirs, files in os.walk(save_folder):
                directory_name = path.replace(args.save_folder + '/', '')
                for file in files:
                    bucket.upload_file(os.path.join(path, file), os.path.join(args.s3_path, directory_name, file))

writer.close()

print(f'Saved at {save_folder}')
if args.s3:
    print(f'and at s3://{args.s3_bucket}/{args.s3_path}/{today}/{expname}')


