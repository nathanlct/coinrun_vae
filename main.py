from collections import defaultdict
import numpy as np
import boto3
import os
import os.path
import sys
from datetime import datetime
import json
from tqdm import tqdm
from termcolor import colored as c

import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import CoinrunDataset
from vae import VAE
from parser import *
import utils

print(TLINE)
print(TINFO + c('INITIALIZATION', 'green', attrs=['bold']))

args = parse_args()

# create save folder
today = datetime.now().strftime('%d-%m-%Y')
expname = args.expname.replace(' ', '_') + datetime.now().strftime('_%Hh%M')
save_folder = os.path.join(args.save_folder, today, expname)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f'{TINFO}Experiment data will be saved at {BOLD(save_folder)}')
    os.makedirs(os.path.join(save_folder, 'checkpoints'))
else:
    print(f'{TERR}Save folder {BOLD(save_folder)} already exists. Aborting')
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
print(f'{TINFO}Using device {BOLD(device)}')

# set seed for reproducibility (https://pytorch.org/docs/stable/notes/randomness.html)
if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False  # can reduce performance

# dataset
set_range = transforms.Lambda(lambda X: 2 * X - 1.)
transform = transforms.Compose([transforms.ToTensor(), set_range])

train_dataset = CoinrunDataset(args.data_path, split='train', transform=transform, reduced=args.debug)
val_dataset = CoinrunDataset(args.data_path, split='test', transform=transform, reduced=args.debug)

num_workers = 0 if args.debug else 4
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    drop_last=True, num_workers=num_workers, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
    drop_last=True, num_workers=num_workers, pin_memory=True)

n_train, n_train_batches = len(train_dataset), len(train_dataloader)
n_val, n_val_batches = len(val_dataset), len(val_dataloader)

assert(n_train_batches == n_train // args.batch_size)
assert(n_val_batches == n_val // args.batch_size)

print(f'{TINFO}Initialized training dataset with {n_train} samples, {n_train_batches} batches')
print(f'{TINFO}Initialized validation dataset with {n_val} samples, {n_val_batches} batches')

# model
model = VAE(args).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# tensorboard logging
writer = SummaryWriter(log_dir=save_folder)

# training loop
print(TLINE)
print(TINFO + c('STARTING EXPERIMENT', 'green', attrs=['bold']))
print(TLINE)

try:
    for epoch in tqdm(range(args.epochs), desc=TINFO+BOLD('EPOCH'), colour='green'):
        # train for one epoch
        model.train()
        model.epoch = epoch
        train_losses = defaultdict(list)

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=TINFO+'(train) '+BOLD('BATCH'), colour='cyan', leave=False)):
            data = batch.to(device)
            optimizer.zero_grad()

            results = model(data)
            train_loss = model.loss(data, *results)
            for k, v in train_loss.items():
                try:
                    train_losses[k].append(v.item())
                except AttributeError:
                    train_losses[k].append(v)

            train_loss['loss'].backward()
            optimizer.step()

        # validate every args.validate_every epochs and at the last epoch
        do_validation = (args.validate_every > 0 and epoch % args.validate_every == 0) or epoch == args.epochs - 1
        if do_validation:
            with torch.no_grad():
                model.eval()
                val_losses = defaultdict(list)
                for batch_idx, batch in enumerate(tqdm(val_dataloader, desc=TINFO+'(val) '+BOLD('BATCH'), colour='blue', leave=False)):

                    data = batch.to(device)
                    results = model(data)
                    val_loss = model.loss(data, *results)
                    for k, v in val_loss.items():
                        try:
                            val_losses[k].append(v.item())
                        except AttributeError:
                            val_losses[k].append(v)

                # upload to tensorboard some validation dataset reconstructions 
                # and reconstruction from random samples in the latent space
                test_input = next(iter(val_dataloader)).to(device)
                recons = model.generate(test_input)
                grid1 = torchvision.utils.make_grid(recons.data, normalize=True, nrow=8)
                samples = model.sample(args.batch_size, device)
                grid2 = torchvision.utils.make_grid(samples.cpu().data, normalize=True, nrow=8)
                writer.add_image('reconstructions', grid1, epoch)
                writer.add_image('samples', grid2, epoch)    
                del test_input, recons, samples

        def print_data(losses, label):
            len_max_key = max([len(k) for k in losses.keys()])
            print(TLOG + BOLD(label))
            for k, v in losses.items():
                s = TLOG + '    ' + k + ' ' * (len_max_key - len(k) + 1)
                s += f'{np.mean(v):.3f} +- {np.std(v):.3f}'
                print(s)

        print()
        print_data(train_losses, 'Training')
        if do_validation:
            print_data(val_losses, 'Validation')

        if (args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0) or epoch == args.epochs - 1:
            cp_name = f'checkpoints/epoch_{epoch}.checkpoint'
            cp_path = os.path.join(save_folder, cp_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, cp_path)

            print(f'{TLOG}Saved checkpoint at {BOLD(cp_path)}')

            if args.s3:
                for path, subdirs, files in os.walk(save_folder):
                    directory_name = path.replace(args.save_folder + '/', '')
                    for file in files:
                        bucket.upload_file(os.path.join(path, file), os.path.join(args.s3_path, directory_name, file))
                s3_path = f's3://{args.s3_bucket}/{args.s3_path}/{today}/{expname}'
                print(f'{TLOG}Uploaded experiment data to {BOLD(s3_path)}')
    writer.close()
except Exception as e:
    print('\n\n')
    s_err = f'Training errored at epoch {epoch+1}/{args.epochs} with error: {e}'
    print(f'{TERR}{s_err}')
    if args.notify:
        utils.notify(message=s_err, title=args.expname)
    exit(0)

print(TLINE)
print(TINFO + c('EXPERIMENT SUCCESS', 'green', attrs=['bold']))
print(TLINE)
if args.notify:
    utils.notify(message=f'Training ended succesfully after {args.epochs} epochs', title=args.expname)