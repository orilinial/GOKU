import argparse
import numpy as np
import torch
from tqdm import tqdm, trange
from utils import ODE_dataset, utils
import os
import models
from config import load_lstm_train_config


def train(args):
    # General settings
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not args.cpu else torch.device('cpu')

    # Create train and test datasets:
    data_transforms = utils.create_transforms(args)
    ds_train = ODE_dataset.ODEDataSet(file_path=args.data_path + 'processed_data.pkl',
                                      ds_type='train',
                                      seq_len=args.seq_len,
                                      random_start=True,
                                      transforms=data_transforms)

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=args.mini_batch_size, shuffle=False)

    # Create model
    model = models.__dict__["create_lstm_" + args.model]().to(device)
    print('Model: LSTM - %s created with %d parameters.' % (args.model, sum(p.numel() for p in model.parameters())))

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loss_array = []

    mse_loss = torch.nn.MSELoss()

    for epoch in trange(args.num_epochs):
        epoch_loss_array = []

        for i_batch, (mini_batch, latent, mask) in enumerate(tqdm(train_dataloader)):
            target = mini_batch[:, -1].to(device)
            mini_batch = mini_batch[:, :-1].to(device)

            # Initialize optimizer
            optimizer.zero_grad()

            # Forward step
            outputs, _ = model(mini_batch)

            # Calculate loss:
            loss = mse_loss(outputs, target)

            # Backward step
            loss.backward()
            optimizer.step()

            epoch_loss_array.append(loss.item())

        # Mean train ELBO loss over all epoch
        epoch_mean_loss = np.mean(epoch_loss_array)
        train_loss_array.append(epoch_mean_loss)

        print("[Epoch %d/%d]  loss = %.4f"
              % (epoch + 1, args.num_epochs, epoch_mean_loss))

    # Save model
    log_dict = {"args": args,
                "model": model.state_dict(),
                "data_args": torch.load(args.data_path + 'data_args.pkl')
                }
    torch.save(log_dict, args.checkpoints_dir + 'lstm_model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    # Run parameters
    parser.add_argument('-n', '--num-epochs', type=int)
    parser.add_argument('-mbs', '--mini-batch-size', type=int)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--norm', type=str, choices=['zscore', 'zero_to_one'], default=None)

    # Data parameters
    parser.add_argument('-sl', '--seq-len', type=int)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--model', type=str, choices=['lv', 'pixel_pendulum', 'cvs', 'pixel_pendulum_friction'],
                        required=True)

    # Optimizer parameters
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001)
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.00001)

    # Model parameters
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints/')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    args = load_lstm_train_config(args)
    args.checkpoints_dir = args.checkpoints_dir + args.model + '/'

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    train(args)
