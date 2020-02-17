import argparse
import numpy as np
import torch
from utils import ODE_dataset, utils
import models
import os
from config import load_goku_train_config


def validate_goku(args, model, val_dataloader, device):
    model.eval()
    with torch.no_grad():
        for (val_batch, _, _) in val_dataloader:
            val_batch = val_batch.to(device)
            t_arr = torch.arange(0.0, end=args.seq_len * args.delta_t, step=args.delta_t, device=device)
            predicted_batch, _, _, _, _, _, _, _, _ = model(val_batch, t=t_arr, variational=False)
            val_loss = torch.abs(predicted_batch - val_batch).mean((0, 1)).sum()

    model.train()
    return val_loss


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

    ds_val = ODE_dataset.ODEDataSet(file_path=args.data_path + 'processed_data.pkl',
                                    ds_type='val',
                                    seq_len=args.seq_len,
                                    random_start=False,
                                    transforms=data_transforms)

    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=args.mini_batch_size)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=len(ds_val))

    # Create model - see models/GOKU.py for options
    model = models.__dict__["create_goku_" + args.model](ode_method=args.method).to(device)
    print('Model: GOKU - %s created with %d parameters.' % (args.model, sum(p.numel() for p in model.parameters())))

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # L1 error on validation set (not test set!) for early stopping
    best_model = models.__dict__["create_goku_" + args.model](ode_method=args.method).to(device)
    best_val_loss = np.inf

    for epoch in range(args.num_epochs):
        epoch_loss_array = []

        for i_batch, (mini_batch, latent_batch, latent_mask) in enumerate(train_dataloader):
            mini_batch = mini_batch.to(device)
            latent_batch = latent_batch.to(device)
            latent_mask = latent_mask.to(device)

            # Forward step
            t = torch.arange(0.0, end=args.seq_len * args.delta_t, step=args.delta_t, device=device)
            pred_x, pred_z, pred_params, z_0, z_0_loc, z_0_log_var, params, params_loc, params_log_var = model(
                mini_batch, t=t, variational=True)

            # Calculate loss:
            # Reconstruction loss
            loss = ((pred_x - mini_batch) ** 2).mean((0, 1)).sum()

            # KL loss
            kl_annealing_factor = utils.annealing_factor_sched(args.kl_start_af, args.kl_end_af,
                                                               args.kl_annealing_epochs, epoch, i_batch,
                                                               len(train_dataloader))
            analytic_kl_z0 = utils.normal_kl(z_0_loc, z_0_log_var,
                                             torch.zeros_like(z_0_loc),
                                             torch.zeros_like(z_0_log_var)).sum(1).mean(0)
            analytic_kl_params = utils.normal_kl(params_loc, params_log_var,
                                                 torch.zeros_like(params_loc),
                                                 torch.zeros_like(params_log_var)).sum(1).mean(0)
            loss += kl_annealing_factor * (analytic_kl_z0 + analytic_kl_params)

            # Grounding loss
            grounding_loss = ((latent_mask * (pred_z - latent_batch)) ** 2).mean((0, 1)).sum()
            loss += args.grounding_loss * grounding_loss

            # Backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            epoch_loss_array.append(loss.item())

        # Calculate validation loss
        val_loss = validate_goku(args, model, val_dataloader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model.load_state_dict(model.state_dict())

        # Mean train ELBO loss over all epoch
        epoch_mean_loss = np.mean(epoch_loss_array)

        print("[Epoch %d/%d]  loss = %.4f" % (epoch + 1, args.num_epochs, epoch_mean_loss))

    # Save model and run hyper parameters
    log_dict = {"args": args,
                "model": best_model.state_dict(),
                "data_args": torch.load(args.data_path + 'data_args.pkl')
                }
    torch.save(log_dict, args.checkpoints_dir + 'goku_model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    # Run parameters
    parser.add_argument('-n', '--num-epochs', type=int)
    parser.add_argument('-mbs', '--mini-batch-size', type=int)
    parser.add_argument('--seed', type=int, default=1)

    # Data parameters
    parser.add_argument('-sl', '--seq-len', type=int)
    parser.add_argument('--delta-t', type=float)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--norm', type=str, choices=['zscore', 'zero_to_one'], default=None)

    # Optimizer parameters
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.0)

    # Model parameters
    parser.add_argument('-m', '--method', type=str, default='rk4')
    parser.add_argument('--model', type=str, choices=['lv', 'pixel_pendulum', 'cvs', 'pixel_pendulum_friction'],
                        required=True)

    # KL Annealing factor parameters
    parser.add_argument('--kl-annealing-epochs', type=int)
    parser.add_argument('--kl-start-af', type=float)
    parser.add_argument('--kl-end-af', type=float)

    # Grounding-loss Annealing factor parameters
    parser.add_argument('--grounding-loss', type=float, default=100.0)

    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints/')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    args = load_goku_train_config(args)
    args.checkpoints_dir = args.checkpoints_dir + args.model + '/'

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    train(args)
