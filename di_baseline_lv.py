import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import argparse
from tqdm import trange
import torch
import torch.nn as nn
from utils.utils import set_seed
import os


class ODE(nn.Module):
    def __init__(self, reverse=False):
        self.reverse = reverse
        super(ODE, self).__init__()

    def forward(self, t, input_t):
        z_t = input_t[:, :2]
        params = input_t[:, 2:]
        a = params[:, 0]
        b = params[:, 1]
        c = params[:, 2]
        d = params[:, 3]

        dzdt = torch.zeros_like(z_t)
        dzdt[:, 0] = a * z_t[:, 0] - b * z_t[:, 0] * z_t[:, 1]
        dzdt[:, 1] = d * z_t[:, 0] * z_t[:, 1] - c * z_t[:, 1]

        reverse = -1.0 if self.reverse else 1.0
        dzdt = reverse * torch.cat((dzdt, torch.zeros_like(params)), dim=1)
        return dzdt


class ParamsModel(nn.Module):
    def __init__(self, data_size):
        super(ParamsModel, self).__init__()
        self.params = nn.Parameter(torch.ones(data_size, 4) * 1.5)
        self.ode_solver = ODE()

    def forward(self, mini_batch, t):
        z0 = mini_batch[:, 0, :]
        ode_init_batch = torch.cat((z0, self.params), dim=1)
        predicted = odeint(self.ode_solver, ode_init_batch, t, method='rk4').permute(1, 0, 2)[:, :, :2]
        return predicted


class GenerativeModel(nn.Module):
    def __init__(self, args):
        super(GenerativeModel, self).__init__()
        self.net = nn.Sequential(nn.Linear(2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, 4))

    def forward(self, x):
        return self.net(x)


def fetch_data(train):
    # Fetch data:
    train_path = 'train' if train else 'test'
    raw_data = torch.load(args.data_path + 'processed_data.pkl')
    processed_data = {}
    seq_len = 100
    new_seq_len = seq_len

    for i in range(raw_data[train_path].shape[0]):
        starting_options = np.nonzero(raw_data[train_path + "_latent_mask"][i, :raw_data[train_path + "_latent_mask"].shape[1] - seq_len].sum(1) == 2)
        starting_options = np.squeeze(starting_options)
        if starting_options.size == 0 or starting_options.size == 0:
            raise Exception('Not enough observed points in latent trajectory %d!' % i)
        if np.all(starting_options[1:] - starting_options[:-1] > 100):
            new_seq_len = max(new_seq_len, np.min(starting_options[1:] - starting_options[:-1]))

    seq_len = new_seq_len

    starting_options_dict = {}
    for i in range(raw_data[train_path].shape[0]):
        starting_options = np.nonzero(raw_data[train_path + "_latent_mask"][i,
                                      :raw_data[train_path + "_latent_mask"].shape[1] - seq_len].sum(1) == 2)
        starting_options = np.squeeze(starting_options)
        if starting_options.size == 0 or starting_options.size == 1:
            raise Exception('Not enough observed points in latent trajectory %d!' % i)

        approved_options = []
        for option in starting_options:
            if raw_data[train_path + "_latent_mask"][i, option:option + seq_len].sum() > 2:
                approved_options.append(option)
        approved_options = np.array(approved_options)
        if approved_options.size == 0:
            raise Exception('Not enough observed points in latent trajectory %d!' % i)
        else:
            starting_options_dict[i] = approved_options

    processed_data["latent_mask"] = raw_data[train_path + "_latent_mask"]
    processed_data["latent_batch"] = raw_data[train_path + "_latent"]
    processed_data["data_batch"] = raw_data[train_path]

    # gt data:
    processed_data["gt_latent_data"] = torch.load(args.data_path + train_path + '_latent_data.pkl')
    gt_params_data_dict = torch.load(args.data_path + train_path + '_params_data.pkl')

    processed_data["gt_params_data"] = np.array([val for key, val in gt_params_data_dict.items()]).transpose()
    processed_data["starting_options_dict"] = starting_options_dict

    return processed_data


def train_params(args, data):
    # General settings
    data_size = data["latent_mask"].shape[0]

    model = ParamsModel(data_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    delta_t = 0.05
    seq_len = 100
    t = torch.arange(0.0, end=seq_len * delta_t, step=delta_t)

    for epoch in range(args.num_epochs):
        mini_batch = torch.zeros((data_size, seq_len, data["latent_mask"].shape[2]))
        latent_mask = torch.zeros((data_size, seq_len, data["latent_mask"].shape[2]))
        gt_latent_data = np.zeros((data_size, seq_len, data["latent_mask"].shape[2]))
        for i in range(data_size):
            starting_t = np.random.choice(data["starting_options_dict"][i])
            mini_batch[i] = torch.tensor(data["latent_batch"][i, starting_t:starting_t + seq_len, :])
            latent_mask[i] = torch.tensor(data["latent_mask"][i, starting_t:starting_t + seq_len, :])
            gt_latent_data[i] = data["gt_latent_data"][i, starting_t:starting_t + seq_len, :]

        predicted = model(mini_batch, t)
        loss = ((latent_mask * (predicted - mini_batch)) ** 2.0).mean((0, 1)).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        latent_error = np.abs(predicted.detach().numpy() - gt_latent_data).mean((0, 1)).sum()

        params = model.params.detach().numpy()
        ab_ratio = params[:, 0]/params[:, 1] - data["gt_params_data"][:, 0]/data["gt_params_data"][:, 1]
        cd_ratio = params[:, 2]/params[:, 3] - data["gt_params_data"][:, 2]/data["gt_params_data"][:, 3]
        params_error = np.sqrt((ab_ratio ** 2 + cd_ratio ** 2).mean())

        print("Epoch: %d/%d, loss = %.3f, params_error = %.3f" %
              (epoch+1, args.num_epochs, latent_error, params_error))

    return model.params.detach()


def create_z0(args, data, params):
    data_size = data["latent_mask"].shape[0]
    print(data_size)

    ode_solver_reverse = ODE(reverse=True)
    ode_solver = ODE(reverse=False)
    seq_len = 400
    delta_t = 0.05

    t = torch.arange(0.0, end=seq_len * delta_t, step=delta_t)

    predicted_z = torch.zeros((data_size, seq_len, 2))
    for sample in trange(data_size):
        first_known_idx = np.nonzero(data["latent_mask"][sample, :, 0])[0][0]
        z_known = torch.FloatTensor(data["latent_batch"][sample, first_known_idx]).unsqueeze(0)
        params_sample = params[sample].unsqueeze(0)
        ode_init_batch = torch.cat((z_known, params_sample), dim=1)

        t_reversed = torch.arange(0.0, end=(first_known_idx + 1) * delta_t, step=delta_t)
        predicted_z0 = odeint(ode_solver_reverse, ode_init_batch, t_reversed, method='rk4').permute(1, 0, 2)[0, -1, :2].unsqueeze(0)

        ode_init_batch = torch.cat((predicted_z0, params_sample), dim=1)
        predicted_z[sample] = odeint(ode_solver, ode_init_batch, t, method='rk4').permute(1, 0, 2)[0, :, :2]

    return predicted_z.detach()


def train_generative(args, train, z_train, z_test):
    nan_ids = torch.nonzero(torch.isnan(z_train))[:, 0].unique()
    not_nan_ids = []

    for i in range(z_train.size(0)):
        if i in nan_ids:
            continue
        not_nan_ids.append(i)

    filtered_z_train = z_train[not_nan_ids]

    z_train = filtered_z_train

    z_train = z_train.view(z_train.size(0) * z_train.size(1), 2)

    model = GenerativeModel(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    targets = torch.FloatTensor(train["data_batch"])[not_nan_ids]
    targets = targets.view(targets.size(0) * targets.size(1), 4)

    for epoch in range(args.num_epochs):
        predicted_x = model(z_train)
        loss = ((predicted_x - targets) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch: %d/%d, loss = %.3f" % (epoch+1, args.num_epochs, loss.item()))
        torch.save(model.state_dict(), args.checkpoints_dir + 'di_baseline_generative.pkl')

    with torch.no_grad():
        predicted_x = model(z_test)

    return predicted_x.detach()


if __name__ == '__main__':
    # Architecture names
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data-path', type=str, default='data/lv/')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints/lv/di/')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    set_seed(args.seed)

    train_data = fetch_data(train=True)
    test_data = fetch_data(train=False)

    params_batch_train = train_params(args, train_data)
    torch.save(params_batch_train, args.checkpoints_dir + 'di_baseline_params_train.pkl')

    params_batch_test = train_params(args, test_data)
    torch.save(params_batch_test, args.checkpoints_dir + 'di_baseline_params_test.pkl')

    predicted_z_train = create_z0(args, train_data, params_batch_train)
    torch.save(predicted_z_train, args.checkpoints_dir + 'di_baseline_z_train.pkl')

    predicted_z_test = create_z0(args, test_data, params_batch_test)
    torch.save(predicted_z_test, args.checkpoints_dir + 'di_baseline_z_test.pkl')

    predicted_x_test = train_generative(args, train_data, predicted_z_train, predicted_z_test)
    torch.save(predicted_x_test, args.checkpoints_dir + 'di_baseline_x_test.pkl')
