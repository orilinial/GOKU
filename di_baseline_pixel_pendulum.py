import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import argparse
import random
import torch
import torch.nn as nn
import os


class ODE(nn.Module):
    def __init__(self):
        super(ODE, self).__init__()

    def forward(self, t, input_t):
        # Setting ODE parameters:
        G = 10.0

        z_t = input_t[:, :2]
        params = input_t[:, 2:]

        L = params[:, 0]

        dxdt = torch.zeros_like(z_t)
        dxdt[:, 0] = z_t[:, 1]
        dxdt[:, 1] = -G/L * torch.sin(z_t[:, 0])

        dxdt = torch.cat((dxdt, torch.zeros_like(params)), dim=1)
        return dxdt


class ParamsModel(nn.Module):
    def __init__(self, data_size):
        super(ParamsModel, self).__init__()
        self.params = nn.Parameter(torch.ones(data_size, 1))
        self.z0 = nn.Parameter(torch.zeros(data_size, 2))
        self.ode_solver = ODE()

    def forward(self, mini_batch, t):
        ode_init_batch = torch.cat((self.z0, self.params), dim=1)
        predicted = odeint(self.ode_solver, ode_init_batch, t, method='rk4').permute(1, 0, 2)[:, :, :2]
        return predicted


class GenerativeModel(nn.Module):
    def __init__(self, input_dim):
        super(GenerativeModel, self).__init__()
        self.input_dim = input_dim
        self.ode_dim = 2

        # ODE result: z_t to reconstructed input x_t
        self.first_layer = nn.Linear(self.ode_dim, 200)
        self.second_layer = nn.Linear(200, 200)
        self.third_layer = nn.Linear(200, 200)
        self.fourth_layer = nn.Linear(200, input_dim[0] * input_dim[1])

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_batch):
        recon_batch = self.relu(self.first_layer(latent_batch))
        recon_batch = recon_batch + self.relu(self.second_layer(recon_batch))
        recon_batch = recon_batch + self.relu(self.third_layer(recon_batch))
        recon_batch = self.sigmoid(self.fourth_layer(recon_batch))
        recon_batch = recon_batch.view(latent_batch.size(0), latent_batch.size(1), self.input_dim[0], self.input_dim[1])
        return recon_batch


def set_seed(seed, fully_deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            torch.backends.cudnn.deterministic = True


def fetch_data(train):
    # Fetch data:
    train_path = 'train' if train else 'test'
    raw_data = torch.load(args.data_path + 'processed_data.pkl')
    processed_data = {}

    processed_data["latent_mask"] = torch.FloatTensor(raw_data[train_path + "_latent_mask"])
    processed_data["latent_batch"] = torch.FloatTensor(raw_data[train_path + "_latent"])
    processed_data["data_batch"] = torch.FloatTensor(raw_data[train_path])

    # gt data:
    processed_data["gt_latent_data"] = torch.load(args.data_path + train_path + '_latent_data.pkl')
    gt_params_data_dict = torch.load(args.data_path + train_path + '_params_data.pkl')

    processed_data["gt_params_data"] = np.array([val for key, val in gt_params_data_dict.items()]).transpose()
    return processed_data


def train_params(args, data, train):
    # General settings
    data_size = data["latent_mask"].shape[0]

    model = ParamsModel(data_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    delta_t = 0.05
    seq_len = 100
    t = torch.arange(0.0, end=seq_len * delta_t, step=delta_t)
    latent_batch = data["latent_batch"][:, :seq_len]
    gt_latent_data = data["gt_latent_data"][:, :seq_len]
    latent_mask = data["latent_mask"][:, :seq_len]

    for epoch in range(args.num_epochs):
        predicted = model(latent_batch, t)
        loss = ((latent_mask * (predicted - latent_batch)) ** 2.0).mean((0, 1)).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        latent_error = np.abs(predicted.detach().numpy() - gt_latent_data).mean((0, 1)).sum()

        params = model.params.detach().numpy()
        params_error = np.abs(data["gt_params_data"] - params).mean()

        print("Epoch: %d/%d, loss = %.3f, params_error = %.3f" %
              (epoch+1, args.num_epochs, latent_error, params_error))

    t = torch.arange(0.0, end=data["latent_mask"].shape[1] * delta_t, step=delta_t)
    predicted_z = model(data["latent_batch"], t)

    return model.params.detach(), predicted_z.detach()


def train_generative(args, train, test, z_train, z_test):
    model = GenerativeModel(input_dim=[28, 28])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    targets = torch.FloatTensor(train["data_batch"])

    num_epochs = 50
    for epoch in range(num_epochs):
        predicted_x = model(z_train)
        loss = torch.abs(predicted_x - targets).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch: %d/%d, loss = %.5f" % (epoch+1, num_epochs, loss.item()))
        torch.save(model.state_dict(), 'baseline_generative.pkl')

    import imageio
    images = []
    predicted_x = (predicted_x / predicted_x.max()) * 255
    for i in range(predicted_x.shape[1]):
        image = predicted_x[0, i].detach().numpy().astype(np.uint8)
        images.append(image)
    imageio.mimsave('example_gif.gif', images)

    with torch.no_grad():
        predicted_x = model(z_test)

    return predicted_x.detach()


if __name__ == '__main__':
    # Architecture names
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--data-path', type=str, default='data/pixel_pendulum/')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints/lv/di/')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    set_seed(args.seed)

    train_data = fetch_data(train=True)
    test_data = fetch_data(train=False)

    params_batch_train, predicted_z_train = train_params(args, train_data, train=True)
    torch.save(params_batch_train, args.checkpoints_node + 'di_baseline_params_train.pkl')
    torch.save(predicted_z_train, args.checkpoints_node + 'baseline_z_train.pkl')

    params_batch_test, predicted_z_test = train_params(args, test_data, train=False)
    torch.save(params_batch_test, args.checkpoints_node + 'di_baseline_params_test.pkl')
    torch.save(predicted_z_test, args.checkpoints_node + 'di_baseline_z_test.pkl')

    predicted_x_test = train_generative(args, train_data, test_data, predicted_z_train, predicted_z_test)
    torch.save(predicted_x_test, args.checkpoints_node + 'di_baseline_x_test.pkl')
