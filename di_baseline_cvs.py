import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import argparse
import random
import torch
import torch.nn as nn
import os


class ODE(nn.Module):
    def __init__(self, reverse=False):
        super(ODE, self).__init__()
        self.reverse = reverse
        self.ode_dim = 4  # FIXME
        self.known_params = {
            "f_hr_max": 3.0,
            "f_hr_min": 2.0 / 3.0,
            "r_tpr_max": 2.134,
            "r_tpr_min": 0.5335,
            "ca": 4.0,
            "cv": 111.0,

            # dS/dt parameters
            "k_width": 0.1838,
            "p_aset": 70,
            "tau": 20,
            "p_0lv": 2.03,
            "r_valve": 0.0025,
            "k_elv": 0.066,
            "v_ed0": 7.14,
            "T_sys": 4. / 15.,
            "cprsw_max": 103.8,
            "cprsw_min": 25.9
        }

    def forward(self, t, input_t):
        z_t = input_t[:, :self.ode_dim]
        params = input_t[:, self.ode_dim:]

        i_ext = params[:, 0]
        r_tpr_mod = params[:, 1]
        sv_mod = 0.0001

        # Parameters:
        # dV/dt parameters
        f_hr_max = self.known_params["f_hr_max"]
        f_hr_min = self.known_params["f_hr_min"]
        r_tpr_max = self.known_params["r_tpr_max"]
        r_tpr_min = self.known_params["r_tpr_min"]
        ca = self.known_params["ca"]
        cv = self.known_params["cv"]
        k_width = self.known_params["k_width"]
        p_aset = self.known_params["p_aset"]
        tau = self.known_params["tau"]

        # State variables
        p_a = 100.0 * z_t[:, 0]
        p_v = 10.0 * z_t[:, 1]
        s = z_t[:, 2]
        sv = 100.0 * z_t[:, 3]  # FIXME

        # Building f_hr and r_tpr:
        f_hr = s * (f_hr_max - f_hr_min) + f_hr_min
        r_tpr = s * (r_tpr_max - r_tpr_min) + r_tpr_min - r_tpr_mod

        # Building dp_a/dt and dp_v/dt:
        dva_dt = -1. * (p_a - p_v) / r_tpr + sv * f_hr
        dvv_dt = -1. * dva_dt + i_ext
        dpa_dt = dva_dt / (ca * 100.)
        dpv_dt = dvv_dt / (cv * 10.)

        # Building dS/dt:
        ds_dt = (1./tau) * (1. - 1./(1 + torch.exp(-1 * k_width * (p_a - p_aset))) - s)

        dsv_dt = i_ext * sv_mod  # FIXME

        dzdt = torch.stack((dpa_dt, dpv_dt, ds_dt, dsv_dt), dim=1)  # FIXME

        reverse = -1.0 if self.reverse else 1.0
        dzdt = reverse * torch.cat((dzdt, torch.zeros_like(params)), dim=1)

        return dzdt


class ParamsModel(nn.Module):
    def __init__(self, data_size):
        super(ParamsModel, self).__init__()
        self.params = nn.Parameter(torch.zeros(data_size, 2))
        self.ode_solver = ODE()

    def forward(self, mini_batch, t):
        z0 = mini_batch[:, 0, :]
        ode_init_batch = torch.cat((z0, self.params), dim=1)
        predicted = odeint(self.ode_solver, ode_init_batch, t, method='rk4').permute(1, 0, 2)[:, :, :4]
        return predicted


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
    seq_len = 100
    new_seq_len = seq_len

    for i in range(raw_data[train_path].shape[0]):
        starting_options = np.nonzero(raw_data[train_path + "_latent_mask"][i, :raw_data[train_path + "_latent_mask"].shape[1] - seq_len].sum(1) == 4)
        starting_options = np.squeeze(starting_options)
        if starting_options.size == 0 or starting_options.size == 0:
            raise Exception('No observed points at all at %d!' % i)

        if np.all(starting_options[1:] - starting_options[:-1] > 100):
            new_seq_len = max(new_seq_len, np.min(starting_options[1:] - starting_options[:-1]))

    seq_len = new_seq_len

    starting_options_dict = {}
    for i in range(raw_data[train_path].shape[0]):
        starting_options = np.nonzero(raw_data[train_path + "_latent_mask"][i,
                                      :raw_data[train_path + "_latent_mask"].shape[1] - seq_len].sum(1) == 4)
        starting_options = np.squeeze(starting_options)
        if starting_options.size == 0 or starting_options.size == 1:
            raise Exception('Sequence length absorbed all points! %d' % i)

        approved_options = []
        for option in starting_options:
            if raw_data[train_path + "_latent_mask"][i, option:option + seq_len].sum() > 4:
                approved_options.append(option)
        approved_options = np.array(approved_options)
        if approved_options.size == 0:
            raise Exception('No good points!')
        else:
            starting_options_dict[i] = approved_options

    processed_data["latent_mask"] = raw_data[train_path + "_latent_mask"]
    processed_data["latent_batch"] = raw_data[train_path + "_latent"]
    processed_data["data_batch"] = raw_data[train_path]

    # gt data:
    processed_data["gt_latent_data"] = torch.load(args.data_path + train_path + '_latent_data.pkl')
    gt_params_data_dict = torch.load(args.data_path + train_path + '_params_data.pkl')

    allowed = ['i_ext', 'r_tpr_mod']
    processed_data["gt_params_data"] = np.array([val for key, val in gt_params_data_dict.items() if key in allowed]).transpose()
    processed_data["starting_options_dict"] = starting_options_dict

    return processed_data


def train_params(args, data):
    # General settings
    data_size = data["latent_mask"].shape[0]

    model = ParamsModel(data_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    delta_t = 1.0
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
        params_error = np.abs(data["gt_params_data"] - params).mean()

        print("Epoch: %d/%d, loss = %.3f, params_error = %.3f" %
              (epoch+1, args.num_epochs, latent_error, params_error))

    return model.params.detach()


def create_z0(args, data, params):
    data_size = data["latent_mask"].shape[0]

    ode_solver_reverse = ODE(reverse=True)
    ode_solver = ODE(reverse=False)
    seq_len = 400
    delta_t = 1.0

    t = torch.arange(0.0, end=seq_len * delta_t, step=delta_t)

    predicted_z = torch.zeros((data_size, seq_len, 4))
    for sample in range(data_size):
        first_known_idx = np.nonzero(data["latent_mask"][sample, :, 0])[0][0]
        params_sample = params[sample].unsqueeze(0)

        if first_known_idx != 0:
            z_known = torch.FloatTensor(data["latent_batch"][sample, first_known_idx]).unsqueeze(0)
            ode_init_batch = torch.cat((z_known, params_sample), dim=1)
            t_reversed = torch.arange(0.0, end=(first_known_idx + 1) * delta_t, step=delta_t)
            predicted_z0 = odeint(ode_solver_reverse, ode_init_batch, t_reversed, method='rk4').permute(1, 0, 2)[0, -1, :4].unsqueeze(0)

        else:
            predicted_z0 = torch.FloatTensor(data["latent_batch"][sample, 0]).unsqueeze(0)

        ode_init_batch = torch.cat((predicted_z0, params_sample), dim=1)
        predicted_z[sample] = odeint(ode_solver, ode_init_batch, t, method='rk4').permute(1, 0, 2)[0, :, :4]

    return predicted_z.detach()


if __name__ == '__main__':
    # Architecture names
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data-path', type=str, default='data/cvs/')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints/cvs/di/')
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

    predicted_x = torch.cat((predicted_z_test[:, :, :2], predicted_z_test[:, :, 2].unsqueeze(2) * (3. - 2./3.) + 2./3.), dim=2)
    torch.save(predicted_x.detach(), args.checkpoints_dir + 'di_baseline_x_test.pkl')
