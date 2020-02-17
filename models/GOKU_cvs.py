import torch
import torch.nn as nn
from utils import utils
from torchdiffeq import odeint_adjoint as odeint
import numpy as np


class AbsODE(nn.Module):
    def __init__(self, ode_dim):
        super(AbsODE, self).__init__()
        self.ode_net = nn.Sequential(nn.Linear(ode_dim, 200),
                                     nn.ReLU(),
                                     nn.Linear(200, 200),
                                     nn.ReLU(),
                                     nn.Linear(200, ode_dim))

    def forward(self, t, input):
        out = self.ode_net(input)
        return out


class ODE(nn.Module):
    def __init__(self):
        super(ODE, self).__init__()
        self.ode_dim = 4
        self.known_params = {
            "f_hr_max": 3.0,
            "f_hr_min": 2.0 / 3.0,
            "r_tpr_max": 2.134,
            "r_tpr_min": 0.5335,
            "ca": 4.0,
            "cv": 111.0,
            "sv_mod": 0.0001,

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
            "cprsw_min": 25.9,
        }

    def forward(self, t, input_t):
        z_t = input_t[:, :self.ode_dim]
        params = input_t[:, self.ode_dim:]

        i_ext = params[:, 0]
        r_tpr_mod = params[:, 1]

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
        sv_mod = self.known_params["sv_mod"]

        # State variables
        p_a = 100.0 * z_t[:, 0]
        p_v = 10.0 * z_t[:, 1]
        s = z_t[:, 2]
        sv = 100.0 * z_t[:, 3]

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

        dsv_dt = i_ext * sv_mod

        dzdt = torch.stack((dpa_dt, dpv_dt, ds_dt, dsv_dt), dim=1)
        dzdt = torch.cat((dzdt, torch.zeros_like(params)), dim=1)
        return dzdt


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_input_dim, rnn_output_dim, rnn_dropout_rate, rnn_layers):
        super(Encoder, self).__init__()

        self.rnn_layers = rnn_layers
        self.input_to_rnn_net = nn.Sequential(nn.Linear(input_dim, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, rnn_input_dim))

        self.rnn = nn.RNN(input_size=rnn_input_dim, hidden_size=rnn_output_dim,
                          nonlinearity='relu', batch_first=True,
                          bidirectional=False, num_layers=self.rnn_layers, dropout=rnn_dropout_rate)

        self.rnn_to_latent_loc = nn.Linear(rnn_output_dim, latent_dim)
        self.rnn_to_latent_log_var = nn.Linear(rnn_output_dim, latent_dim)

        bidirectional = True
        lstm_output_dim = rnn_output_dim * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(input_size=rnn_input_dim, hidden_size=rnn_output_dim,
                            batch_first=True, bidirectional=bidirectional,
                            num_layers=rnn_layers, dropout=rnn_dropout_rate)

        self.lstm_to_latent_loc = nn.Linear(lstm_output_dim, latent_dim)
        self.lstm_to_latent_log_var = nn.Linear(lstm_output_dim, latent_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, mini_batch):
        mini_batch = self.input_to_rnn_net(mini_batch)

        reversed_mini_batch = utils.reverse_sequences_torch(mini_batch)

        rnn_output, _ = self.rnn(reversed_mini_batch)
        rnn_output = rnn_output[:, -1]
        latent_z_0_loc = self.rnn_to_latent_loc(rnn_output)
        latent_z_0_log_var = self.rnn_to_latent_log_var(rnn_output)

        lstm_all_output, _ = self.lstm(mini_batch)
        lstm_output = lstm_all_output[:, -1]
        latent_params_loc = self.lstm_to_latent_loc(lstm_output)
        latent_params_log_var = self.lstm_to_latent_log_var(lstm_output)

        return latent_z_0_loc, latent_z_0_log_var, latent_params_loc, latent_params_log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_method):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.ode_method = ode_method
        self.ode_solver = ODE()
        self.ode_dim = 4
        self.params_dim = 2
        self.abs_ode_dim = self.ode_dim

        # Latent vector to ODE input vector
        self.latent_to_ode_net = nn.Sequential(nn.Linear(latent_dim, 200),
                                               nn.ReLU(),
                                               nn.Linear(200, self.ode_dim),
                                               nn.Sigmoid())

        # Latent vector to ODE params
        self.latent_to_params_net = nn.Sequential(nn.Linear(latent_dim, 200),
                                                  nn.ReLU(),
                                                  nn.Linear(200, self.params_dim))

        # ODE result: z_t to reconstructed input x_t
        self.z_to_x_net = nn.Sequential(nn.Linear(self.ode_dim, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 1))

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent_batch, latent_params_batch, t):
        # ODE part
        z0_batch = self.latent_to_ode_net(latent_batch)
        sv = (z0_batch[:, 3] * 0.2 + 0.85).unsqueeze(1)
        z0_batch = torch.cat((z0_batch[:, 0:3], sv), dim=1)

        params_batch = self.latent_to_params_net(latent_params_batch)
        i_ext = self.sigmoid(params_batch[:, 0]) * -2.0
        r_tpr_mod = self.sigmoid(params_batch[:, 1]) * 0.5
        params_batch = torch.stack((i_ext, r_tpr_mod), dim=1)

        ode_init_batch = torch.cat((z0_batch, params_batch), dim=1)
        predicted_z = odeint(self.ode_solver, ode_init_batch, t, method=self.ode_method).permute(1, 0, 2)[:, :, :self.ode_dim]

        # ODE result to reconstructed  input
        predicted_x_f_hr = self.z_to_x_net(predicted_z)
        predicted_x = torch.cat((predicted_z[:, :, 0:2], predicted_x_f_hr), dim=2)

        params_batch = {"i_ext": params_batch[:, 0],
                        "r_tpr_mod": params_batch[:, 1]}

        return predicted_x, predicted_z, params_batch
