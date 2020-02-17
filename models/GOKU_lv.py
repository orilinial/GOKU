import torch
import torch.nn as nn
from utils import utils
from torchdiffeq import odeint_adjoint as odeint


class ODE(nn.Module):
    def __init__(self):
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

        dzdt = torch.cat((dzdt, torch.zeros_like(params)), dim=1)
        return dzdt


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_input_dim, rnn_output_dim, rnn_dropout_rate, rnn_layers):
        super(Encoder, self).__init__()

        self.rnn_layers = rnn_layers
        self.input_to_rnn_net = nn.Sequential(nn.Linear(input_dim, 200),
                                              nn.ReLU(),
                                              nn.Linear(200, rnn_input_dim))

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
        self.ode_dim = 2
        self.params_dim = 4

        # Latent vector to ODE input vector
        self.latent_to_z0_net = nn.Sequential(nn.Linear(latent_dim, 200),
                                              nn.ReLU(),
                                              nn.Linear(200, self.ode_dim),
                                              nn.Softplus())

        # Latent vector to ODE params
        self.latent_to_params_net = nn.Sequential(nn.Linear(latent_dim, 200),
                                                  nn.ReLU(),
                                                  nn.Linear(200, self.params_dim),
                                                  nn.Softplus())

        # ODE result: z_t to reconstructed input x_t
        self.generative_net = nn.Sequential(nn.Linear(self.ode_dim, 200),
                                            nn.ReLU(),
                                            nn.Linear(200, input_dim))

        # Activations
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, latent_batch, latent_params_batch, t):
        # Latent to ODE
        z0_batch = self.latent_to_z0_net(latent_batch)

        # latent_batch to params
        params_batch = self.latent_to_params_net(latent_params_batch)

        ode_init_batch = torch.cat((z0_batch, params_batch), dim=1)

        # ODE solution at any time in t
        predicted_z = odeint(self.ode_solver, ode_init_batch, t, method=self.ode_method).permute(1, 0, 2)[:, :, :self.ode_dim]

        # ODE result to reconstructed / predicted input
        predicted_x = self.generative_net(predicted_z)

        params_batch = {"a": params_batch[:, 0],
                        "b": params_batch[:, 1],
                        "c": params_batch[:, 2],
                        "d": params_batch[:, 3]}

        return predicted_x, predicted_z, params_batch
