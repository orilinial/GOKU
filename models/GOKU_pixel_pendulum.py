import torch.nn as nn
import torch
from torchdiffeq import odeint_adjoint as odeint
# from ODE_funcs import Pendulum as ODE
from utils import utils


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


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_input_dim, rnn_output_dim, rnn_dropout_rate, rnn_layers):
        super(Encoder, self).__init__()
        self.first_layer = nn.Linear(input_dim[0] * input_dim[1], 200)
        self.second_layer = nn.Linear(200, 200)
        self.third_layer = nn.Linear(200, 200)
        self.fourth_layer = nn.Linear(200, rnn_input_dim)

        self.relu = nn.ReLU()

        self.rnn_layers = rnn_layers

        self.rnn = nn.RNN(input_size=rnn_input_dim, hidden_size=rnn_output_dim,
                          nonlinearity='relu', batch_first=True,
                          bidirectional=False, num_layers=rnn_layers, dropout=rnn_dropout_rate)

        bidirectional = True
        lstm_output_dim = rnn_output_dim * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(input_size=rnn_input_dim, hidden_size=rnn_output_dim,
                            batch_first=True, bidirectional=bidirectional,
                            num_layers=rnn_layers, dropout=rnn_dropout_rate)

        self.rnn_to_z0_loc = nn.Linear(rnn_output_dim, latent_dim)
        self.rnn_to_z0_log_var = nn.Linear(rnn_output_dim, latent_dim)

        self.lstm_to_latent_loc = nn.Linear(lstm_output_dim, latent_dim)
        self.lstm_to_latent_log_var = nn.Linear(lstm_output_dim, latent_dim)

    def forward(self, input_batch):
        # Create data batch for the RNN
        out = input_batch.view(input_batch.size(0), input_batch.size(1), input_batch.size(2) * input_batch.size(3))
        out = self.relu(self.first_layer(out))
        out = out + self.relu(self.second_layer(out))
        out = out + self.relu(self.third_layer(out))
        out = self.relu(self.fourth_layer(out))

        # RNN consumes batch backwards to create z0
        reversed_mini_batch = utils.reverse_sequences_torch(out)
        h0 = torch.zeros(self.rnn_layers, input_batch.size(0), self.rnn.hidden_size, device=input_batch.device)
        rnn_output, _ = self.rnn(reversed_mini_batch, h0)
        rnn_output = rnn_output[:, -1]
        z_0_loc = self.rnn_to_z0_loc(rnn_output)
        z_0_log_var = self.rnn_to_z0_log_var(rnn_output)

        # LSTM creates params
        lstm_all_output, _ = self.lstm(out)
        lstm_output = lstm_all_output[:, -1]
        latent_params_loc = self.lstm_to_latent_loc(lstm_output)
        latent_params_log_var = self.lstm_to_latent_log_var(lstm_output)

        return z_0_loc, z_0_log_var, latent_params_loc, latent_params_log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_method):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.ode_method = ode_method
        self.ode_solver = ODE()
        self.ode_dim = 2
        self.params_dim = 2

        # Latent vector to ODE input vector
        self.latent_to_hidden_z0 = nn.Linear(latent_dim, 200)
        self.hidden_to_ode = nn.Linear(200, self.ode_dim)

        # Latent vector to ODE params
        self.latent_to_hidden_params = nn.Linear(latent_dim, 200)
        self.hidden_to_params = nn.Linear(200, self.params_dim)

        # ODE result: z_t to reconstructed input x_t
        self.first_layer = nn.Linear(self.ode_dim, 200)
        self.second_layer = nn.Linear(200, 200)
        self.third_layer = nn.Linear(200, 200)
        self.fourth_layer = nn.Linear(200, input_dim[0] * input_dim[1])

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, latent_batch, latent_params_batch, t):
        # Latent to ODE
        z0_batch = self.relu(self.latent_to_hidden_z0(latent_batch))
        z0_batch = self.hidden_to_ode(z0_batch)

        # latent_batch to params
        params_batch = self.relu(self.latent_to_hidden_params(latent_params_batch))
        # params_batch = self.sigmoid(self.hidden_to_params(params_batch)) + 0.5
        params_batch = self.softplus(self.hidden_to_params(params_batch))

        ode_init_batch = torch.cat((z0_batch, params_batch), dim=1)

        # ODE solution at any time in t
        predicted_z = odeint(self.ode_solver, ode_init_batch, t, method=self.ode_method).permute(1, 0, 2)[:, :, :self.ode_dim]

        recon_batch = self.relu(self.first_layer(predicted_z))
        recon_batch = recon_batch + self.relu(self.second_layer(recon_batch))
        recon_batch = recon_batch + self.relu(self.third_layer(recon_batch))
        recon_batch = self.sigmoid(self.fourth_layer(recon_batch))
        recon_batch = recon_batch.view(predicted_z.size(0), predicted_z.size(1), self.input_dim[0], self.input_dim[1])

        params_batch = {"l": params_batch[:, 0]}

        return recon_batch, predicted_z, params_batch
