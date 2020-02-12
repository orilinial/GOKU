import torch.nn as nn
import torch
from utils import utils
from torchdiffeq import odeint_adjoint as odeint


class ODE(nn.Module):
    def __init__(self, ode_dim):
        super(ODE, self).__init__()
        self.ode_net = nn.Sequential(nn.Linear(ode_dim, 200),
                                     nn.ReLU(),
                                     nn.Linear(200, 200),
                                     nn.ReLU(),
                                     nn.Linear(200, ode_dim))

    def forward(self, t, z_t):
        return self.ode_net(z_t)


class EncoderLV(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_input_dim, rnn_output_dim, rnn_dropout, rnn_layers):
        super(EncoderLV, self).__init__()
        self.input_to_rnn_net = nn.Sequential(nn.Linear(input_dim, 200),
                                              nn.ReLU(),
                                              nn.Linear(200, rnn_input_dim),
                                              nn.ReLU())

        self.rnn = nn.RNN(input_size=rnn_input_dim, hidden_size=rnn_output_dim,
                          nonlinearity='relu', batch_first=True,
                          bidirectional=False, num_layers=rnn_layers, dropout=rnn_dropout)

        self.rnn_to_latent_loc = nn.Linear(rnn_output_dim, latent_dim)
        self.rnn_to_latent_log_var = nn.Linear(rnn_output_dim, latent_dim)

    def forward(self, mini_batch):
        mini_batch = self.input_to_rnn_net(mini_batch)

        reversed_mini_batch = utils.reverse_sequences_torch(mini_batch)
        rnn_output, _ = self.rnn(reversed_mini_batch)
        rnn_output = rnn_output[:, -1]

        z_0_loc = self.rnn_to_latent_loc(rnn_output)
        z_0_log_var = self.rnn_to_latent_log_var(rnn_output)
        return z_0_loc, z_0_log_var


class DecoderLV(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_method):
        super(DecoderLV, self).__init__()
        self.ode_method = ode_method
        self.ode_solver = ODE(latent_dim)

        self.generative_net = nn.Sequential(nn.Linear(latent_dim, 200),
                                            nn.ReLU(),
                                            nn.Linear(200, input_dim))

    def forward(self, z_0, t):
        predicted_z = odeint(self.ode_solver, z_0, t, method=self.ode_method).permute(1, 0, 2)
        predicted_batch = self.generative_net(predicted_z)

        return predicted_batch, predicted_z


class EncoderPixelPendulum(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_input_dim, rnn_output_dim, rnn_dropout, rnn_layers):
        super(EncoderPixelPendulum, self).__init__()
        self.first_layer = nn.Linear(input_dim[0] * input_dim[1], 200)
        self.second_layer = nn.Linear(200, 200)
        self.third_layer = nn.Linear(200, 200)
        self.fourth_layer = nn.Linear(200, rnn_input_dim)

        self.relu = nn.ReLU()

        self.rnn = nn.RNN(input_size=rnn_input_dim, hidden_size=rnn_output_dim,
                          nonlinearity='relu', batch_first=True,
                          bidirectional=False, num_layers=rnn_layers, dropout=rnn_dropout)

        self.rnn_to_latent_loc = nn.Linear(rnn_output_dim, latent_dim)
        self.rnn_to_latent_log_var = nn.Linear(rnn_output_dim, latent_dim)

    def forward(self, mini_batch):
        mini_batch = mini_batch.view(mini_batch.size(0), mini_batch.size(1), mini_batch.size(2) * mini_batch.size(3))
        mini_batch = self.relu(self.first_layer(mini_batch))
        mini_batch = mini_batch + self.relu(self.second_layer(mini_batch))
        mini_batch = mini_batch + self.relu(self.third_layer(mini_batch))
        mini_batch = self.relu(self.fourth_layer(mini_batch))

        reversed_mini_batch = utils.reverse_sequences_torch(mini_batch)
        rnn_output, _ = self.rnn(reversed_mini_batch)
        rnn_output = rnn_output[:, -1]

        z_0_loc = self.rnn_to_latent_loc(rnn_output)
        z_0_log_var = self.rnn_to_latent_log_var(rnn_output)
        return z_0_loc, z_0_log_var


class DecoderPixelPendulum(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_method):
        super(DecoderPixelPendulum, self).__init__()
        self.input_dim = input_dim
        self.ode_method = ode_method
        self.ode_solver = ODE(latent_dim)

        self.first_layer = nn.Linear(latent_dim, 200)
        self.second_layer = nn.Linear(200, 200)
        self.third_layer = nn.Linear(200, 200)
        self.fourth_layer = nn.Linear(200, input_dim[0] * input_dim[1])

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_0, t):
        predicted_z = odeint(self.ode_solver, z_0, t, method=self.ode_method).permute(1, 0, 2)

        recon_batch = self.relu(self.first_layer(predicted_z))
        recon_batch = recon_batch + self.relu(self.second_layer(recon_batch))
        recon_batch = recon_batch + self.relu(self.third_layer(recon_batch))
        recon_batch = self.sigmoid(self.fourth_layer(recon_batch))
        recon_batch = recon_batch.view(predicted_z.size(0), predicted_z.size(1), self.input_dim[0], self.input_dim[1])

        return recon_batch, predicted_z


class EncoderCVS(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_input_dim, rnn_output_dim, rnn_dropout, rnn_layers):
        super(EncoderCVS, self).__init__()
        self.input_to_rnn_net = nn.Sequential(nn.Linear(input_dim, 200),
                                              nn.ReLU(),
                                              nn.Linear(200, rnn_input_dim),
                                              nn.ReLU())

        self.rnn = nn.RNN(input_size=rnn_input_dim, hidden_size=rnn_output_dim,
                          nonlinearity='relu', batch_first=True,
                          bidirectional=False, num_layers=rnn_layers, dropout=rnn_dropout)

        self.rnn_to_latent_loc = nn.Linear(rnn_output_dim, latent_dim)
        self.rnn_to_latent_log_var = nn.Linear(rnn_output_dim, latent_dim)

    def forward(self, mini_batch):
        mini_batch = self.input_to_rnn_net(mini_batch)

        reversed_mini_batch = utils.reverse_sequences_torch(mini_batch)
        rnn_output, _ = self.rnn(reversed_mini_batch)
        rnn_output = rnn_output[:, -1]

        z_0_loc = self.rnn_to_latent_loc(rnn_output)
        z_0_log_var = self.rnn_to_latent_log_var(rnn_output)
        return z_0_loc, z_0_log_var


class DecoderCVS(nn.Module):
    def __init__(self, input_dim, latent_dim, ode_method):
        super(DecoderCVS, self).__init__()
        self.ode_method = ode_method
        self.ode_solver = ODE(latent_dim)

        self.generative_net = nn.Sequential(nn.Linear(latent_dim, 200),
                                            nn.ReLU(),
                                            nn.Linear(200, input_dim))

    def forward(self, z_0, t):
        predicted_z = odeint(self.ode_solver, z_0, t, method=self.ode_method).permute(1, 0, 2)
        predicted_batch = self.generative_net(predicted_z)

        return predicted_batch, predicted_z


class LatentODE(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_input_dim, rnn_output_dim, ode_method, rnn_dropout, rnn_layers, encoder, decoder):
        super(LatentODE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, rnn_dropout, rnn_layers)
        self.decoder = decoder(input_dim, latent_dim, ode_method)

    def forward(self, mini_batch, t, variational=True):
        z_0_loc, z_0_log_var = self.encoder(mini_batch)

        if variational:
            z_0 = torch.distributions.normal.Normal(z_0_loc, torch.exp(z_0_log_var / 2.0)).rsample()

        else:
            z_0 = z_0_loc

        predicted_batch, predicted_z = self.decoder(z_0, t)

        return predicted_batch, predicted_z, z_0, z_0_loc, z_0_log_var


def create_latent_ode_lv(input_dim=4, latent_dim=4, rnn_input_dim=32, rnn_output_dim=32, ode_method='rk4', rnn_dropout=0.0, rnn_layers=2):
    return LatentODE(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, ode_method, rnn_dropout, rnn_layers, encoder=EncoderLV, decoder=DecoderLV)


def create_latent_ode_pixel_pendulum(input_dim=[28, 28], latent_dim=16, rnn_input_dim=32, rnn_output_dim=32, ode_method='rk4', rnn_dropout=0.0, rnn_layers=2):
    return LatentODE(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, ode_method, rnn_dropout, rnn_layers, encoder=EncoderPixelPendulum, decoder=DecoderPixelPendulum)


def create_latent_ode_cvs(input_dim=3, latent_dim=16, rnn_input_dim=32, rnn_output_dim=32, ode_method='rk4', rnn_dropout=0.0, rnn_layers=2):
    return LatentODE(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, ode_method, rnn_dropout, rnn_layers, encoder=EncoderCVS, decoder=DecoderCVS)
