import models.GOKU_pendulum_friction as GOKU_pendulum_friction
import models.GOKU_pendulum as GOKU_pendulum
import models.GOKU_double_pendulum as GOKU_double_pendulum
import models.GOKU_cvs as GOKU_cvs
import torch.nn as nn
import torch


class GOKU(nn.Module):
    def __init__(self, input_dim, latent_dim, rnn_input_dim, rnn_output_dim, ode_method, rnn_dropout, rnn_layers, encoder, decoder):
        super(GOKU, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, rnn_dropout, rnn_layers)
        self.decoder = decoder(input_dim, latent_dim, ode_method)

    def forward(self, mini_batch, t, variational=True):
        latent_z_0_loc, latent_z_0_log_var, latent_params_loc, latent_params_log_var = self.encoder(mini_batch)

        if variational:
            latent_z_0 = torch.distributions.normal.Normal(latent_z_0_loc, torch.exp(latent_z_0_log_var / 2.0)).rsample()
            latent_params = torch.distributions.normal.Normal(latent_params_loc, torch.exp(latent_params_log_var / 2.0)).rsample()

        else:
            latent_z_0 = latent_z_0_loc
            latent_params = latent_params_loc

        predicted_batch, predicted_z, predicted_params = self.decoder(latent_z_0, latent_params, t)

        return predicted_batch, predicted_z, predicted_params, latent_z_0, latent_z_0_loc, latent_z_0_log_var, latent_params, latent_params_loc, latent_params_log_var


def create_goku_pendulum(input_dim=[28, 28], latent_dim=16, rnn_input_dim=32, rnn_output_dim=16, ode_method='rk4', rnn_dropout=0.0, rnn_layers=2):
    return GOKU(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, ode_method, rnn_dropout, rnn_layers, GOKU_pendulum.Encoder, GOKU_pendulum.Decoder)


def create_goku_pendulum_friction(input_dim=[28, 28], latent_dim=16, rnn_input_dim=32, rnn_output_dim=16, ode_method='rk4', rnn_dropout=0.0, rnn_layers=2):
    return GOKU(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, ode_method, rnn_dropout, rnn_layers, GOKU_pendulum_friction.Encoder, GOKU_pendulum_friction.Decoder)


def create_goku_cvs(input_dim=3, latent_dim=64, rnn_input_dim=64, rnn_output_dim=64, ode_method='rk4', rnn_dropout=0.0, rnn_layers=2):
    return GOKU(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, ode_method, rnn_dropout, rnn_layers, GOKU_cvs.Encoder, GOKU_cvs.Decoder)


def create_goku_double_pendulum(input_dim=[32, 32], latent_dim=16, rnn_input_dim=32, rnn_output_dim=16, ode_method='rk4', rnn_dropout=0.0, rnn_layers=2):
    return GOKU(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, ode_method, rnn_dropout, rnn_layers, GOKU_double_pendulum.Encoder, GOKU_double_pendulum.Decoder)
