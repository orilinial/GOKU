import torch.nn as nn


class LSTM_LV(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=4):
        super(LSTM_LV, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_to_output_net = nn.Sequential(nn.Linear(hidden_dim, 200),
                                                  nn.ReLU(),
                                                  nn.Linear(200, input_dim))

    def forward(self, x, h=None):
        if h is None:
            out, h = self.lstm(x)
        else:
            out, h = self.lstm(x, h)

        out = self.hidden_to_output_net(out[:, -1])
        return out, h


class LSTMPixelPendulum(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMPixelPendulum, self).__init__()
        self.input_dim = input_dim
        lstm_input_dim = 32

        # self.input_to_lstm = nn.Sequential(nn.Linear(input_dim[0] * input_dim[1], 200),
        #                                    nn.ReLU(),
        #                                    nn.Linear(200, lstm_input_dim))

        self.first_layer = nn.Linear(input_dim[0] * input_dim[1], 200)
        self.second_layer = nn.Linear(200, 200)
        self.third_layer = nn.Linear(200, 200)
        self.fourth_layer = nn.Linear(200, lstm_input_dim)

        # self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers, batch_first=True)
        self.rnn = nn.RNN(input_size=lstm_input_dim, hidden_size=hidden_dim,
                          nonlinearity='relu', batch_first=True,
                          bidirectional=False, num_layers=num_layers, dropout=0)

        self.first_layerT = nn.Linear(hidden_dim, 200)
        self.second_layerT = nn.Linear(200, 200)
        self.third_layerT = nn.Linear(200, 200)
        self.fourth_layerT = nn.Linear(200, input_dim[0] * input_dim[1])

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # self.lstm_to_output = nn.Sequential(nn.Linear(hidden_dim, 200),
        #                                     nn.ReLU(),
        #                                     nn.Linear(200, input_dim[0] * input_dim[1]),
        #                                     nn.Sigmoid())

    def forward(self, mini_batch, h=None):
        mini_batch = mini_batch.view(mini_batch.size(0), mini_batch.size(1), mini_batch.size(2) * mini_batch.size(3))
        mini_batch = self.relu(self.first_layer(mini_batch))
        mini_batch = mini_batch + self.relu(self.second_layer(mini_batch))
        mini_batch = mini_batch + self.relu(self.third_layer(mini_batch))
        mini_batch = self.relu(self.fourth_layer(mini_batch))

        if h is None:
            predicted_z, h = self.rnn(mini_batch)
        else:
            predicted_z, h = self.rnn(mini_batch, h)

        recon_batch = self.relu(self.first_layerT(predicted_z[:, -1]))
        recon_batch = recon_batch + self.relu(self.second_layerT(recon_batch))
        recon_batch = recon_batch + self.relu(self.third_layerT(recon_batch))
        recon_batch = self.sigmoid(self.fourth_layerT(recon_batch))
        recon_batch = recon_batch.view(predicted_z.size(0), self.input_dim[0], self.input_dim[1])

        out = recon_batch.view(predicted_z.size(0), self.input_dim[0], self.input_dim[1])

        return out, h


class LSTM_CVS(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=4):
        super(LSTM_CVS, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_to_output_net = nn.Sequential(nn.Linear(hidden_dim, 200),
                                                  nn.ReLU(),
                                                  nn.Linear(200, input_dim))

    def forward(self, x, h=None):
        if h is None:
            out, h = self.lstm(x)
        else:
            out, h = self.lstm(x, h)

        out = self.hidden_to_output_net(out[:, -1])
        return out, h


def create_lstm_lv(input_dim=4, hidden_dim=128, num_layers=4):
    return LSTM_LV(input_dim, hidden_dim, num_layers)


def create_lstm_pixel_pendulum(input_dim=[28, 28], hidden_dim=16, num_layers=2):
    return LSTMPixelPendulum(input_dim, hidden_dim, num_layers)


def create_lstm_pixel_pendulum_friction(input_dim=[28, 28], hidden_dim=16, num_layers=2):
    return LSTMPixelPendulum(input_dim, hidden_dim, num_layers)


def create_lstm_cvs(input_dim=3, hidden_dim=128, num_layers=4):
    return LSTM_CVS(input_dim, hidden_dim, num_layers)
