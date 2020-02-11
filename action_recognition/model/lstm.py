import torch.nn as nn
import torch

class LSTM(nn.Module):
    """
    Model LSTM with 2 layers, drop out = 0.25
    """
    def __init__(self, input_size, hidden_size, output_size, n_seq):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_seq = n_seq
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            batch_first=True, bidirectional=False, dropout=0.25)
        self.fc = nn.Linear(n_seq * hidden_size, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()

        out, _ = self.lstm(x)
        out = torch.reshape(out,(-1, self.n_seq * self.hidden_size))
        out = self.fc(out)

        return out

class Bi_LSTM(nn.Module):
    """
    Model Bi-LSTM with 2 layers, drop out = 0.25
    """
    def __init__(self, input_size, hidden_size, output_size, n_seq):
        super(Bi_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_seq = n_seq
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=3,
                            batch_first=True, bidirectional=True, dropout=0.25)
        self.fc = nn.Linear(n_seq * 2 * hidden_size, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = torch.reshape(out, (-1, self.n_seq * 2 * self.hidden_size))
        out = self.fc(out)

        return out

class RNN(nn.Module):
    """
    Model RNN with 2 layers, drop out = 0.25
    """
    def __init__(self, input_size, hidden_size, output_size, n_seq):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_seq = n_seq
        self.rnn = nn.RNN(input_size=input_size, hidden_size= self.hidden_size, num_layers= 3, batch_first= True, dropout= 0.25)
        self.fc = nn.Linear(n_seq * hidden_size, output_size)
    def forward(self, x):
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        out = torch.reshape(out,(-1, self.n_seq * self.hidden_size))
        out = self.fc(out)
        out = out
        return out
