import torch.nn as nn

class LSTM(nn.Module):
    """
    Model LSTM with 2 layers, drop out = 0.25
    """
    def __init__(self, input_size, hidden_size, output_size, n_seq):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = 2
        self.output_size = output_size
        self.drop_out = 0.25
        self.n_seq = n_seq
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=self.n_layers,
                            batch_first=True, bidirectional=False, dropout=self.drop_out)
        self.fc = nn.Linear(n_seq * hidden_size, output_size)
        self.sigmoid = nn.Softmax()

    def forward(self, x):
        x = x.permute(1, 0, 2)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = out.permute(1, 0, 2)
        out = out.view(-1, self.n_seq * self.hidden_size)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out

class Bi_LSTM(nn.Module):
    """
    Model Bi-LSTM with 2 layers, drop out = 0.25
    """
    def __init__(self, input_size, hidden_size, output_size, n_seq):
        super(Bi_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = 2
        self.output_size = output_size
        self.drop_out = 0.25
        self.n_seq = n_seq
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=self.n_layers,
                            batch_first=True, bidirectional=True, dropout=self.drop_out)
        self.fc = nn.Linear(n_seq * 2 * hidden_size, output_size)
        self.sigmoid = nn.Softmax()

    def forward(self, x):
        x = x.permute(1, 0, 2)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = out.permute(1, 0, 2)
        out = out.view(-1, self.n_seq * 2 * self.hidden_size)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out