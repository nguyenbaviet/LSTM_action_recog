import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, drop_out, n_seq):
        super(CRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.drop_out = drop_out
        self.n_seq = n_seq

        self.conv1 = nn.Conv2d(1, 1, (3, 3), padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(1)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers,
                            batch_first=True, bidirectional=True, dropout=drop_out)
        self.fc1 = nn.Linear(n_seq * 2 * hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 1, self.n_seq, self.input_size)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = out.view(self.n_seq, -1, self.input_size)
        out, _ = self.lstm(out)
        out = out.permute(1, 0, 2)
        out = out.view(-1, self.n_seq * 2 * self.hidden_size)
        out = self.fc1(out)
        out = self.softmax(out)

        return out
