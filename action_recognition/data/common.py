from torch.utils.data import Dataset, TensorDataset
import numpy as np
import torch

class ActionRecog(Dataset):
    """
    load data from MERL data set
    return tensor in format [x, y] with x's shape: (None, 32, 36), y's shape (None, 32, 6)
    """
    def __init__(self, x_folder, y_folder):
        self.n_seq = 32
        self.x_folder = x_folder
        self.y_folder = y_folder

        self.data = self.get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def get_data(self):
        with open(self.x_folder) as f:
            file = f.read().splitlines()
            x = np.array([i for i in [row.split(',') for row in file]], dtype=np.float)
        blocks = int(len(x) / self.n_seq)
        x = np.array(np.split(x, blocks))
        x = np.array(x, dtype=float)
        with open(self.y_folder) as f:
            file = f.read().splitlines()
        y = np.array([i for i in [row.replace('  ', ' ').strip().split() for row in file]], dtype=int)
        new_y = []
        for i in y:
            ny = np.zeros(6)
            ny[i - 1] = 1
            new_y.append(ny)
        y = np.array(new_y, dtype=float)
        return TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())

class UCI(Dataset):
    """
    load data from UCI dataset
    return tensor in format [x, y] with x's shape: (None, 128, 9), y's shape: (None, 128, 6)
    """
    def __init__(self, x_folder, phase):
        # x_folder : UCI HAR Dataset/
        self.x_folder = x_folder
        self.phase = phase
        self.x = self.load_x()
        self.y = self.one_hot(self.load_y())

        self.data = TensorDataset(torch.from_numpy(self.x).float(), torch.from_numpy(self.y).float())

    def load_x(self):
        x_signals = []
        INPUT_SIGNAL_TYPES = ['body_acc_x_','body_acc_y_','body_acc_z_','body_gyro_x_','body_gyro_y_',
                              'body_gyro_z_','total_acc_x_','total_acc_y_','total_acc_y_']
        x_folder = [ self.x_folder + self.phase + '/Inertial Signals/' + signal + self.phase + '.txt' for signal in INPUT_SIGNAL_TYPES ]
        for x in x_folder:
            with open(x) as f:
                x_signals.append(
                    [np.array(serie, dtype=np.float32) for serie in [
                        row.replace('  ', ' ').strip().split(' ') for row in f
                    ]]
                )
        return np.transpose(np.array(x_signals), (1, 2, 0))
    def load_y(self):
        with open(self.x_folder + self.phase + '/y_'+ self.phase +'.txt') as f:
            y = np.array(
                [elem for elem in [
                    row.replace('  ', ' ').strip().split(' ') for row in f
                ]],
                dtype=np.int32
            )
        return y -1
    def one_hot(self, y):
        n_classes = 6
        return np.eye(n_classes)[np.array(y, dtype=np.int32)].reshape(-1, 6)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item]
