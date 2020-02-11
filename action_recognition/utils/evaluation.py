import torch

def compute_correct(x_folder, y_folder):
    """
    Compute number of correct predicts
    """
    _, x = torch.max(x_folder, 1)
    # _, y = torch.max(y_folder, 1)
    # print(y_folder.size())
    # print('x:', x.size())
    return (x == y_folder).sum()