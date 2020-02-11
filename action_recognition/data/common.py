from torch.utils.data import Dataset, TensorDataset
import numpy as np
import torch

class ActionRecog(Dataset):
    """
    load data from MERL dataset
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

class MERL_3D(Dataset):
    """
    Load data from MERL dataset and convert 2D coordiante to 3D coordinate
    return tensor in format [x, y] with x's shape: (None, 32, 54), y's shape (None, 32, 6)
    """
    def __init__(self, x_folder, y_folder):
        self.n_seq = 32
        self.x_folder = x_folder
        self.y_folder = y_folder
        self.data = self.get_data()
    def get_x(self):
        with open(self.x_folder) as f:
            file = f.read().splitlines()
            x = np.array([i for i in [row.split(',') for row in file]], dtype= np.float)
        return x
    def convert_to_3d(self, keypoints):
        return 'hihi'
        
class UCI(Dataset):
    """
    load data from HAR dataset
    return tensor in format [x, y] with x's shape: (None, 128, 9), y's shape: (None, 128, 6)
    """
    def __init__(self, x_folder, phase):
        # x_folder : HAR HAR Dataset/
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

class KinectV2Joint:
    def __init__(self):
        pass

    def read_json(self, link):
        with open(link) as f:
            file = f.read()
        print(len(file))
        print(file[0])
    def draw_keypoint(self):
        pass
if __name__=='__main__':
    import json
    import cv2

    base_link = '/home/nbviet/PycharmProjects/LSTM_keypoint/database/Kinect v2 joints'
    json_link = base_link + '/kp1.json'
    vid_link = base_link +'/cl1.webm'
    out = cv2.VideoWriter('output5.avi', cv2.VideoWriter_fourcc(*"MJPG"), 10, (1920, 1080))

    vid = cv2.VideoCapture(vid_link)

    with open('keypoint_cl_2.json') as f:
        file = json.load(f)
    print('json:', len(file))
    frame = 0
    for index, f in enumerate(file):
    # while(vid.isOpened()):

        _, img = vid.read()
        if img is None:
            break
        frame += 1
        img = cv2.resize(img, (1920, 1080))

        for i in range(25):
            coordinate = (int(f[2 * i]), int(f[2 * i + 1]))
            img = cv2.circle(img, coordinate, 5, (0, 0, 255), 3)
        cv2.imshow('img', img)
        out.write(img)
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break


    # with open(json_link) as f:
    #     file = json.load(f)
    # coordinate = []
    # for f in file:
    #     for keypoints in f['bodies']:
    #         if keypoints['tracked']:
    #             tmp = []
    #             for keypoint in keypoints['joints']:
    #                 if(keypoint['colorX'] is None):
    #                     keypoint['colorX'] = 0
    #                     keypoint['colorY'] = 0
    #
    #                 tmp.append(round(keypoint['colorX'] * 1920))
    #                 tmp.append(round(keypoint['colorY'] * 1080))
    #             coordinate.append(tmp)
    #             break      # 1 person was tracked in video
    # # coordinate = coordinate[0::3]   # data repeated 3 times
    # with open('keypoint_cl_2.json', 'w') as f:
    #     json.dump(coordinate, f)