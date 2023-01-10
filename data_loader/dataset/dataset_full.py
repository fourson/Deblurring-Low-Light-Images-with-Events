import os
import fnmatch

import numpy as np
import cv2
from torch.utils.data import Dataset

from utils.util import normalize_events


class TrainDataset(Dataset):
    """
        for learning S (sharp image (APS or RGB))

        as input:
        E: events (13 channel voxel grid), as float32
        B: blur APS, [0, 1], as float32
        Bi: blur image (APS or RGB), [0, 1], as float32

        as target:
        Bi_clean: clean blur image (APS or RGB), [0, 1], as float32
        F: bi-directional optical flow, as float32
        S: sharp image (APS or RGB), [0, 1] float, as float32
    """

    def __init__(self, data_dir, transform=None, RGB=False):
        self.RGB = RGB

        self.E_dir = os.path.join(data_dir, 'subnetwork1', 'events_voxel_grid')
        self.B_dir = os.path.join(data_dir, 'share', 'APS_blur')

        if self.RGB:
            self.Bi_dir = os.path.join(data_dir, 'subnetwork2', 'RGB_blur')
            self.Bi_clean_dir = os.path.join(data_dir, 'subnetwork2', 'RGB_blur_clean')
            self.S_dir = os.path.join(data_dir, 'subnetwork2', 'RGB')
        else:
            self.Bi_clean_dir = os.path.join(data_dir, 'share', 'APS_blur_clean')
            self.S_dir = os.path.join(data_dir, 'subnetwork2', 'APS')

        self.F_dir = os.path.join(data_dir, 'share', 'flow')

        self.names = [file_name[:-4] for file_name in fnmatch.filter(os.listdir(self.E_dir), '*.npy')]

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        # as input:
        # (H, W, 13)
        E = normalize_events(np.load(os.path.join(self.E_dir, name + '.npy')))
        # (H, W, 1)
        B = cv2.imread(os.path.join(self.B_dir, name + '.png'), -1)[..., None]

        # as target:
        # (H, W, 4)
        F = np.load(os.path.join(self.F_dir, name + '.npy'))

        if self.RGB:
            # as input:
            # (H, W, 3)
            Bi = cv2.cvtColor(cv2.imread(os.path.join(self.Bi_dir, name + '.png'), -1), cv2.COLOR_BGR2RGB)
            # as target:
            # (H, W, 3)
            Bi_clean = cv2.cvtColor(cv2.imread(os.path.join(self.Bi_clean_dir, name + '.png'), -1), cv2.COLOR_BGR2RGB)
            # (H, W, 3)
            S = cv2.cvtColor(cv2.imread(os.path.join(self.S_dir, name + '.png'), -1), cv2.COLOR_BGR2RGB)
        else:
            # as input:
            # (H, W, 1)
            Bi = B
            # as target:
            # (H, W, 1)
            Bi_clean = cv2.imread(os.path.join(self.Bi_clean_dir, name + '.png'), -1)[..., None]
            # (H, W, 1)
            S = cv2.imread(os.path.join(self.S_dir, name + '.png'), -1)[..., None]

        if self.transform:
            E = self.transform(E)
            B = self.transform(B)
            Bi = self.transform(Bi)

            Bi_clean = self.transform(Bi_clean)
            F = self.transform(F)
            S = self.transform(S)

        return {'E': E, 'B': B, 'Bi': Bi, 'Bi_clean': Bi_clean, 'F': F, 'S': S, 'name': name}


class InferDataset(Dataset):
    """
        for learning S (sharp image (APS or RGB))

        as input:
        E: events (13 channel voxel grid), as float32
        B: blur APS, [0, 1], as float32
        Bi: blur image (APS or RGB), [0, 1], as float32
    """

    def __init__(self, data_dir, transform=None, RGB=False, real=False):
        self.RGB = RGB

        if not real:
            # for synthetic data
            self.E_dir = os.path.join(data_dir, 'subnetwork1', 'events_voxel_grid')
            self.B_dir = os.path.join(data_dir, 'share', 'APS_blur')
            if self.RGB:
                self.Bi_dir = os.path.join(data_dir, 'subnetwork2', 'RGB_blur')
        else:
            # for real data
            self.E_dir = os.path.join(data_dir, 'events_voxel_grid')
            self.B_dir = os.path.join(data_dir, 'APS_blur')
            if self.RGB:
                self.Bi_dir = os.path.join(data_dir, 'RGB_blur')

        self.names = [file_name[:-4] for file_name in fnmatch.filter(os.listdir(self.E_dir), '*.npy')]

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        # as input:
        # (H, W, 13)
        E = normalize_events(np.load(os.path.join(self.E_dir, name + '.npy')))
        # (H, W, 1)
        B = cv2.imread(os.path.join(self.B_dir, name + '.png'), -1)[..., None]

        if self.RGB:
            # as input:
            # (H, W, 3)
            Bi = cv2.cvtColor(cv2.imread(os.path.join(self.Bi_dir, name + '.png'), -1), cv2.COLOR_BGR2RGB)
        else:
            # as input:
            # (H, W, 1)
            Bi = B

        if self.transform:
            E = self.transform(E)
            B = self.transform(B)
            Bi = self.transform(Bi)

        return {'E': E, 'B': B, 'Bi': Bi, 'name': name}
