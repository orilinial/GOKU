import numpy as np
from torch.utils.data import Dataset
import torch
import random


class ODEDataSet(Dataset):
    def __init__(self, file_path, ds_type, seq_len, random_start, transforms=None):
        self.transforms = transforms
        self.random_start = random_start
        # self.train = train
        self.ds_type = ds_type
        self.seq_len = seq_len

        data_dict = torch.load(file_path)

        if ds_type == 'train':
            buffer = int(round(data_dict["train"].shape[0] * (1 - 0.1)))
            self.data = torch.FloatTensor(data_dict["train"])[:buffer]
            self.latent = torch.FloatTensor(data_dict["train_latent"])[:buffer]
            self.mask = torch.FloatTensor(data_dict["train_latent_mask"])[:buffer]

        elif ds_type == 'val':
            buffer = int(round(data_dict["train"].shape[0] * (1 - 0.1)))
            self.data = torch.FloatTensor(data_dict["train"])[buffer:]
            self.latent = torch.FloatTensor(data_dict["train_latent"])[buffer:]
            self.mask = torch.FloatTensor(data_dict["train_latent_mask"])[buffer:]

        elif ds_type == 'test':
            self.data = torch.FloatTensor(data_dict["test"])
            self.latent = torch.FloatTensor(data_dict["test_latent"])
            self.mask = torch.FloatTensor(data_dict["test_latent_mask"])

        # self.possible_start_dict = {}
        # for sample in range(self.mask.size(0)):
        #     possible_start = torch.zeros(self.mask.size(1))
        #     for dim in range(self.mask.size(2)):
        #         non_zero_elements = torch.nonzero(self.mask[sample, :, dim]).squeeze()
        #         for elem in non_zero_elements:
        #             possible_start[max(0, elem-seq_len+1):elem+1] = 1
        #             possible_start = possible_start[:(self.mask.size(1)-seq_len)]
        #     self.possible_start_dict[sample] = torch.nonzero(possible_start).squeeze()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        if self.random_start:
            start_time = random.randint(0, self.data.size(1) - self.seq_len)
        else:
            start_time = 0

        sample = self.data[idx, start_time:start_time + self.seq_len]
        latent = self.latent[idx, start_time:start_time + self.seq_len]
        mask = self.mask[idx, start_time:start_time + self.seq_len]

        for transform in self.transforms:
            sample = self.transforms[transform](sample)

        return sample, latent, mask


class NormalizeZScore(object):
    """Normalize sample by mean and std."""
    def __init__(self, data_norm_params):
        self.mean = torch.FloatTensor(data_norm_params["mean"])
        self.std = torch.FloatTensor(data_norm_params["std"])

    def __call__(self, sample):
        new_sample = torch.zeros_like(sample, dtype=torch.float)
        for feature in range(self.mean.size(0)):
            if self.std[feature] > 0:
                new_sample[:, feature] = (sample[:, feature] - self.mean[feature]) / self.std[feature]
            else:
                new_sample[:, feature] = (sample[:, feature] - self.mean[feature])

        return new_sample

    def denormalize(self, batch):
        denormed_batch = torch.zeros_like(batch)
        for feature in range(batch.size(2)):
            denormed_batch[:, :, feature] = (batch[:, :, feature] * self.std[feature]) + self.mean[feature]

        return denormed_batch


class NormalizeToUnitSegment(object):
    """Normalize sample to the segment [0, 1] by max and min"""
    def __init__(self, data_norm_params):
        self.min_val = data_norm_params["min"]
        self.max_val = data_norm_params["max"]

    def __call__(self, sample):
        new_sample = (sample - self.min_val) / (self.max_val - self.min_val)
        return new_sample

    def denormalize(self, batch):
        denormed_batch = (batch * (self.max_val - self.min_val)) + self.min_val
        return denormed_batch
