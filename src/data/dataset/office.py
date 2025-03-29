import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.data_utils import Datum


class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        self.base_path = base_path
        if train:
            path = os.path.join(self.base_path, "office_caltech_10/{}_train.pkl".format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
        else:
            path = os.path.join(self.base_path, "office_caltech_10/{}_test.pkl".format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)

        self.site_domian = {"amazon": 0, "caltech": 1, "dslr": 2, "webcam": 3}
        self.domain = self.site_domian[site]
        self.lab2cname = {
            "back_pack": 0,
            "bike": 1,
            "calculator": 2,
            "headphones": 3,
            "keyboard": 4,
            "laptop_computer": 5,
            "monitor": 6,
            "mouse": 7,
            "mug": 8,
            "projector": 9,
        }
        self.classnames = {
            "back_pack",
            "bike",
            "calculator",
            "headphones",
            "keyboard",
            "laptop_computer",
            "monitor",
            "mouse",
            "mug",
            "projector",
        }
        self.target = [self.lab2cname[text] for text in self.label]

        self.label = self.label.tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.target)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.target)):
            img_path = os.path.join(self.base_path, self.paths[i])
            data_idx = img_path
            target_idx = self.target[i]
            label_idx = self.label[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=int(self.domain), classname=label_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.target[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
