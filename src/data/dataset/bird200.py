import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class Cub2011(Dataset):
    base_folder = "CUB_200_2011/images"

    def __init__(self, root, train=True, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"), sep=" ", names=["img_id", "filepath"]
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"), sep=" ", names=["img_id", "target"]
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        class_names = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "classes.txt"), sep=" ", names=["class_id", "target"]
        )
        self.class_names_str = [name.split(".")[1].replace("_", " ") for name in class_names.target]

        self.classes = [name for name in class_names.target]  # garmin
        self._labels = self.data["target"].values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
