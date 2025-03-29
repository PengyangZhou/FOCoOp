import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, ImageFolder, CIFAR100, Places365, SVHN
from torchvision.transforms import transforms

from src.data.dataset.PACS_dataset import PACSDataset
from src.data.dataset.bird200 import Cub2011
from src.data.dataset.caltech101 import Caltech101
from src.data.dataset.car196 import StanfordCars
from src.data.dataset.domainnet import DomainNetDataset
from src.data.dataset.dtd import DTD
from src.data.dataset.flowers102 import Flowers102
from src.data.dataset.food101 import Food101
from src.data.dataset.load_cifar10_corrupted import CIFAR10_CORRUPT, CIFAR100_CORRUPT
from src.data.dataset.office import OfficeDataset
from src.data.dataset.pet37 import OxfordIIITPet
from src.data.dataset.tinyimagenet import TinyImageNet
from src.data.dataset.tinyimagenet_corrupt import TinyImageNet_CORRUPT


def dirichlet_load_train_clip(dataset_path, id_dataset, num_client, num_shot, alpha, seed, transform=None):
    root = dataset_path

    if transform is None:
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )  # for CLIP
        transform = transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
        )

    if id_dataset in ["ImageNet", "ImageNet100", "ImageNet10", "ImageNet20"]:
        train_data = ImageFolder(os.path.join(root, id_dataset, "train"), transform=transform)

    elif id_dataset == "food101":
        train_data = Food101(root, split="train", download=True, transform=transform)
        targets = train_data._labels
        class_names = train_data.classes
        num_class = 101
    elif id_dataset == "pet37":
        train_data = OxfordIIITPet(root, split="trainval", download=True, transform=transform)
        targets = train_data._labels
        class_names = train_data.classes
        num_class = 37

    elif id_dataset == "bird200":
        train_data = Cub2011(root, train=True, transform=transform)  # 5994
        targets = train_data._labels
        class_names = train_data.classes
        num_class = 200
    elif id_dataset == "car196":
        train_data = StanfordCars(root, split="train", download=True, transform=transform)
        # targets = train_data._labels
        class_names = train_data.classes
        num_class = len(class_names)

    elif id_dataset == "cifar10":
        train_data = CIFAR10(root, train=True, download=True, transform=transform)
        targets = train_data.targets
        class_names = train_data.classes
        num_class = 10
    elif id_dataset == "cifar100":
        train_data = CIFAR100(root, train=True, download=False, transform=transform)
        targets = train_data.targets
        class_names = train_data.classes
        num_class = 100
    elif id_dataset == "caltech101":
        train_data = Caltech101(root, split="train", download=False, transform=transform)
        targets = train_data.targets
        class_names = train_data.classes
        num_class = 100
    elif id_dataset == "tinyimagenet":
        train_data = TinyImageNet(root=root, train=True, transform=transform)
        num_class = 200
        targets = train_data.targets
        class_names = train_data.classes
    else:
        raise ValueError(f"Unsupported dataset: {id_dataset}")

    distribution = create_dirichlet_distribution(alpha, num_client, num_class, seed)
    train_split = split_by_distribution(np.array(targets), distribution)

    if num_shot is not None:
        for idx, indices in train_split.items():
            class_indices = {}
            for i in list(indices):
                label = targets[i]
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(i)

            sampled_indices = []
            for label, inds in class_indices.items():
                if len(inds) > num_shot:
                    sampled_indices.extend(np.random.choice(inds, num_shot, replace=False))
                else:
                    sampled_indices.extend(np.random.choice(inds, num_shot, replace=True))

            train_split[idx] = np.array(sampled_indices)

    train_datasets = [Subset(train_data, train_split[idx]) for idx in range(num_client)]

    logging.info(f"-------- dirichlet distribution with alpha = {alpha}, {num_client} clients --------")
    logging.info(f"in-distribution train datasets: {[len(dataset) for dataset in train_datasets]}")

    return train_datasets, num_class, class_names


def pathological_load_train_clip(
    dataset_path, id_dataset, num_client, num_shot, class_per_client, seed, transform=None, non_overlap=True
):
    root = dataset_path
    if transform is None:
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )  # for CLIP
        transform = transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
        )

    if id_dataset in ["ImageNet", "ImageNet100", "ImageNet10", "ImageNet20"]:
        train_data = ImageFolder(os.path.join(root, id_dataset, "train"), transform=transform)

    elif id_dataset == "food101":
        train_data = Food101(root, split="train", download=False, transform=transform)
        targets = train_data._labels
        class_names = train_data.classes
        num_class = 101
    elif id_dataset == "pet37":
        train_data = OxfordIIITPet(root, split="trainval", download=False, transform=transform)
        targets = train_data._labels
        class_names = train_data.classes
        num_class = 37
    elif id_dataset == "flowers102":
        train_data = Flowers102(root, split="train", download=False, transform=transform)
        targets = train_data._labels
        class_names = train_data.classes
        num_class = 102
    elif id_dataset == "caltech101":
        train_data = Caltech101(root, split="train", download=False, transform=transform)
        targets = train_data._labels
        class_names = train_data.classes
        num_class = 101
    elif id_dataset == "dtd":
        train_data = DTD(root, split="train", download=False, transform=transform)
        targets = train_data._labels
        class_names = train_data.classes
        num_class = 47

    elif id_dataset == "cifar10":
        train_data = CIFAR10(root, train=False, download=True, transform=transform)
        targets = train_data.targets
        class_names = train_data.classes
        num_class = 10
    elif id_dataset == "cifar100":
        train_data = CIFAR100(root, train=False, download=True, transform=transform)
        targets = train_data.targets
        class_names = train_data.classes
        num_class = 100
    elif id_dataset == "tinyimagenet":
        train_data = TinyImageNet(root=root, train=True, transform=transform)
        num_class = 200
        targets = train_data.targets
        class_names = train_data.classes
    else:
        raise ValueError(f"Unsupported dataset: {id_dataset}")

    if non_overlap:
        distribution = create_non_overlap_distribution(num_client, num_class, seed)
    else:
        distribution = create_pathological_distribution(class_per_client, num_client, num_class, seed)
    train_split = split_by_distribution(np.array(targets), distribution)

    if num_shot is not None:
        for idx, indices in train_split.items():

            class_indices = {}
            for i in list(indices):

                label = targets[i]
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(i)

            sampled_indices = []
            for label, inds in class_indices.items():
                if len(inds) > num_shot:
                    sampled_indices.extend(np.random.choice(inds, num_shot, replace=False))
                else:
                    sampled_indices.extend(np.random.choice(inds, num_shot, replace=True))

            train_split[idx] = np.array(sampled_indices)

    train_datasets = [Subset(train_data, train_split[idx]) for idx in range(num_client)]
    logging.info(
        f"-------- pathological distribution with {class_per_client} classes per client, {num_client} clients --------"
    )
    logging.info(f"in-distribution train datasets: {[len(dataset) for dataset in train_datasets]}")

    return train_datasets, num_class, class_names


def dirichlet_load_test_clip(dataset_path, id_dataset, num_client, alpha, seed, corrupt_list=None):
    root = dataset_path
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    )  # for CLIP
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )

    if id_dataset in ["ImageNet", "ImageNet100", "ImageNet10", "ImageNet20"]:
        test_data = ImageFolder(os.path.join(root, id_dataset, "train"), transform=transform)

    elif id_dataset == "food101":
        test_data = Food101(root, split="test", download=True, transform=transform)
        targets = test_data._labels
        class_names = test_data.classes
        num_class = 101

    elif id_dataset == "pet37":
        test_data = OxfordIIITPet(root, split="test", download=True, transform=transform)
        targets = test_data._labels
        class_names = test_data.classes
        num_class = 37

    elif id_dataset == "cifar10":
        test_data = CIFAR10(root, train=False, download=True, transform=transform)
        if corrupt_list is not None:
            cor_test = []
            for idx, cor_type in enumerate(corrupt_list):
                cor_test.append(CIFAR10_CORRUPT(root=dataset_path, cortype=cor_type, transform=transform))

        targets = test_data.targets
        class_names = test_data.classes
        num_class = 10
    elif id_dataset == "cifar100":
        test_data = CIFAR100(root, train=False, download=True, transform=transform)
        if corrupt_list is not None:
            cor_test = []
            for idx, cor_type in enumerate(corrupt_list):
                cor_test.append(CIFAR100_CORRUPT(root=dataset_path, cortype=cor_type, transform=transform))
        targets = test_data.targets
        class_names = test_data.classes
        num_class = 100
    elif id_dataset == "caltech101":
        test_data = Caltech101(root, split="test", download=False, transform=transform)
        targets = test_data.targets
        class_names = test_data.classes
        num_class = 100
    elif id_dataset == "tinyimagenet":
        test_data = TinyImageNet(root=dataset_path, train=False, transform=transform)
        if corrupt_list is not None:
            cor_test = []
            for idx, cor_type in enumerate(corrupt_list):
                cor_test.append(TinyImageNet_CORRUPT(root=dataset_path, cortype=cor_type, transform=transform))
        targets = test_data.targets
        class_names = test_data.classes
        num_class = 200
    else:
        raise ValueError(f"Unsupported dataset: {id_dataset}")

    distribution = create_dirichlet_distribution(alpha, num_client, num_class, seed)
    test_id_split = split_by_distribution(np.array(targets), distribution)
    test_id_datasets = [Subset(test_data, test_id_split[idx]) for idx in range(num_client)]

    logging.info(f"-------- dirichlet distribution with alpha = {alpha}, {num_client} clients --------")
    logging.info(f"in-distribution test datasets: {[len(dataset) for dataset in test_id_datasets]}")

    if id_dataset not in ["cifar10", "cifar100", "tinyimagenet"]:
        return test_id_datasets, None, num_class, class_names
    else:
        test_cor_split = split_by_distribution(np.array(cor_test[0].targets), distribution)
        cor_test_datasets = [
            {cor_type: Subset(cor_test[idx], test_cor_split[client_idx]) for idx, cor_type in enumerate(corrupt_list)}
            for client_idx in range(num_client)
        ]
        return test_id_datasets, cor_test_datasets, num_class, class_names


def pathological_load_test_clip(
    dataset_path, id_dataset, num_client, class_per_client, seed, corrupt_list=None, non_overlap=True
):
    root = dataset_path
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    )  # for CLIP
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )

    if id_dataset in ["ImageNet", "ImageNet100", "ImageNet10", "ImageNet20"]:
        test_data = ImageFolder(os.path.join(root, id_dataset, "train"), transform=transform)

    elif id_dataset == "food101":
        test_data = Food101(root, split="test", download=False, transform=transform)
        targets = test_data._labels
        class_names = test_data.classes
        num_class = 101
    elif id_dataset == "pet37":
        test_data = OxfordIIITPet(root, split="test", download=False, transform=transform)
        targets = test_data._labels
        class_names = test_data.classes
        num_class = 37
    elif id_dataset == "flowers102":
        test_data = Flowers102(root, split="test", download=False, transform=transform)
        targets = test_data._labels
        class_names = test_data.classes
        num_class = 102
    elif id_dataset == "caltech101":
        test_data = Caltech101(root, split="test", download=False, transform=transform)
        targets = test_data._labels
        class_names = test_data.classes
        num_class = 101
    elif id_dataset == "dtd":
        test_data = DTD(root, split="test", download=False, transform=transform)
        targets = test_data._labels
        class_names = test_data.classes
        num_class = 47

    elif id_dataset == "cifar10":
        test_data = CIFAR10(root, train=False, download=True, transform=transform)
        if corrupt_list is not None:
            cor_test = []
            for idx, cor_type in enumerate(corrupt_list):
                cor_test.append(CIFAR10_CORRUPT(root=dataset_path, cortype=cor_type, transform=transform))
        targets = test_data.targets
        class_names = test_data.classes
        num_class = 10
    elif id_dataset == "cifar100":
        test_data = CIFAR100(root, train=False, download=True, transform=transform)
        if corrupt_list is not None:
            cor_test = []
            for idx, cor_type in enumerate(corrupt_list):
                cor_test.append(CIFAR100_CORRUPT(root=dataset_path, cortype=cor_type, transform=transform))
        targets = test_data.targets
        class_names = test_data.classes
        num_class = 100
    elif id_dataset == "tinyimagenet":
        test_data = TinyImageNet(root=dataset_path, train=False, transform=transform)
        if corrupt_list is not None:
            cor_test = []
            for idx, cor_type in enumerate(corrupt_list):
                cor_test.append(TinyImageNet_CORRUPT(root=dataset_path, cortype=cor_type, transform=transform))
        targets = test_data.targets
        class_names = test_data.classes
        num_class = 200
    else:
        raise ValueError(f"Unsupported dataset: {id_dataset}")

    if non_overlap:
        distribution = create_non_overlap_distribution(num_client, num_class, seed)
    else:
        distribution = create_pathological_distribution(class_per_client, num_client, num_class, seed)
    test_id_split = split_by_distribution(np.array(targets), distribution)
    test_id_datasets = [Subset(test_data, test_id_split[idx]) for idx in range(num_client)]

    logging.info(
        f"-------- pathological distribution with class_per_client = {class_per_client}, {num_client} clients --------"
    )
    logging.info(f"in-distribution test datasets: {[len(dataset) for dataset in test_id_datasets]}")

    if id_dataset not in ["cifar10", "cifar100", "tinyimagenet"]:
        return test_id_datasets, None, num_class, class_names
    else:
        test_cor_split = split_by_distribution(np.array(cor_test[0].targets), distribution)
        cor_test_datasets = [
            {cor_type: Subset(cor_test[idx], test_cor_split[client_idx]) for idx, cor_type in enumerate(corrupt_list)}
            for client_idx in range(num_client)
        ]
        return test_id_datasets, cor_test_datasets, num_class, class_names


def load_ood_dataset_clip(dataset_path: str, ood_dataset: str):
    root = dataset_path
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
    )  # for CLIP
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )

    if ood_dataset == "iNaturalist":
        ood_data = ImageFolder(root=os.path.join(root, "iNaturalist"), transform=transform)
    elif ood_dataset == "place365":
        ood_data = ImageFolder(root=os.path.join(root, "Places"), transform=transform)
    elif ood_dataset == "Texture":
        ood_data = ImageFolder(root=os.path.join(root, "dtd/images"), transform=transform)
    elif ood_dataset == "SUN":
        ood_data = ImageFolder(root=os.path.join(root, "SUN"), transform=transform)
    elif ood_dataset == "LSUN_C":
        ood_data = ImageFolder(root=os.path.join(root, "LSUN"), transform=transform)
    elif ood_dataset == "LSUN-R":
        ood_data = ImageFolder(root=os.path.join(root, "LSUN_resize"), transform=transform)
    elif ood_dataset == "Texture":
        ood_data = ImageFolder(root=os.path.join(root, "dtd/images"), transform=transform)
    elif ood_dataset == "isun":
        ood_data = ImageFolder(root=os.path.join(root, "iSUN"), transform=transform)
    elif ood_dataset == "place365":
        ood_data = Places365(root=root, download=True, transform=transform)
    elif ood_dataset == "SVHN":
        ood_data = SVHN(root=root, split="test", transform=transform, download=True)
    else:
        raise NotImplementedError

    return ood_data


def load_test_ood_clip(dataset_path, ood_dataset, seed, partial):
    random_number_generator = np.random.default_rng(seed)
    ood_data = load_ood_dataset_clip(dataset_path, ood_dataset)

    if partial:
        idx = random_number_generator.choice(
            [i for i in range(len(ood_data))], size=int(0.01 * len(ood_data)), replace=False
        )
        ood_data = Subset(ood_data, idx)
        logging.info(f"out of distribution test dataset's length: {len(ood_data)}")
        return ood_data
    else:
        logging.info(f"out of distribution test dataset's length: {len(ood_data)}")
        return ood_data


def load_domain_train(dataset_path, dataset, leave_out=None):
    transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ]
    )

    train_datasets = []
    if dataset == "office":
        domains = ["amazon", "caltech", "dslr", "webcam"]
        DatasetClass = OfficeDataset
    elif dataset == "domainnet":
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        DatasetClass = DomainNetDataset
    elif dataset == "pacs":
        domains = ["art_painting", "cartoon", "photo", "sketch"]
        DatasetClass = PACSDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if leave_out is not None:
        assert leave_out in domains, f"Invalid domain to leave out: {leave_out}"
        domains.remove(leave_out)

    for domain in domains:
        trainset = DatasetClass(dataset_path, domain, transform=transform)
        classnames = sorted(list(trainset.classnames))
        num_class = len(classnames)
        train_datasets.append(trainset)

    return train_datasets, num_class, classnames


def load_domain_test(dataset_path, dataset, leave_out=None):
    transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ]
    )

    if dataset == "office":
        domains = ["amazon", "caltech", "dslr", "webcam"]
        DatasetClass = OfficeDataset
        num_client = 3
    elif dataset == "domainnet":
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        DatasetClass = DomainNetDataset
        num_client = 5
    elif dataset == "pacs":
        domains = ["art_painting", "cartoon", "photo", "sketch"]
        DatasetClass = PACSDataset
        num_client = 3
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if leave_out is not None:
        assert leave_out in domains, f"Invalid domain to leave out: {leave_out}"

    testset = DatasetClass(dataset_path, leave_out, transform=transform)
    classnames = sorted(list(testset.classnames))
    num_class = len(classnames)

    test_datasets = [testset for _ in range(num_client)]

    return test_datasets, num_class, classnames


def create_dirichlet_distribution(alpha: float, num_client: int, num_class: int, seed: int):
    random_number_generator = np.random.default_rng(seed)
    distribution = random_number_generator.dirichlet(np.repeat(alpha, num_client), size=num_class).transpose()
    distribution /= distribution.sum()
    return distribution


def create_pathological_distribution(class_per_client: int, num_client: int, num_class: int, seed: int):
    random_number_generator = np.random.default_rng(seed)
    repeat_count = (num_client * class_per_client + num_class - 1) // num_class
    classes_sequence = []
    classes = np.array([i for i in range(num_class)])
    for _ in range(repeat_count):
        random_number_generator.shuffle(classes)
        classes_sequence.extend(classes.tolist())
    clients_classes = [
        classes_sequence[i : (i + class_per_client)] for i in range(0, num_client * class_per_client, class_per_client)
    ]
    distribution = np.zeros((num_client, num_class))
    for cid in range(num_client):
        for class_idx in clients_classes[cid]:
            distribution[cid, class_idx] = 1
    for class_idx in range(num_class):
        distribution[:, class_idx] /= (
            distribution[:, class_idx].sum() + 1e-9
        )  # careful 除以0 pathological采样的时候导致有些类没有覆盖到 就会存在这个除零问题
    distribution /= distribution.sum()
    return distribution


def create_non_overlap_distribution(num_client: int, num_class: int, seed: int):
    class_per_client = num_class // num_client
    remaining_classes = num_class % num_client

    random_number_generator = np.random.default_rng(seed)
    classes = np.array([i for i in range(num_class)])
    random_number_generator.shuffle(classes)

    clients_classes = [classes[i * class_per_client : (i + 1) * class_per_client] for i in range(num_client)]

    for i in range(remaining_classes):
        clients_classes[i] = np.append(clients_classes[i], classes[num_client * class_per_client + i])

    distribution = np.zeros((num_client, num_class))
    for cid in range(num_client):
        for class_idx in clients_classes[cid]:
            distribution[cid, class_idx] = 1

    return distribution


def split_by_distribution(targets, distribution):
    num_client, num_class = distribution.shape[0], distribution.shape[1]
    sample_number = np.floor(distribution * len(targets))
    class_idx = {class_label: np.where(targets == class_label)[0] for class_label in range(num_class)}

    idx_start = np.zeros((num_client + 1, num_class), dtype=np.int32)
    for i in range(0, num_client):
        idx_start[i + 1] = idx_start[i] + sample_number[i]

    client_samples = {idx: {} for idx in range(num_client)}
    for client_idx in range(num_client):
        samples_idx = np.array([], dtype=np.int32)
        for class_label in range(num_class):
            start, end = idx_start[client_idx, class_label], idx_start[client_idx + 1, class_label]
            samples_idx = np.concatenate((samples_idx, class_idx[class_label][start:end].tolist())).astype(np.int32)
        client_samples[client_idx] = samples_idx

    return client_samples


def load_ood_dataset(dataset_path: str, ood_dataset: str):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    if ood_dataset == "LSUN_C":
        ood_data = ImageFolder(
            root=os.path.join(dataset_path, "LSUN"),
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std), transforms.RandomCrop(32, padding=4)]
            ),
        )
    elif ood_dataset == "LSUN-R":
        ood_data = ImageFolder(
            root=os.path.join(dataset_path, "LSUN_resize"),
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
        )
    elif ood_dataset == "Texture":
        ood_data = ImageFolder(
            root=os.path.join(dataset_path, "dtd/images"),
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        )
    elif ood_dataset == "isun":
        ood_data = ImageFolder(
            root=os.path.join(dataset_path, "iSUN"),
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]),
        )
    elif ood_dataset == "place365":
        ood_data = Places365(
            root=dataset_path,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        )
    elif ood_dataset == "SVHN":
        ood_data = SVHN(
            root=dataset_path,
            split="test",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            download=True,
        )
    else:
        raise NotImplementedError("out of distribution dataset should be LSUN_C, dtd, isun")

    return ood_data


def load_PACS_test(dataset_path, leave_out):
    domain_datasets = dict()
    dataset_names = ["art_painting", "cartoon", "photo", "sketch"]
    trans = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    for name in dataset_names:
        domain_datasets[name] = PACSDataset(dataset_path, name, transform=trans)
    num_class = 7
    train_datasets = [domain_datasets[leave_out] for _ in range(3)]
    return train_datasets, None, num_class


def dirichlet_load_train(dataset_path, id_dataset, num_client, alpha, seed, fourier_mix_alpha=1.0):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    if id_dataset == "cifar10":
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = CIFAR10(root=dataset_path, download=True, train=True, transform=trans)
        num_class = 10

    elif id_dataset == "cifar100":
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = CIFAR100(root=dataset_path, download=True, train=True, transform=trans)
        num_class = 100

    elif id_dataset == "tinyimagenet":
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = TinyImageNet(root=dataset_path, train=True, transform=trans)
        num_class = 200

    elif id_dataset == "caltech101":
        from src.data.dataset.caltech101 import Caltech101

        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean, std)]
        )
        train_data = Caltech101(root=dataset_path, split="train", transform=trans)
        class_names = train_data.get_sorted_class_names()
        num_class = 100
        assert len(class_names) == num_class, f"Expected number of classes is {num_class}, but got {len(class_names)}"

    else:
        raise NotImplementedError("in distribution dataset should be CIFAR10 or CIFAR100.")

    distribution = create_dirichlet_distribution(alpha, num_client, num_class, seed)
    train_split = split_by_distribution(np.array(train_data.targets), distribution)
    train_datasets = [Subset(train_data, train_split[idx]) for idx in range(num_client)]

    logging.info(f"-------- dirichlet distribution with alpha = {alpha}, {num_client} clients --------")
    logging.info(f"in-distribution train datasets: {[len(dataset) for dataset in train_datasets]}")
    # return train_datasets, num_class
    return train_datasets, num_class, class_names


def pathological_load_train(dataset_path, id_dataset, num_client, class_per_client, seed):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    if id_dataset == "cifar10":
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = CIFAR10(root=dataset_path, download=True, train=True, transform=trans)
        num_class = 10
    elif id_dataset == "cifar100":
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = CIFAR100(root=dataset_path, download=True, train=True, transform=trans)
        num_class = 100
    elif id_dataset == "tinyimagenet":
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = TinyImageNet(root=dataset_path, train=True, transform=trans)
        num_class = 200
    else:
        raise NotImplementedError("in distribution dataset should be CIFAR10 or CIFAR100.")

    distribution = create_pathological_distribution(class_per_client, num_client, num_class, seed)
    id_train_split = split_by_distribution(np.array(train_data.targets), distribution)
    train_datasets = [Subset(train_data, id_train_split[idx]) for idx in range(num_client)]

    logging.info(
        f"-------- pathological distribution with {class_per_client} classes per client, {num_client} clients --------"
    )
    logging.info(f"in-distribution train datasets: {[len(dataset) for dataset in train_datasets]}")

    return train_datasets, num_class


def dirichlet_load_test(dataset_path, id_dataset, num_client, alpha, corrupt_list, seed):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    if (id_dataset == "cifar10") or (id_dataset == "cifar10_fourier_aug"):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_data = CIFAR10(root=dataset_path, download=True, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(CIFAR10_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 10
    elif (id_dataset == "cifar100") or (id_dataset == "cifar100_fourier_aug"):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_data = CIFAR100(root=dataset_path, download=True, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(CIFAR100_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 100
    elif id_dataset == "tinyimagenet" or id_dataset == "tinyimagenet_fourier_aug":
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_data = TinyImageNet(root=dataset_path, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(TinyImageNet_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 200
    else:
        raise NotImplementedError("in distribution dataset should be CIFAR10 or CIFAR100.")

    distribution = create_dirichlet_distribution(alpha, num_client, num_class, seed)
    id_split = split_by_distribution(np.array(test_data.targets), distribution)
    cor_split = split_by_distribution(np.array(cor_test[0].targets), distribution)
    id_datasets = [Subset(test_data, id_split[idx]) for idx in range(num_client)]
    cor_datasets = [
        {cor_type: Subset(cor_test[idx], cor_split[client_idx]) for idx, cor_type in enumerate(corrupt_list)}
        for client_idx in range(num_client)
    ]

    logging.info(f"-------- dirichlet distribution with alpha = {alpha}, {num_client} clients --------")
    logging.info(f"in-distribution test datasets: {[len(dataset) for dataset in id_datasets]}")
    return id_datasets, cor_datasets, num_class


def pathological_load_test(dataset_path, id_dataset, num_client, class_per_client, corrupt_list, seed):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if id_dataset == "cifar10":
        test_data = CIFAR10(root=dataset_path, download=True, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(CIFAR10_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 10
    elif id_dataset == "cifar100":
        test_data = CIFAR100(root=dataset_path, download=True, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(CIFAR100_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 100
    elif id_dataset == "tinyimagenet":
        test_data = TinyImageNet(root=dataset_path, train=False, transform=trans)
        cor_test = []
        for idx, cor_type in enumerate(corrupt_list):
            cor_test.append(TinyImageNet_CORRUPT(root=dataset_path, cortype=cor_type, transform=trans))
        num_class = 200
    else:
        raise NotImplementedError("in distribution dataset should be CIFAR10 or CIFAR100.")

    distribution = create_pathological_distribution(class_per_client, num_client, num_class, seed)
    id_split = split_by_distribution(np.array(test_data.targets), distribution)
    cor_split = split_by_distribution(np.array(cor_test[0].targets), distribution)
    id_datasets = [Subset(test_data, id_split[idx]) for idx in range(num_client)]
    cor_datasets = [
        {cor_type: Subset(cor_test[idx], cor_split[client_idx]) for idx, cor_type in enumerate(corrupt_list)}
        for client_idx in range(num_client)
    ]

    logging.info(
        f"-------- pathological distribution with {class_per_client} classes per client, {num_client} clients --------"
    )
    logging.info(f"in-distribution test datasets: {[len(dataset) for dataset in id_datasets]}")
    return id_datasets, cor_datasets, num_class


def load_test_ood(dataset_path, ood_dataset, seed, partial):
    random_number_generator = np.random.default_rng(seed)
    ood_data = load_ood_dataset(dataset_path, ood_dataset)

    if partial:
        idx = random.sample([i for i in range(len(ood_data))], int(0.2 * len(ood_data)))
        ood_data = Subset(ood_data, idx)
        logging.info(f"out of distribution test dataset's length: {len(ood_data)}")
        return ood_data
    else:
        logging.info(f"out of distribution test dataset's length: {len(ood_data)}")
        return ood_data
