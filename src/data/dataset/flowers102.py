from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import PIL.Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

        cat_to_name = {
            "21": "fire lily",
            "3": "canterbury bells",
            "45": "bolero deep blue",
            "1": "pink primrose",
            "34": "mexican aster",
            "27": "prince of wales feathers",
            "7": "moon orchid",
            "16": "globe-flower",
            "25": "grape hyacinth",
            "26": "corn poppy",
            "79": "toad lily",
            "39": "siam tulip",
            "24": "red ginger",
            "67": "spring crocus",
            "35": "alpine sea holly",
            "32": "garden phlox",
            "10": "globe thistle",
            "6": "tiger lily",
            "93": "ball moss",
            "33": "love in the mist",
            "9": "monkshood",
            "102": "blackberry lily",
            "14": "spear thistle",
            "19": "balloon flower",
            "100": "blanket flower",
            "13": "king protea",
            "49": "oxeye daisy",
            "15": "yellow iris",
            "61": "cautleya spicata",
            "31": "carnation",
            "64": "silverbush",
            "68": "bearded iris",
            "63": "black-eyed susan",
            "69": "windflower",
            "62": "japanese anemone",
            "20": "giant white arum lily",
            "38": "great masterwort",
            "4": "sweet pea",
            "86": "tree mallow",
            "101": "trumpet creeper",
            "42": "daffodil",
            "22": "pincushion flower",
            "2": "hard-leaved pocket orchid",
            "54": "sunflower",
            "66": "osteospermum",
            "70": "tree poppy",
            "85": "desert-rose",
            "99": "bromelia",
            "87": "magnolia",
            "5": "english marigold",
            "92": "bee balm",
            "28": "stemless gentian",
            "97": "mallow",
            "57": "gaura",
            "40": "lenten rose",
            "47": "marigold",
            "59": "orange dahlia",
            "48": "buttercup",
            "55": "pelargonium",
            "36": "ruby-lipped cattleya",
            "91": "hippeastrum",
            "29": "artichoke",
            "71": "gazania",
            "90": "canna lily",
            "18": "peruvian lily",
            "98": "mexican petunia",
            "8": "bird of paradise",
            "30": "sweet william",
            "17": "purple coneflower",
            "52": "wild pansy",
            "84": "columbine",
            "12": "colt's foot",
            "11": "snapdragon",
            "96": "camellia",
            "23": "fritillary",
            "50": "common dandelion",
            "44": "poinsettia",
            "53": "primula",
            "72": "azalea",
            "65": "californian poppy",
            "80": "anthurium",
            "76": "morning glory",
            "37": "cape flower",
            "56": "bishop of llandaff",
            "60": "pink-yellow dahlia",
            "82": "clematis",
            "58": "geranium",
            "75": "thorn apple",
            "41": "barbeton daisy",
            "95": "bougainvillea",
            "43": "sword lily",
            "83": "hibiscus",
            "78": "lotus",
            "88": "cyclamen",
            "94": "foxglove",
            "81": "frangipani",
            "74": "rose",
            "89": "watercress",
            "73": "water lily",
            "46": "wallflower",
            "77": "passion flower",
            "51": "petunia",
        }

        self.classes = [cat_to_name[key] for key in sorted(cat_to_name.keys(), key=int)]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)
