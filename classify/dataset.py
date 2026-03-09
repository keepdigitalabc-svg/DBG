
import numpy as np
import torchvision.datasets as datasets



class ImbalanceCIFAR10(datasets.CIFAR100):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class ImbalanceCIFAR100(datasets.CIFAR100):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super(ImbalanceCIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.num_per_cls_dict = dict()
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

import os
import numpy as np
from torchvision import datasets, transforms

IMAGENET_MEAN, IMAGENET_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def build_transforms_224():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    transform_val = transforms.Compose([
        # transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform_train, transform_val


class ImbalanceImageFolder(datasets.ImageFolder):
    """
    在 ImageFolder(train/) 上生成长尾训练集：
      - imb_type: 'exp'（指数长尾），'step'（阶梯长尾），'none'（均衡）
      - imb_factor: 小类/大类样本数比（如 0.01 表示最尾类≈头部类的 1%）
      - rand_number: 随机种子
    """
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0,
                 transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.samples = getattr(self, "samples", getattr(self, "imgs", []))
        assert len(self.samples) > 0, f"No samples found in {root}"
        self.targets = [s[1] for s in self.samples]

        rng = np.random.default_rng(rand_number)

        classes = np.unique(np.array(self.targets, dtype=np.int64))
        cls_num = len(classes)

        # 以训练集中“最大类样本数”作为头部规模
        per_cls_counts = [sum(t == c for t in self.targets) for c in range(cls_num)]
        img_max = max(per_cls_counts)

        img_num_per_cls = self._get_img_num_per_cls(cls_num, imb_type, imb_factor, img_max)

        new_samples, new_targets = [], []
        targets_np = np.array(self.targets, dtype=np.int64)

        for c, the_img_num in zip(range(cls_num), img_num_per_cls):
            idx = np.where(targets_np == c)[0]
            take = min(len(idx), int(the_img_num))
            if take <= 0:
                continue
            rng.shuffle(idx)
            selec_idx = idx[:take]
            for i in selec_idx:
                new_samples.append(self.samples[i])
                new_targets.append(c)

        self.samples = new_samples
        self.imgs = new_samples  # 兼容旧属性
        self.targets = new_targets

        self.num_per_cls_dict = {c: 0 for c in range(cls_num)}
        for t in self.targets:
            self.num_per_cls_dict[t] += 1

    @staticmethod
    def _get_img_num_per_cls(cls_num, imb_type, imb_factor, img_max):
        img_num_per_cls = []
        if imb_type == 'exp':
            for i in range(cls_num):
                num = img_max * (imb_factor ** (i / (cls_num - 1.0))) if cls_num > 1 else img_max
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            half = cls_num // 2
            img_num_per_cls = [int(img_max)] * half + [int(img_max * imb_factor)] * (cls_num - half)
        else:  # 'none'
            img_num_per_cls = [int(img_max)] * cls_num
        return img_num_per_cls

    def get_cls_num_list(self):
        cls_num = len(self.num_per_cls_dict)
        return [self.num_per_cls_dict.get(i, 0) for i in range(cls_num)]


def get_dataset224_longtail(
    data_root,
    imb_type='exp',
    imb_factor=0.01,
    rand_number=0,
    train_subdir='train',
    val_subdir='val'
):
    """
    直接传参，不用 args：
      - data_root: 根目录（内含 train/ 与 val/）
      - imb_type: 'exp' | 'step' | 'none'
      - imb_factor: float（例 0.01）
      - rand_number: int 随机种子
      - train_subdir/val_subdir: 子目录名（默认 train/、val/）

    返回:
      train_imb (Dataset), val_all (Dataset), class_names (list[str]), per_class_counts (list[int])
    """
    transform_train, transform_val = build_transforms_224()

    train_imb = ImbalanceImageFolder(
        root=os.path.join(data_root, train_subdir),
        imb_type=imb_type,
        imb_factor=imb_factor,
        rand_number=rand_number,
        transform=transform_train,
    )

    val_all = datasets.ImageFolder(
        root=os.path.join(data_root, val_subdir),
        transform=transform_val
    )

    class_names = train_imb.classes
    per_class_counts = train_imb.get_cls_num_list()

    return train_imb, val_all, class_names, per_class_counts
