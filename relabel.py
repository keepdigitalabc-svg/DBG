import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

from dataset import get_dataset224_longtail
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100, force_balance_after_merge, unforce_balance_after_merge
import logging
from tqdm import tqdm
from datetime import datetime
import random
import numpy as np
from torch.utils.data import Dataset, Subset, ConcatDataset
from classify.resnet import resnet32
class FilenameAsLabelDataset(Dataset):
    """
    从目录结构拿到原标签 org_label；从文件名最后一个'_'分段中的数字提取 extra_label。
    例如: img_1+1024_71.png -> extra_label = 71
    返回: (image, org_label, extra_label)
    """
    def __init__(self, root, transform=None, num_classes=10):
        self.root = root
        self.transform = transform
        self.num_classes = num_classes
        self.samples = []
        self.class_counts = [0] * num_classes  # 目录标签（原标签）的计数

        for dirpath, _, filenames in os.walk(root):
            label_str = os.path.basename(dirpath)
            if not label_str.isdigit():
                continue  # 忽略非数字命名的文件夹
            org_label = int(label_str)
            if org_label >= num_classes:
                continue

            for fname in filenames:
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                full_path = os.path.join(dirpath, fname)

                # --- 提取 extra_label：取'_'最后一段里的“末尾数字” ---
                # 举例: 'img_1+1024_71.png' -> last_part = '71.png' -> 提取 71
                last_part = fname.rsplit('_', 1)[-1]
                # 去掉扩展名
                last_part = os.path.splitext(last_part)[0]
                # 提取末尾连续数字
                i = len(last_part) - 1
                while i >= 0 and last_part[i].isdigit():
                    i -= 1
                digits = last_part[i+1:]
                if digits == '':
                    # 若没找到数字，就退化为使用 org_label 作为 extra_label（也可改成 -1 表示无）
                    extra_label = org_label
                else:
                    extra_label = int(digits)

                # 规范化 extra_label 到 [0, num_classes) 范围之外则回退为 org_label
                if not (0 <= extra_label < self.num_classes):
                    extra_label = org_label

                self.samples.append((full_path, org_label, extra_label))
                self.class_counts[org_label] += 1

        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, org_label, extra_label = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 返回三个元素：图像、原标签、额外标签
        return image, org_label, extra_label
def seed_everything(seed=42):
 random.seed(seed)
 os.environ["PYTHONHASHSEED"] = str(seed)
 np.random.seed(seed)
 torch.manual_seed(seed)
 torch.cuda.manual_seed(seed)
 torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic = True
 torch.backends.cudnn.benchmark = False
seed_everything()
# ----------------------
# ✅ argparse 参数配置
# ----------------------
parser = argparse.ArgumentParser(description="Training with adversarial + imbalanced data")
parser.add_argument('--dataset', type=str, default='cifar-100')
parser.add_argument('--attack', type=str, default='Deepfool')
parser.add_argument('--imbalance', type=str, default='0.01')
parser.add_argument('--network', type=str, default='resnet32')  # 可扩展
parser.add_argument('--sample_type', type=str, default='concated')
parser.add_argument('--device', type=str, default='cuda:5')
parser.add_argument('--epsilon', type=str, default='2.0')
parser.add_argument('--attack_sample_type', type=str, default='m2l')

# === NEW: 置信度与间隔阈值、温度、是否使用 LA ===
parser.add_argument('--conf_thres', type=float, default=0.95, help='top-1 置信度阈值')
parser.add_argument('--margin_thres', type=float, default=0.30, help='top1 - top2 的最小间隔')
parser.add_argument('--temperature', type=float, default=1.0, help='softmax 温度系数 tau')
parser.add_argument('--use_la', action='store_true', help='使用 logit adjustment 后的 logits 计算概率')
parser.add_argument('--per_class_conf', default=True, help='按预测类样本量动态设置置信度阈值')
parser.add_argument('--conf_min', type=float, default=0.5, help='样本最多的类使用的置信度阈值')
parser.add_argument('--conf_max', type=float, default=0.9, help='样本最少的类使用的置信度阈值')
parser.add_argument('--per_class_margin', default=True, help='按预测类样本量动态设置间隔阈值')
parser.add_argument('--margin_min', type=float, default=0.5, help='样本最多的类使用的间隔阈值')
parser.add_argument('--margin_max', type=float, default=0.9, help='样本最少的类使用的间隔阈值')
args = parser.parse_args()

args = parser.parse_args()

# 参数解包
dataset = args.dataset
attack = args.attack
imbalanced = args.imbalance
network = args.network
sample_type = args.sample_type
epsilon = args.epsilon
device = args.device
# 类别数判断
num_classes = 10 if dataset == 'cifar-10' else 100
if dataset=="imagenet200":
    num_classes=200

# ----------------------
# ✅ 日志配置
# ----------------------
os.makedirs('resnet_finetune', exist_ok=True)
log_filename = f"./resnet_finetune/imbalance_{network}_{imbalanced}-{epsilon}_{attack}{args.attack_sample_type}_{dataset}_{sample_type}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)

# ----------------------
# ✅ 数据准备
# ----------------------
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    # transforms.Resize((256, 256)),
    # transforms.RandomCrop((224, 224), padding=4),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
class DeNormalize(object):
    def __init__(self, mean, std):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        tensor: (C, H, W)，已经Normalize过的图像
        return: (C, H, W)，反归一化回0~1范围
        """
        return tensor * self.std[:, None, None] + self.mean[:, None, None]


# 你的 mean/std
denormalize = DeNormalize(
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010)
)
transform_test = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# 原始训练集
if dataset == "cifar-10":
    train_dataset_base = IMBALANCECIFAR10(
        root='../data/imbalanced_cifar-10',
        train=True,
        download=False,
        transform=transform_train,
        imb_type='exp',
        imb_factor=float(imbalanced),
        rand_number=42
    )
    testset = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=False, download=False,
                                           transform=transform_test)
elif dataset == "cifar-100":
    train_dataset_base = IMBALANCECIFAR100(
        root='./data',
        train=True,
        download=False,
        transform=transform_train,
        imb_type='exp',
        imb_factor=float(imbalanced),
        rand_number=42
    )
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                            transform=transform_test)
else:
    train_dataset_base, testset, classnames, per_cls = get_dataset224_longtail(
        data_root="./tiny-imagenet-200",
        # label_json_path="/data/lhz/data/imagenet100/ImageNet_class_index.json",  # 你的 JSON 路径
        imb_type="exp",  # 'exp' | 'step' | 'none'
        imb_factor=float(imbalanced),  # 尾部/头部比例
        rand_number=42
    )
    trainloader = DataLoader(train_dataset_base, batch_size=128, shuffle=True, num_workers=4)

    # trainset = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=True, download=True, transform=transform_train)
    # trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

logging.info("Loaded base dataset")
classnum = train_dataset_base.get_cls_num_list()
# 转成 numpy
# classnum = np.array(per_cls, dtype=np.float32)
# === NEW: 生成 per-class 阈值（线性插值）：count 多 → 阈值低；count 少 → 阈值高
cls_counts = torch.tensor(classnum, dtype=torch.float32)  # shape: [num_classes]
c_min, c_max = float(cls_counts.min().item()), float(cls_counts.max().item())
if c_max == c_min:
    # 所有类样本数相等，退化为常数阈值（取各自区间的中点）
    conf_thres_vec = torch.full((num_classes,), (args.conf_min + args.conf_max) / 2.0, dtype=torch.float32)
    margin_thres_vec = torch.full((num_classes,), (args.margin_min + args.margin_max) / 2.0, dtype=torch.float32)
else:
    # 线性映射到 [0,1]，再映射到阈值区间；count 越大，权重越接近 1 → 阈值越接近 min
    lin = (cls_counts - c_min) / (c_max - c_min)          # [0,1], 最少类=0, 最多类=1
    conf_thres_vec = args.conf_max - lin * (args.conf_max - args.conf_min)
    margin_thres_vec = args.margin_max - lin * (args.margin_max - args.margin_min)

# 放到 device，方便索引
conf_thres_vec = conf_thres_vec.to(device)
margin_thres_vec = margin_thres_vec.to(device)
# 计算先验概率
# class_prior = classnum / classnum.sum()

# logit_adjustment = 1.0 * torch.log(torch.from_numpy(class_prior)).to(device)
# 对抗样本扩展数据集
# extended_path = f'./attack/adversarial_samples_{dataset}_{attack}{args.attack_sample_type}_{network}{imbalanced}-{args.epsilon}'
# extended_p = f"cifar-{num_classes}_{imbalanced}_advtrain_targettmax0.6alike"
extended_p = f"extended_hard_{dataset}_{imbalanced}_595"
extended_path = f"./"+extended_p
if not os.path.exists(extended_path):
    raise FileNotFoundError(f"扩展数据集路径不存在: {extended_path}")

extended_dataset = FilenameAsLabelDataset(
    root=extended_path,
    transform=transform_test,
    num_classes=num_classes
)
dataloader = DataLoader(extended_dataset, batch_size=128, shuffle=True, num_workers=4)
if network == "resnet":
    net = models.resnet34(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
else:
    net = resnet32(num_classes=num_classes)
weight_path1=f'./classify/best_resnet32_{dataset}_imbalanced{imbalanced}_la.pth'
weight_path2=f'./classify/best_resnet32_{dataset}_imbalanced1.pth'
weight_path3=f"./resnet_finetune/extended_hard0.6top80.pth"
net.load_state_dict(torch.load(weight_path1, map_location='cpu'))
net = net.to(device)
net.eval()

# class_counts = torch.tensor(extended_dataset.class_counts, dtype=torch.float)
tau = 1.0  # 可调
save_root = f"./relabel55-99_"+extended_p
os.makedirs(save_root, exist_ok=True)
from torchvision.utils import save_image
from torchvision.utils import save_image
import csv  # === NEW: 保存 CSV

# === NEW: 统计初始化（原始类 / 预测类 各一份） ===
def init_stats(nc):
    return {
        "sum_top1": torch.zeros(nc, dtype=torch.float64),
        "sum_margin": torch.zeros(nc, dtype=torch.float64),
        "count": torch.zeros(nc, dtype=torch.long),
        "hit_conf": torch.zeros(nc, dtype=torch.long),
        "hit_margin": torch.zeros(nc, dtype=torch.long),
        "hit_both": torch.zeros(nc, dtype=torch.long),
    }

stats_orig = init_stats(num_classes)  # 按 org_label 统计
stats_pred = init_stats(num_classes)  # 按 pred 统计
# 初始化一次（循环前）
final_count = torch.zeros(num_classes, dtype=torch.long)
# ========== 2. Label Assignment (重标注) ==========
with torch.no_grad():
    for batch_idx, (inputs, org_label, extra_label) in enumerate(tqdm(dataloader, desc="Relabeling")):
        inputs = inputs.to(device)
        org_label = org_label.to(device)

        # 模型预测
        outputs, _ = net(inputs)

        # 选择 logits（是否使用 LA）
        logits = outputs

        # softmax 概率与 top-2
        probs = torch.softmax(logits, dim=1)
        top2_vals, top2_idx = probs.topk(k=2, dim=1, largest=True, sorted=True)
        top1_prob = top2_vals[:, 0]
        top2_prob = top2_vals[:, 1]
        margin = top1_prob - top2_prob
        preds = top2_idx[:, 0]
        # === NEW: 使用“预测类”的 per-class 阈值（若不开关，则退化为全局阈值）
        if args.per_class_conf:
            conf_thres_per_sample = conf_thres_vec[preds]  # shape: [B]
        else:
            conf_thres_per_sample = torch.full_like(top1_prob, args.conf_thres)

        if args.per_class_margin:
            margin_thres_per_sample = margin_thres_vec[preds]  # shape: [B]
        else:
            margin_thres_per_sample = torch.full_like(margin, args.margin_thres)

        high_conf_mask = top1_prob >= conf_thres_per_sample
        large_margin_mask = margin >= margin_thres_per_sample
        both_mask = high_conf_mask & large_margin_mask

        for i in range(inputs.size(0)):
            o = int(org_label[i].item())
            p = int(preds[i].item())
            t1 = float(top1_prob[i].item())
            mg = float(margin[i].item())

            # 原始类视角
            stats_orig["sum_top1"][o] += t1
            stats_orig["sum_margin"][o] += mg
            stats_orig["count"][o] += 1
            stats_orig["hit_conf"][o] += int(high_conf_mask[i].item())
            stats_orig["hit_margin"][o] += int(large_margin_mask[i].item())
            stats_orig["hit_both"][o] += int(both_mask[i].item())

            # 预测类视角
            stats_pred["sum_top1"][p] += t1
            stats_pred["sum_margin"][p] += mg
            stats_pred["count"][p] += 1
            stats_pred["hit_conf"][p] += int(high_conf_mask[i].item())
            stats_pred["hit_margin"][p] += int(large_margin_mask[i].item())
            stats_pred["hit_both"][p] += int(both_mask[i].item())

        # ========== 保存图片到对应类别文件夹 ==========
        for i in range(inputs.size(0)):
            high_conf = high_conf_mask[i].item()
            large_margin = large_margin_mask[i].item()

            # 原来的 target 规则：高置信+大间隔 -> 用预测类；否则用原标签
            candidate = int(preds[i].item()) if (high_conf and large_margin) else int(org_label[i].item())

            ori = int(org_label[i].item())
            aux = int(extra_label[i].item())  # 新增：额外标签

            # 新增约束：candidate 必须是原标签或额外标签其中之一，否则不保存
            if candidate != ori and candidate != aux:
                continue  # 直接跳过，不保存
                # candidate = ori

            # 通过约束后再保存
            save_dir = os.path.join(save_root, str(candidate))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{batch_idx}_{i}.png")

            img = denormalize(inputs[i].detach().cpu().clone())
            img = torch.clamp(img, 0, 1)
            save_image(img, save_path)

            # 只有在“高置信+大间隔 且 通过额外标签约束”时才计入 final_count
            if (high_conf and large_margin):
                final_count[candidate] += 1
# === NEW: 汇总并打印/落盘 ===
def finalize_and_dump(name, stats, out_csv_path):
    cnt = stats["count"].to(torch.float64).clamp_min(1)  # 防止除0
    mean_top1 = (stats["sum_top1"] / cnt).numpy()
    mean_margin = (stats["sum_margin"] / cnt).numpy()
    count = stats["count"].numpy()
    prop_conf = (stats["hit_conf"].to(torch.float64) / cnt).numpy()
    prop_margin = (stats["hit_margin"].to(torch.float64) / cnt).numpy()
    prop_both = (stats["hit_both"].to(torch.float64) / cnt).numpy()

    # 打印到日志
    logging.info(f"===== Per-class stats ({name}) =====")
    logging.info("class\tcount\tmean_top1\tmean_margin\tp_conf\tp_margin\tp_both")
    for c in range(len(count)):
        logging.info(f"{c}\t{int(count[c])}\t{mean_top1[c]:.4f}\t{mean_margin[c]:.4f}\t"
                     f"{prop_conf[c]:.4f}\t{prop_margin[c]:.4f}\t{prop_both[c]:.4f}")

    # 整体均值（按样本数加权）
    total = int(stats["count"].sum().item())
    overall_top1 = float(stats["sum_top1"].sum().item() / max(total, 1))
    overall_margin = float(stats["sum_margin"].sum().item() / max(total, 1))
    overall_p_conf = float(stats["hit_conf"].sum().item() / max(total, 1))
    overall_p_margin = float(stats["hit_margin"].sum().item() / max(total, 1))
    overall_p_both = float(stats["hit_both"].sum().item() / max(total, 1))
    logging.info(f"[OVERALL {name}] N={total} | mean_top1={overall_top1:.4f} | "
                 f"mean_margin={overall_margin:.4f} | p_conf={overall_p_conf:.4f} | "
                 f"p_margin={overall_p_margin:.4f} | p_both={overall_p_both:.4f}")

    # 保存 CSV
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "count", "mean_top1", "mean_margin",
                         "p_conf(>=conf_thres)", "p_margin(>=margin_thres)", "p_both"])
        for c in range(len(count)):
            writer.writerow([
                c, int(count[c]), f"{mean_top1[c]:.6f}", f"{mean_margin[c]:.6f}",
                f"{prop_conf[c]:.6f}", f"{prop_margin[c]:.6f}", f"{prop_both[c]:.6f}"
            ])
        # 追加一行 overall
        writer.writerow([
            "OVERALL", total, f"{overall_top1:.6f}", f"{overall_margin:.6f}",
            f"{overall_p_conf:.6f}", f"{overall_p_margin:.6f}", f"{overall_p_both:.6f}"
        ])

# 调用汇总输出
# finalize_and_dump("by_original", stats_orig, os.path.join(save_root, "stats_by_original.csv"))
# finalize_and_dump("by_pred", stats_pred, os.path.join(save_root, "stats_by_pred.csv"))




# 在 finalize_and_dump 之后打印
logging.info("===== Final reassigned counts (after thresholds) =====")
for c in range(num_classes):
    logging.info(f"class {c}: {final_count[c].item()}")
logging.info(f"Total reassigned images: {final_count.sum().item()}")
print(f"✅ 重标注完成，统计已保存到 {save_root}/stats_by_original.csv 与 stats_by_pred.csv")
