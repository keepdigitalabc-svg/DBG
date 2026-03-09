import argparse
import os
import logging
from tqdm import tqdm
from datetime import datetime
import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

from imbalance_cifar import (
    IMBALANCECIFAR10, IMBALANCECIFAR100,
    force_balance_after_merge,
    unforce_balance_after_merge,
    unforce_balance_after_merge_05,
    FilenameAsLabelDataset, FilenameAsLabelDataset1
)

# ----------------------
# reproducibility
# ----------------------
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
# argparse
# ----------------------
parser = argparse.ArgumentParser(description="Training ViT-Base with adversarial + imbalanced data")
parser.add_argument('--dataset', type=str, default='cifar-100', choices=['cifar-10', 'cifar-100'])
parser.add_argument('--attack', type=str, default='Deepfool')
parser.add_argument('--imbalance', type=str, default='0.005')
parser.add_argument('--sample_type', type=str, default='unbalanced', choices=['balanced', 'unbalanced', 'concated', 'test'])
parser.add_argument('--device', type=str, default='cuda:5')
parser.add_argument('--epsilon', type=str, default='1.0')
parser.add_argument('--attack_sample_type', type=str, default='m2l')
parser.add_argument('--noned', type=str, default="False")   # 是否使用 none 扩展目录
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)  # ViT 占显存更大，如 OOM 请降到 64/32
parser.add_argument('--lr', type=float, default=1e-4)       # ViT 常用学习率
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--warmup_epochs', type=int, default=10)
args = parser.parse_args()

dataset = args.dataset
attack = args.attack
imbalanced = args.imbalance
sample_type = args.sample_type
epsilon = args.epsilon
noned = (args.noned == "True")
num_classes = 10 if dataset == 'cifar-10' else 100
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# ----------------------
# logging
# ----------------------
os.makedirs('vit_finetune', exist_ok=True)
log_filename = f"./vit_finetune/imbalance_vitb16_{imbalanced}-{epsilon}_{attack}{args.attack_sample_type}_{dataset}_{sample_type}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)

# ----------------------
# data
# ----------------------
weights = ViT_B_16_Weights.IMAGENET1K_V1
preprocess = weights.transforms()

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

if dataset == "cifar-10":
    train_dataset_base = IMBALANCECIFAR10(
        root='./data',
        train=True,
        download=False,
        transform=transform_train,
        imb_type='exp',
        imb_factor=float(imbalanced),
        rand_number=42
    )
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
else:
    train_dataset_base = IMBALANCECIFAR100(
        root='./data',
        train=True,
        download=False,
        transform=transform_train,
        imb_type='exp',
        imb_factor=float(imbalanced),
        rand_number=42
    )
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)

logging.info("Loaded base dataset")

if noned:
    extended_path = f'./cifar-{num_classes}_{imbalanced}_advtrain_none'
else:
    extended_path = f'./cifar-{num_classes}_{imbalanced}_advtrain'
if not os.path.exists(extended_path):
    raise FileNotFoundError(f"扩展数据集路径不存在: {extended_path}")

extended_dataset = FilenameAsLabelDataset(
    root=extended_path,
    transform=transform_test,
    num_classes=num_classes
)

ext_pth = f"relabel55-99_extended_hard_{num_classes}_{imbalanced}"
extended_dataset1 = FilenameAsLabelDataset(
    root="./"+ext_pth,
    transform=transform_test,
    num_classes=num_classes
)
extended_dataset2 = FilenameAsLabelDataset1(
    root=f'./relabel55-98_extended_hard0.90.01un',
    transform=transform_test,
    num_classes=num_classes
)

logging.info(f"Loaded adversarial dataset: {extended_path}")
logging.info(f"扩展数据样本数: {len(extended_dataset)}")

if sample_type == "balanced":
    train_dataset = force_balance_after_merge(train_dataset_base, extended_dataset)
elif sample_type == "unbalanced":
    train_dataset, fid, mmd = unforce_balance_after_merge(
        train_dataset_base, extended_dataset, testset, device=device,
        save=f'./vit_finetune/best_vitb16_{dataset}_{sample_type}_{attack}{args.attack_sample_type}_{imbalanced}-{epsilon}'
    )
    # train_dataset, fid, mmd = unforce_balance_after_merge(
    #     train_dataset, extended_dataset, testset, device=device,
    #     save=f'./vit_finetune/best_vitb16_{dataset}_{sample_type}_{attack}{args.attack_sample_type}_{imbalanced}-{epsilon}'
    # )
elif sample_type == "concated":
    train_dataset = ConcatDataset([train_dataset_base, extended_dataset])
elif sample_type == "test":
    train_dataset = train_dataset_base
else:
    raise ValueError("sample_type must be one of: balanced, unbalanced, concated, test")

trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# ----------------------
# model (ViT-Base/16)
# ----------------------
net = vit_b_16()
# 替换分类头
if hasattr(net, "heads") and hasattr(net.heads, "head"):
    in_features = net.heads.head.in_features
    net.heads.head = nn.Linear(in_features, num_classes)
elif hasattr(net, "classifier"):  # 兼容早期接口
    in_features = net.classifier.in_features
    net.classifier = nn.Linear(in_features, num_classes)
else:
    raise RuntimeError("Unexpected ViT head structure.")
net = net.to(device)
logging.info("Loaded ViT-Base/16 (ImageNet pretrained) and moved to device")

# ----------------------
# loss & optimizer & scheduler
# ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def lr_lambda(epoch):
    # 线性 warmup 到 warmup_epochs，然后余弦退火到 0
    if epoch < args.warmup_epochs:
        return float(epoch + 1) / float(max(1, args.warmup_epochs))
    progress = (epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ----------------------
# train & eval
# ----------------------
best_acc = 0.0
save_path = f'./vit_pth/'+extended_path.strip('./')+'_.pth'

for epoch in range(args.epochs):
    net.train()
    running_loss = 0.0

    for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    # eval
    net.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    acc = 100.0 * correct / max(1, total)
    logging.info(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}, Val Acc: {acc:.2f}%")

    for i in range(num_classes):
        if class_total[i] > 0:
            acc_i = 100.0 * class_correct[i] / class_total[i]
            logging.info(f"Class {i} Accuracy: {acc_i:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")

    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), save_path)
        logging.info(f"✅ Saved best model with acc: {acc:.2f}% -> {save_path}")

logging.info(f"✅ Best accuracy: {best_acc:.2f}%")
logging.info("训练完成！")
