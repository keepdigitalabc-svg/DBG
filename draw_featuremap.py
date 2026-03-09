import argparse
import os
from dataselect import build_pseudo_labeled_extend_dataset, save_filtered_images
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

from dataset import get_dataset224_longtail
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100, force_balance_after_merge, unforce_balance_after_merge,unforce_balance_after_merge_05, FilenameAsLabelDataset, FilenameAsLabelDataset1
from hard_adapter import unforce_balance_after_merge_soft
import logging
from tqdm import tqdm
from datetime import datetime
import random
import numpy as np
from resnet import resnet32
from torch.utils.data import Dataset
import random
import numpy as np
from collections import defaultdict
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
parser.add_argument('--sample_type', type=str, default='unbalanced')
parser.add_argument('--device', type=str, default='cuda:5')
parser.add_argument('--epsilon', type=str, default='1.0')
parser.add_argument('--attack_sample_type', type=str, default='m2l')
parser.add_argument('--noned', type=str, default="False")
args = parser.parse_args()

# 参数解包
dataset = args.dataset
attack = args.attack
imbalanced = args.imbalance
network = args.network
sample_type = args.sample_type
epsilon = args.epsilon
if args.noned == "True":
    noned = True
else:
    noned = False
# 类别数判断
num_classes = 10 if dataset == 'cifar-10' else 100
if dataset == 'imagenet200':
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
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
    # transforms.Normalize((0.485, 0.456, 0.406),
    #                      (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
    # transforms.Normalize((0.485, 0.456, 0.406),
    #                      (0.229, 0.224, 0.225))
])

# 原始训练集
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
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                           transform=transform_test)
elif dataset=="cifar-100":
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
if imbalanced == '0.01':
    imb = '001'
else:
    imb = '01'
# 对抗样本扩展数据集
if noned:
    extended_path = f'./{dataset}_{imbalanced}_advtrain_none'
else:
    extended_path = f'./{dataset}_{imbalanced}_advtrain'
if not os.path.exists(extended_path):
    raise FileNotFoundError(f"扩展数据集路径不存在: {extended_path}")

extended_dataset = FilenameAsLabelDataset(
    root=extended_path,
    transform=transform_test,
    num_classes=num_classes
)
ext_pth = f"relabel55-99_extended_hard_imagenet200_0.01"
# ext_pth = f'./cifar-{num_classes}_{imbalanced}_advtrain_targettmax0.6alike'
print(ext_pth)
extended_dataset1 = FilenameAsLabelDataset(
    # root=f'./{dataset}_{imbalanced}_advtrain_targettmax0.6alike',
    root="./"+ext_pth,
    transform=transform_test,
    num_classes=num_classes
)
extended_dataset2 = FilenameAsLabelDataset1(
    root=f'./relabel55-98_extended_hard0.90.01un',
    transform=transform_test,
    num_classes=num_classes
)

model = resnet32(num_classes=num_classes)
model.load_state_dict(torch.load(f'./classify/best_resnet32_{dataset}_imbalanced{imbalanced}_la.pth', map_location='cpu'))
# model.load_state_dict(torch.load(f'./resnet_finetune/extended_hard0.6top80.pth', map_location='cpu'))
model.eval()
# extended_dataset1 = build_pseudo_labeled_extend_dataset(
#     base_dataset=train_dataset_base,
#     extend_dataset=extended_dataset1,
#     model=model,
#     num_classes=num_classes,
#     device=torch.device(device),
#     feat_idx=-1,
#     feat_norm=True,
#     metric="cosine",
#     q_low=2.0,
#     q_high=95.0,
#     temperature=0.1
# )
logging.info("Loaded adversarial dataset: {extended_path}")
logging.info(f"扩展数据样本数: {len(extended_dataset)}")
# save_filtered_images(extended_dataset1, out_dir_hard = f"extended_hard_{dataset}_{imbalanced}_293", out_dir_soft="extended_softfused")
# if True:
#     a = 1/0
# 合并采样方式选择
if sample_type == "balanced":
    train_dataset = force_balance_after_merge(train_dataset_base, extended_dataset)
elif sample_type == "unbalanced":
#     train_dataset, fid, mmd,_ = unforce_balance_after_merge_soft(
#     base_dataset=train_dataset_base,
#     extended_dataset=extended_dataset1,       # 注意：这里传的是 plds
#     num_classes=num_classes,
#     pick_from_hard_only=True     # 只从 is_hard=1 的扩展样本里补，稳一点
# )
    # train_dataset, fid, mmd, _ = unforce_balance_after_merge_soft(
    #     base_dataset=train_dataset,
    #     extended_dataset=extended_dataset,  # 注意：这里传的是 plds
    #     num_classes=num_classes,
    #     pick_from_hard_only=True  # 只从 is_hard=1 的扩展样本里补，稳一点
    # )
    train_dataset, fid, mmd = unforce_balance_after_merge_05(train_dataset_base, extended_dataset1, testset, device=device,
                                                          save = f'./resnet_finetune/best_{network}_{dataset}_{sample_type}_{attack}{args.attack_sample_type}_{imbalanced}-{epsilon}')
    # train_dataset, fid, mmd = unforce_balance_after_merge(train_dataset, extended_dataset2, testset, device=device,
    #                                                       save = f'./resnet_finetune/best_{network}_{dataset}_{sample_type}_{attack}{args.attack_sample_type}_{imbalanced}-{epsilon}')
    train_dataset, fid, mmd = unforce_balance_after_merge(train_dataset, extended_dataset, testset, device=device,
                                                          save=f'./resnet_finetune/best_{network}_{dataset}_{sample_type}_{attack}{args.attack_sample_type}_{imbalanced}-{epsilon}')
# train_dataset = ConcatDataset([train_dataset, extended_dataset1])
elif sample_type == "concated":
    train_dataset = ConcatDataset([train_dataset_base, extended_dataset])
elif sample_type == "test":
    train_dataset = train_dataset_base
else:
    raise ValueError("sample_type must be one of: balanced, unbalanced, concated")
# print("=== 合并后各类样本数 ===")
# total = 0
# for c in range(num_classes):
#     cnt = mmd.get(c, 0)
#     print(f"Class {c:03d}: {cnt}")
#     total += cnt
# print(f"Total: {total} (len(merged_dataset) = {len(mmd)})")

trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
# 测试集

testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

# ----------------------
# ✅ 网络模型
# ----------------------
if network == "resnet":
    net = models.resnet34(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
else:
    net = resnet32(num_classes=num_classes)
net = net.to(device)
logging.info("Loaded network and moved to device")

def lr_lambda(epoch):
    if epoch < 5:
        return(epoch + 1) / 5
    if epoch >= 180:
        return 0.01  # Decay by 0.1 at epoch 180
    elif epoch >= 160:
        return 0.1  # Decay by 0.1 at epoch 160
    else:
        return 1  # No decay before epoch 160
# ----------------------
# ✅ 损失函数、优化器
# ----------------------
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
# ----------------------
# ✅ 训练过程
# ----------------------
best_acc = 0.0

def get_lambda_t(epoch, warmup=10, start=0.1, end=1.0):
    if epoch <= 0: return start
    if epoch >= warmup: return end
    r = epoch / float(warmup)
    return start + (end - start) * r
# if noned:
#     save_path = f'./resnet_finetune/best_{network}_{dataset}_{sample_type}_{attack}{args.attack_sample_type}_{imbalanced}-{epsilon}_none.pth'
# else:
#     save_path = f'./resnet_finetune/best_{network}_{dataset}_{sample_type}_{attack}{args.attack_sample_type}_{imbalanced}-{epsilon}.pth'
if noned:
    save_path = f'./'+ ext_pth +'_none_ablation.pth'
else:
    save_path = f'./'+ ext_pth +'_ours.pth'

for epoch in range(300):
    net.train()
    running_loss = 0.0

    for inputs,labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/100"):
        # try:
        #     neg_inputs, neg_labels = next(neg_iter)
        # except StopIteration:
        #     neg_iter = iter(negloader)
        #     neg_inputs, neg_labels = next(neg_iter)
        # neg_inputs, neg_labels = neg_inputs.to(device), neg_labels.to(device)
        # inputs, labels, weights = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs,_ = net(inputs)
        # outputs_neg, _ = net(neg_inputs)
        # loss_neg = negative_ce_loss_from_logits(outputs_neg, neg_labels, reduction="mean")
        # per_sample_loss = ce_none(outputs, labels)
        # loss = (per_sample_loss * weights).sum() / weights.sum()
        loss = criterion(outputs, labels).mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    is_training = net.training
    # 验证
    net.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 逐个样本统计每类准确率
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    acc = 100 * correct / total
    logging.info(f"Epoch {epoch + 1}, Loss: {running_loss:.3f}, Val Acc: {acc:.2f}%")

    for i in range(num_classes):
        if class_total[i] > 0:
            acc_i = 100 * class_correct[i] / class_total[i]
            logging.info(f"Class {i} Accuracy: {acc_i:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")

    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), save_path)
        logging.info(f"✅ Saved best model with acc: {acc:.2f}%")
logging.info(f"✅ Best accuracy: {best_acc:.2f}%")
logging.info("训练完成！")