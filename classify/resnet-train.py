import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models
from torch.utils.data import DataLoader
import os
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import argparse
from resnet import resnet32, resnet50
import random
import numpy as np
from tqdm import tqdm
from dataset import get_dataset224_longtail
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
# 设置设备
parser = argparse.ArgumentParser(description="Training with imbalanced data")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument('--dataset', type=str, default='imagenet200')
parser.add_argument('--net', type=str, default='resnet32')
parser.add_argument('--imb', type=float, default=0.01)
# 数据预处理
args = parser.parse_args()
if args.dataset == 'cifar-100' or args.dataset == 'imagenet100':
    num_classes = 100
elif args.dataset == 'imagenet200':
    num_classes = 200
else:
    num_classes = 10
# transform_train = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomCrop(224, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465),
#                          (0.2023, 0.1994, 0.2010))
# ])
#
# transform_test = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465),
#                          (0.2023, 0.1994, 0.2010))
# ])
transform_train = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    # transforms.Resize((64,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])
# 加载 CIFAR 数据
if args.dataset == 'cifar-100':
    train_dataset = IMBALANCECIFAR100(
        root='./data',  # 数据集存储路径
        train=True,  # 使用训练集
        download=True,  # 自动下载数据集
        transform=transform_train,  # 数据预处理
        imb_type='exp',  # 不平衡类型: 'exp'(指数)/'step'(阶梯)
        imb_factor=args.imb,  # 不平衡因子(最小类样本比例)
        rand_number=42  # 随机种子(确保可复现性)
    )
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    # trainset = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=True, download=True, transform=transform_train)
    # trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
elif args.dataset == 'cifar-10':
    train_dataset = IMBALANCECIFAR10(
        root='./data',              # 数据集存储路径
        train=True,                  # 使用训练集
        download=True,               # 自动下载数据集
        transform=transform_train,         # 数据预处理
        imb_type='exp',              # 不平衡类型: 'exp'(指数)/'step'(阶梯)
        imb_factor=args.imb,             # 不平衡因子(最小类样本比例)
        rand_number=42               # 随机种子(确保可复现性)
    )
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    # trainset = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=True, download=True, transform=transform_train)
    # trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
else:
    train_dataset, testset, classnames, per_cls = get_dataset224_longtail(
        data_root="../tiny-imagenet-200",
        # label_json_path="/data/lhz/data/imagenet100/ImageNet_class_index.json",  # 你的 JSON 路径
        imb_type="exp",  # 'exp' | 'step' | 'none'
        imb_factor=args.imb,  # 尾部/头部比例
        rand_number=0
    )
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    # trainset = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=True, download=True, transform=transform_train)
    # trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
# 加载预训练模型
if args.net == 'resnet32':
    net = resnet32(num_classes=num_classes)
elif args.net == 'resnet50':
    net = resnet50(num_classes=num_classes)
# 替换输出层（CIFAR-10 有 10 类）
# net.fc = nn.Linear(net.fc.in_features, 10)
net = net.to(device)
classnum = train_dataset.get_cls_num_list()
# 转成 numpy
classnum = np.array(classnum, dtype=np.float32)

# 计算先验概率
class_prior = classnum / classnum.sum()

logit_adjustment = 1.0 * torch.log(torch.from_numpy(class_prior)).to(device)
# 损失函数和优化器
def lr_lambda(epoch):
    if epoch < 5:
        return(epoch + 1) / 5
    if epoch >= 180:
        return 0.01  # Decay by 0.1 at epoch 180
    elif epoch >= 160:
        return 0.1  # Decay by 0.1 at epoch 160
    else:
        return 1  # No decay before epoch 160


criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.05)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
# 训练模型
best_acc = 0.0
save_path = f'./best_{args.net}_{args.dataset}_imbalanced{args.imb}_la.pth'
net.train()
runloss =[]
for epoch in range(200):  # 训练 10 个 epoch
    running_loss = 0.0
    runloss = []
    for inputs, labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, _ = net(inputs)
        outputs = outputs + logit_adjustment.unsqueeze(0)
        loss = criterion(outputs, labels).mean()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    is_training = net.training
    # 验证阶段
    net.eval()
    correct = 0
    total = 0
    # num_classes = 10  # 你需要根据实际数据修改，比如 CIFAR-10 是10类

    class_correct = list(0. for _ in range(num_classes))
    class_total = list(0. for _ in range(num_classes))
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
    print(f"Epoch {epoch + 1}, Loss: {running_loss:.3f}, Val Acc: {acc:.2f}%")

    # 打印每类的准确率
    for i in range(num_classes):
        if class_total[i] > 0:
            acc_i = 100 * class_correct[i] / class_total[i]
            print(f"Class {i} Accuracy: {acc_i:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f"Class {i} has no samples in validation set.")
    # 保存最好模型
    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), save_path)
        print(f"✅ Saved best model with acc: {acc:.2f}%")
    net.train(is_training)
print(f'训练完成！best:{best_acc}')
