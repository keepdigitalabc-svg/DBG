import gc
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import torch.nn as nn
import numpy as np
import torch
import torchvision
from torch import optim
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, Subset, ConcatDataset
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from collections import Counter
import random
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import models, transforms
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csgraph, lil_matrix
import numpy as np
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix
import logging

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List, DefaultDict, Tuple
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from hard_adapter import SoftHardAdapter
def visualize_tsne(features_base, features_ext, features_test, labels_base, labels_ext, labels_test, save, layer, random_state=42, num_classes=100):
    """
    使用 t-SNE 可视化 base 和 ext 特征，并按类别上色（不同颜色深浅 + 黑边 + 无图例）

    参数:
        features_base: torch.Tensor [N1, D]
        features_ext:  torch.Tensor [N2, D]
        labels_base:   np.array 或 torch.Tensor [N1]，类别编号 0~num_classes-1
        labels_ext:    np.array 或 torch.Tensor [N2]
        save:          保存路径前缀（不含扩展名）
        random_state:  随机种子（保证可复现）
        num_classes:   类别总数
    """
    # 转 numpy
    base_np = features_base.cpu().numpy()
    ext_np = features_ext.cpu().numpy()
    test_np = features_test.cpu().numpy()
    labels_base = labels_base.cpu().numpy() if hasattr(labels_base, 'cpu') else labels_base
    labels_ext = labels_ext.cpu().numpy() if hasattr(labels_ext, 'cpu') else labels_ext
    labels_test = labels_test.cpu().numpy() if hasattr(labels_test, 'cpu') else labels_test

    # 在拼接后加入 PCA（比如降到 50维）
    all_features = np.concatenate([base_np, ext_np, test_np], axis=0)
    pca = PCA(n_components=200, random_state=random_state)
    all_features_pca = pca.fit_transform(all_features)

    # 然后再做 t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=random_state)
    all_tsne = tsne.fit_transform(all_features)
    base_tsne = all_tsne[:len(base_np)]
    ext_tsne = all_tsne[len(base_np):len(ext_np)+len(base_np)]
    test_tsne = all_tsne[len(ext_np)+len(base_np):]

    # 可视化
    norm = plt.Normalize(vmin=0, vmax=num_classes - 1)
    cmap1 = plt.cm.Blues
    cmap2 = plt.cm.Reds
    cmap3 = plt.cm.Greens

    plt.figure(figsize=(10, 8))

    for cls in range(num_classes):
        mask_base = (labels_base == cls)
        if np.any(mask_base):
            plt.scatter(base_tsne[mask_base, 0], base_tsne[mask_base, 1],
                        color=cmap1(norm(num_classes - 1 - cls)), s=10, alpha=0.5)

        mask_ext = (labels_ext == cls)
        if np.any(mask_ext):
            plt.scatter(ext_tsne[mask_ext, 0], ext_tsne[mask_ext, 1],
                        color=cmap2(norm(num_classes - 1 - cls)), s=10, alpha=0.5, marker='^')

        mask_test = (labels_test == cls)
        if np.any(mask_test):
            plt.scatter(test_tsne[mask_test, 0], test_tsne[mask_test, 1],
                        color=cmap3(norm(num_classes - 1 - cls)), s=10, alpha=0.5, marker='*')

    plt.title("t-SNE Visualization (Base + Extended + test)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.savefig(f"{save}_tsne_layer_{layer}.png", dpi=300)
    plt.close()

def prototype_distance(X, Y):
    proto_X = X.mean(dim=0)
    proto_Y = Y.mean(dim=0)
    return torch.norm(proto_X - proto_Y, p=2).item()

def distance_to_total_prototype_squared_sum(X, Y):
    """
    计算 Y 中所有样本与 X 总原型的距离的平方和
    X: Tensor, shape [N, D]
    Y: Tensor, shape [M, D]
    Return: float, 所有 Y 中样本到 X 总原型的距离的平方和
    """
    proto_X = X.mean(dim=0)  # [D]
    dists_squared = torch.norm(Y - proto_X, dim=1).pow(2)  # [M] 每个样本距离的平方
    return dists_squared.sum().item()  # 返回总和（float）

def distance_to_class_prototypes_squared_sum(X, X_labels, Y, Y_labels):
    """
    计算 Y 中每类样本与其对应 X 类原型的距离平方和
    返回：
        dict[int -> float]，每个类别的总距离平方和
    """
    unique_classes = torch.unique(torch.cat([X_labels, Y_labels]))
    class_prototypes = {}

    # 1. 计算 X 中每个类的原型
    for cls in unique_classes:
        class_mask = (X_labels == cls)
        if class_mask.sum() == 0:
            continue  # 忽略在 X 中没有出现的类
        class_feats = X[class_mask]
        class_prototypes[cls.item()] = class_feats.mean(dim=0)

    # 2. 遍历 Y 中每个样本，按类累加距离平方
    dists_per_class = {cls.item(): 0.0 for cls in unique_classes}
    for y, label in zip(Y, Y_labels):
        cls = label.item()
        if cls not in class_prototypes:
            continue  # 忽略在 X 中没有原型的类
        proto = class_prototypes[cls]
        dist_sq = torch.norm(y - proto, p=2).pow(2).item()
        dists_per_class[cls] += dist_sq/len(unique_classes)**2

    return dists_per_class  # e.g., {0: 14.3, 1: 7.6, ...}
def distance_to_class_prototypes_tensor(X, X_labels, Y, Y_labels, num_classes):
    """
    返回一个长度为 num_classes 的 Tensor，每个位置是该类的距离平方和（未出现类为0）
    """
    dist_dict = distance_to_class_prototypes_squared_sum(X, X_labels, Y, Y_labels)
    result = torch.zeros(num_classes)
    for cls, val in dist_dict.items():
        result[cls] = val/(num_classes)**2
    return result  # shape: [num_classes]
# SWD：Sliced Wasserstein Distance
def sliced_wasserstein_distance(X, Y, num_projections=100):
    X = X.numpy()
    Y = Y.numpy()
    d = X.shape[1]
    distances = []

    min_len = min(X.shape[0], Y.shape[0])  # 👈 截断最短长度

    for _ in range(num_projections):
        proj = np.random.randn(d)
        proj /= np.linalg.norm(proj)
        proj_X = X @ proj
        proj_Y = Y @ proj

        proj_X.sort()
        proj_Y.sort()

        proj_X = proj_X[:min_len]
        proj_Y = proj_Y[:min_len]

        distances.append(np.mean(np.abs(proj_X - proj_Y)))
    return np.mean(distances)
def sliced_wasserstein_distance_per_class(X, Y, labels_X, labels_Y, num_projections=100):
    """
    计算每个类别的 Sliced Wasserstein Distance (SWD)，使用统一采样的方式消除样本数量影响。

    参数:
        X (np.ndarray): 第一个样本集 (n_samples, n_features)
        Y (np.ndarray): 第二个样本集 (n_samples, n_features)
        labels_X (np.ndarray): X 的类别标签 (n_samples,)
        labels_Y (np.ndarray): Y 的类别标签 (n_samples,)
        num_projections (int): 随机投影次数

    返回:
        dict: 每个类别的 SWD 值 {class_id: swd}
    """
    unique_classes = np.unique(np.concatenate([labels_X, labels_Y]))
    class_swds = {}

    for cls in unique_classes:
        X_cls = X[labels_X == cls]
        Y_cls = Y[labels_Y == cls]

        if len(X_cls) == 0 or len(Y_cls) == 0:
            class_swds[cls] = np.nan
            continue

        d = X_cls.shape[1]
        distances = []

        min_len = min(len(X_cls), len(Y_cls))  # 统一样本数

        for _ in range(num_projections):
            # 随机方向
            proj = np.random.randn(d)
            proj /= np.linalg.norm(proj)

            # 随机采样
            idx_X = np.random.choice(len(X_cls), size=min_len, replace=False)
            idx_Y = np.random.choice(len(Y_cls), size=min_len, replace=False)

            sampled_X = X_cls[idx_X]
            sampled_Y = Y_cls[idx_Y]

            # 投影后排序
            proj_X = sampled_X @ proj
            proj_Y = sampled_Y @ proj
            proj_X_sorted = np.sort(proj_X)
            proj_Y_sorted = np.sort(proj_Y)

            # 1D Wasserstein 距离
            distances.append(np.mean(np.abs(proj_X_sorted - proj_Y_sorted)))

        class_swds[cls] = np.mean(distances)

    return class_swds


# 风格差异（特征协方差矩阵的差异）
def style_distance(X, Y):
    def compute_covariance(F):
        F = F - F.mean(dim=0, keepdim=True)
        return torch.matmul(F.T, F) / (F.size(0) - 1)

    # 使用 float32 以节省内存
    X = X.float()
    Y = Y.float()

    # 若维度仍太大，使用 PCA 降维
    if X.shape[1] > 256:
        from sklearn.decomposition import PCA
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        pca = PCA(n_components=256)
        X_np = pca.fit_transform(X_np)
        Y_np = pca.transform(Y_np)
        X = torch.from_numpy(X_np).to(X.device)
        Y = torch.from_numpy(Y_np).to(Y.device)

    cov_X = compute_covariance(X)
    cov_Y = compute_covariance(Y)

    return torch.norm(cov_X - cov_Y, p='fro').item()

def visualize_lpp(feats1, feats2, feats3, labels1, labels2, labels3, save_path, layer='default', n_neighbors=5, out_dim=64, num_classes=100):
    """
    使用 feats1 拟合 PCA + LPP 方向，并将 feats1 和 feats2 同时投影后可视化。
    - feats1, feats2: torch.Tensor, shape [N, C]
    - labels1, labels2: numpy array or torch.Tensor, shape [N]，值为0~num_classes-1
    - save_path: 保存图像的路径（不含扩展名）
    """
    device = feats1.device
    feats1_np = feats1.detach().cpu().numpy()
    feats2_np = feats2.detach().cpu().numpy()
    feats3_np = feats3.detach().cpu().numpy()

    if torch.is_tensor(labels1):
        labels1 = labels1.cpu().numpy()
    if torch.is_tensor(labels2):
        labels2 = labels2.cpu().numpy()
    if torch.is_tensor(labels3):
        labels3 = labels3.cpu().numpy()

    # --- 1. 仅使用 feats1 拟合 PCA ---
    print(f"[LPP] feats1 shape: {feats1_np.shape}, feats2 shape: {feats2_np.shape}, feats3 shape: {feats3_np.shape}")
    pca = PCA(n_components=out_dim)
    feats3_pca = pca.fit_transform(feats3_np)
    feats1_pca = pca.transform(feats1_np)
    feats2_pca = pca.transform(feats2_np)
    print(f"[LPP] After PCA shape: {feats3_pca.shape}")

    # --- 2. 用 feats1 构建 k-NN 图 ---
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(feats3_pca)
    knn_graph = neigh.kneighbors_graph(feats3_pca, mode='connectivity')

    # --- 3. 构建图拉普拉斯矩阵 ---
    W = 0.5 * (knn_graph + knn_graph.T)
    D = np.asarray(np.diag(np.asarray(W.sum(axis=1)).flatten()))
    L = D - W.toarray()

    # --- 4. LPP 特征方向 ---
    X1_tensor = torch.tensor(feats3_pca, dtype=torch.float32)
    L_tensor = torch.tensor(L, dtype=torch.float32)
    A = X1_tensor.T @ L_tensor @ X1_tensor
    eigvals, eigvecs = torch.linalg.eigh(A)

    # --- 5. 投影 ---
    feats1_proj = torch.tensor(feats1_pca, dtype=torch.float32) @ eigvecs[:, :2]
    feats2_proj = torch.tensor(feats2_pca, dtype=torch.float32) @ eigvecs[:, :2]
    feats3_proj = torch.tensor(feats3_pca, dtype=torch.float32) @ eigvecs[:, :2]
    proj1 = feats1_proj.detach().cpu().numpy()
    proj2 = feats2_proj.detach().cpu().numpy()
    proj3 = feats3_proj.detach().cpu().numpy()

    # --- 6. 可视化 ---
    plt.figure(figsize=(10, 8))
    norm = plt.Normalize(vmin=0, vmax=num_classes - 1)

    # 使用颜色映射：Blues 和 Reds，按类别深浅变化
    cmap1 = plt.cm.Blues
    cmap2 = plt.cm.Reds
    cmap3 = plt.cm.Greens

    # base set
    for cls in range(num_classes):
        mask = labels1 == cls
        if np.any(mask):
            plt.scatter(proj1[mask, 0], proj1[mask, 1],
                        color=cmap1(norm(num_classes - 1 - cls)), s=10, alpha=0.5)

    # ext set
    for cls in range(num_classes):
        mask = labels2 == cls
        if np.any(mask):
            plt.scatter(proj2[mask, 0], proj2[mask, 1],
                        color=cmap2(norm(num_classes - 1 - cls)), marker='^', s=10, alpha=0.5)

    for cls in range(num_classes):
        mask = labels3 == cls
        if np.any(mask):
            plt.scatter(proj3[mask, 0], proj3[mask, 1],
                        color=cmap3(norm(num_classes - 1 - cls)), marker='*', s=10, alpha=0.5)

    # plt.legend(fontsize=6, bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol=2)
    plt.title(f'LPP Visualization (Layer: {layer})')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{save_path}_LPP_layer_{layer}.png', dpi=300)
    plt.close()

def visualize_pca(features_base, features_ext, features_test, labels_base, labels_ext, labels_test, save, layer, num_classes=100):
    """
    使用PCA可视化基础特征和扩展特征的分布（按类染色）
    参数:
        features_base (torch.Tensor): 基础数据集特征 [N1, D]
        features_ext (torch.Tensor): 扩展数据集特征 [N2, D]
        labels_base (torch.Tensor or np.array): 基础集标签 [N1]
        labels_ext (torch.Tensor or np.array): 扩展集标签 [N2]
        save (str): 图片保存路径前缀（不含扩展名）
        layer (str): 当前层名（用于保存图像命名）
        num_classes (int): 类别数（用于颜色归一化）
    """
    # 转 numpy
    base_np = features_base.cpu().numpy()
    ext_np = features_ext.cpu().numpy()
    test_np = features_test.cpu().numpy()
    labels_base = labels_base.cpu().numpy() if hasattr(labels_base, 'cpu') else labels_base
    labels_ext = labels_ext.cpu().numpy() if hasattr(labels_ext, 'cpu') else labels_ext
    labels_test = labels_test.cpu().numpy() if hasattr(labels_test, 'cpu') else labels_test

    # PCA 降维
    pca = PCA(n_components=2, random_state=42)
    pca.fit(test_np)
    base_pca = pca.transform(base_np)
    ext_pca = pca.transform(ext_np)
    test_pca = pca.transform(test_np)

    # 准备颜色映射
    norm = plt.Normalize(vmin=0, vmax=num_classes - 1)
    cmap1 = plt.cm.Blues
    cmap2 = plt.cm.Reds
    cmap3 = plt.cm.Greens

    # 绘图
    plt.figure(figsize=(10, 8))

    # base 类点
    for cls in range(num_classes):
        mask = labels_base == cls
        if np.any(mask):
            plt.scatter(base_pca[mask, 0], base_pca[mask, 1],
                        color=cmap1(norm(num_classes - 1 - cls)), alpha=0.5, s=10)

    # ext 类点
    for cls in range(num_classes):
        mask = labels_ext == cls
        if np.any(mask):
            plt.scatter(ext_pca[mask, 0], ext_pca[mask, 1],
                        color=cmap2(norm(num_classes - 1 - cls)), alpha=0.5, s=10, marker='^')

    for cls in range(num_classes):
        mask = labels_test == cls
        if np.any(mask):
            plt.scatter(test_pca[mask, 0], test_pca[mask, 1],
                        color=cmap3(norm(num_classes - 1 - cls)), alpha=0.5, s=10, marker='*')

    # plt.legend(fontsize=6, bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol=2)
    plt.title(f'PCA Visualization (Layer: {layer})')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()

    if save:
        plt.savefig(save + "_pca_" + layer + ".png", dpi=300)
    plt.close()

def evaluate_fid_mmd(base_dataset, extended_dataset, batch_size=32,device='cpu'):
    # min_len = min(len(base_dataset), len(extended_dataset))
    # print(f"📉 对齐样本数量为: {min_len}")

    base_loader = base_dataset
    ext_loader = extended_dataset

    model = models.inception_v3(pretrained=True, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.to(device)

    feats_base = get_features(base_loader, model,device=device)
    feats_ext = get_features(ext_loader, model,device=device)

    mu1, sigma1 = compute_stats(feats_base)
    mu2, sigma2 = compute_stats(feats_ext)

    fid = calculate_fid_torch(mu1, sigma1, mu2, sigma2).item()
    mmd = calculate_mmd_torch(feats_base, feats_ext, gamma=3000).item()
    del model, feats_base, feats_ext, mu1, sigma1, mu2, sigma2
    gc.collect()
    return fid, mmd

def get_features(dataloader, model, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for imgs in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs[0].to(device) if isinstance(imgs, (tuple, list)) else imgs.to(device)
            feats.append(model(imgs))
    return torch.cat(feats, dim=0)

# ---------- FID: 均值 & 协方差 ----------
def compute_stats(feats):
    mu = feats.mean(dim=0)
    X = feats - mu
    sigma = (X.T @ X) / (X.size(0) - 1)
    return mu, sigma

# ---------- FID: matrix square root (GPU-compatible) ----------
def sqrtm_newton_schulz(A, num_iters=50):
    dim = A.size(0)
    normA = A.norm()
    Y = A / normA
    I = torch.eye(dim, device=A.device)
    Z = torch.eye(dim, device=A.device)
    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Y * torch.sqrt(normA)

def calculate_fid_torch(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm_newton_schulz(sigma1 @ sigma2)
    return diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2 * covmean)

# ---------- MMD (RBF Kernel, Torch) ----------
def rbf_kernel_torch(X, Y, gamma=1.0):
    XX = (X ** 2).sum(dim=1).unsqueeze(1)
    YY = (Y ** 2).sum(dim=1).unsqueeze(0)
    dists = XX + YY - 2 * X @ Y.T
    return torch.exp(-gamma * dists)

def calculate_mmd_torch(X, Y, gamma=1.0):
    XX = rbf_kernel_torch(X, X, gamma)
    YY = rbf_kernel_torch(Y, Y, gamma)
    XY = rbf_kernel_torch(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()
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
class FilenameAsLabelDataset(Dataset):
    def __init__(self, root, transform=None, num_classes=10):
        self.root = root
        self.transform = transform
        self.num_classes = num_classes

        self.samples = []
        self.class_counts = [0] * num_classes  # 每类样本计数初始化为 0

        for dirpath, _, filenames in os.walk(root):
            label_str = os.path.basename(dirpath)
            if not label_str.isdigit():
                continue  # 忽略非数字命名的文件夹
            label = int(label_str)
            if label >= num_classes:
                continue  # 超出有效类别范围

            for fname in filenames:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(dirpath, fname)
                    self.samples.append((full_path, label))
                    self.class_counts[label] += 1

        # 排序（可选，保证一致性）
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, label


class FilenameAsLabelDataset1(Dataset):
    def __init__(self, root, transform=None, num_classes=100):  # ← 设为100
        self.root = root
        self.transform = transform
        self.num_classes = num_classes

        self.samples = []
        self.class_counts = [0] * num_classes  # 0..99

        for dirpath, _, filenames in os.walk(root):
            label_str = os.path.basename(dirpath)
            if not label_str.isdigit():
                continue
            label = int(label_str)

            # 只保留 70–99
            if not (70 <= label <= 99):
                continue

            # 仍然保留“超出有效类别范围”的保护
            if label >= num_classes:
                continue

            for fname in filenames:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_path = os.path.join(dirpath, fname)
                    self.samples.append((full_path, label))
                    self.class_counts[label] += 1

        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
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
        self.num_per_cls_dict = dict()
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

def extract_targets(ds):
    if isinstance(ds, ConcatDataset):
        parts = [extract_targets(d) for d in ds.datasets]
        return np.concatenate(parts)
    if isinstance(ds, Subset):
        parent_t = extract_targets(ds.dataset)
        return np.array(parent_t)[ds.indices]
    for attr in ("targets", "labels", "train_labels", "test_labels"):
        if hasattr(ds, attr):
            t = getattr(ds, attr)
            try:
                return np.array(t)
            except Exception:
                try:
                    import torch
                    if hasattr(t, "cpu"):
                        return t.cpu().numpy()
                except Exception:
                    pass
                return np.array(list(t))
    # 兜底：从 __getitem__ 读取第二个返回值作为 label
    return np.array([ds[i][1] for i in range(len(ds))])

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
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
def extract_features(dataloader, model, device='cuda'):
    model.to(device)
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader):
            imgs = imgs.to(device)
            feats = model(imgs)  # shape: [B, D]
            features.append(feats.cpu())
            labels.append(lbls)

    features = torch.cat(features, dim=0)  # [N, D]
    labels = torch.cat(labels, dim=0)      # [N]
    return features, labels

def compute_class_prototypes(features, labels):
    """
    计算每个类别的原型（平均特征）
    返回：dict[class_id] = prototype_tensor
    """
    class_prototypes = {}
    for cls in torch.unique(labels):
        cls_mask = (labels == cls)
        proto = features[cls_mask].mean(dim=0)
        class_prototypes[int(cls.item())] = proto
    return class_prototypes
# class FeatureExtractor(nn.Module):
#     def __init__(self, layer='layer2'):
#         super().__init__()
#         model = models.resnet18(pretrained=True)
#         self.backbone = nn.Sequential(
#             model.conv1, model.bn1, model.relu, model.maxpool,
#             model.layer1 if layer == 'layer1' else nn.Sequential(model.layer1, model.layer2)
#         )
#
#     def forward(self, x):
#         with torch.no_grad():
#             return self.backbone(x).flatten(1)  # Flatten spatially

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, layer_name='avgpool'):
        """
        :param layer_name: 可选层名，例如：
            'conv2d_1a_3x3', 'conv2d_2a_3x3', 'conv2d_2b_3x3', 'maxpool1',
            'conv2d_3b_1x1', 'conv2d_4a_3x3', 'maxpool2',
            'mixed_5b' ~ 'mixed_7c', 'avgpool'
        """
        super().__init__()
        inception = models.inception_v3(pretrained=True, aux_logits=True, transform_input=False)
        inception.eval()
        self.layer_name = layer_name

        self.resize = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        self.blocks = nn.ModuleDict()
        self.layer_output_sizes = {}

        # 分段提取网络结构
        block_list = list(inception.children())
        block_names = [
            'conv2d_1a_3x3', 'conv2d_2a_3x3', 'conv2d_2b_3x3', 'maxpool1',
            'conv2d_3b_1x1', 'conv2d_4a_3x3', 'maxpool2',
            'mixed_5b', 'mixed_5c', 'mixed_5d',
            'mixed_6a', 'mixed_6b', 'mixed_6c', 'mixed_6d', 'mixed_6e',
            'mixed_7a', 'mixed_7b', 'mixed_7c',
            'avgpool'
        ]

        # 手动拼接模块
        self.blocks['conv2d_1a_3x3'] = block_list[0]
        self.blocks['conv2d_2a_3x3'] = block_list[1]
        self.blocks['conv2d_2b_3x3'] = block_list[2]
        self.blocks['maxpool1'] = block_list[3]
        self.blocks['conv2d_3b_1x1'] = block_list[4]
        self.blocks['conv2d_4a_3x3'] = block_list[5]
        self.blocks['maxpool2'] = block_list[6]

        for i in range(7, 15):
            self.blocks[block_names[i]] = block_list[i]
        for i in range(15, 19):
            self.blocks[block_names[i]] = block_list[i+1]

        # self.blocks['avgpool'] = block_list[17]

        # 截取直到目标层的模块序列
        assert layer_name in self.blocks, f"Invalid layer name: {layer_name}"
        idx = block_names.index(layer_name)
        self.active_layers = block_names[:idx + 1]

    def forward(self, x):
        with torch.no_grad():
            x = self.resize(x)  # 调整为 299×299
            for name in self.active_layers:
                x = self.blocks[name](x)
            return torch.flatten(x, 1)  # 展平为 (B, C) 或 (B, C*H*W)
            # return x  # 展平为 (B, C) 或 (B, C*H*W)
def extract_targets_generic(ds, num_classes: int):
    """
    返回 targets: List[int]
    - 对 PseudoLabeledExtendDataset：优先用硬标签；软标签用 argmax
    - 对普通 (img,label) 数据集：直接读取第二项
    """
    targets = []
    if hasattr(ds, "hard_labels") and hasattr(ds, "soft_labels"):
        # 视作 PseudoLabeledExtendDataset
        hard = ds.hard_labels
        soft = ds.soft_labels
        # 硬标签优先；-1 用 soft argmax
        argmax_soft = torch.argmax(soft, dim=1)
        labels = torch.where(hard >= 0, hard, argmax_soft).tolist()
        return labels
    else:
        # 一般数据集：逐个获取 label
        for i in range(len(ds)):
            item = ds[i]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                targets.append(int(item[1]))
            else:
                raise ValueError("数据集条目需为 (img, label) 或伪标签数据集格式")
        return targets

def unforce_balance_after_merge_soft(
    base_dataset,
    extended_dataset,      # 可为 PseudoLabeledExtendDataset 或普通数据集
    num_classes: int,
    pick_from_hard_only: bool = True,
    seed: int = 42,
    verbose: bool = True,  # 打印控制
):
    """
    返回：
      merged_dataset: ConcatDataset(HardLabeledAdapter(base), subset_of_extended)
      selected_indices: List[int]  # 在 extended_dataset 上被选中的索引（相对于 extended_dataset）
      class_counts_after: Dict[int, int]
      per_class_added: Dict[int, Dict[str, int]]  # 新增：每类从 extended 加入的硬/软样本统计
    """
    rng = np.random.RandomState(seed)

    # 1) base 每类计数
    base_targets = extract_targets_generic(base_dataset, num_classes)
    base_class_counts: DefaultDict[int, int] = defaultdict(int)
    for y in base_targets:
        base_class_counts[int(y)] += 1
    max_base_count = max(base_class_counts.values())

    # 2) extended：分别构建硬/软样本池（按类）
    has_pseudo = hasattr(extended_dataset, "hard_labels") and hasattr(extended_dataset, "soft_labels")
    if has_pseudo:
        hard = extended_dataset.hard_labels
        soft = extended_dataset.soft_labels
        argmax_soft = torch.argmax(soft, dim=1)

        # 每类两个池：hard/soft
        extended_hard_pool: DefaultDict[int, List[int]] = defaultdict(list)
        extended_soft_pool: DefaultDict[int, List[int]] = defaultdict(list)

        is_hard_vec = (hard >= 0).to(torch.uint8)

        for idx in range(len(hard)):
            if hard[idx] >= 0:
                cls = int(hard[idx].item())
                extended_hard_pool[cls].append(idx)
            else:
                cls = int(argmax_soft[idx].item())
                extended_soft_pool[cls].append(idx)
    else:
        # 普通数据集：全部视为硬样本
        labels = extract_targets_generic(extended_dataset, num_classes)
        extended_hard_pool: DefaultDict[int, List[int]] = defaultdict(list)
        extended_soft_pool: DefaultDict[int, List[int]] = defaultdict(list)  # 为空
        for idx, y in enumerate(labels):
            extended_hard_pool[int(y)].append(idx)
        is_hard_vec = torch.ones(len(labels), dtype=torch.uint8)

    # 3) 为每类补样本：先硬后软（仅当 pick_from_hard_only=False 才会取软）
    selected_extended_indices: List[int] = []
    per_class_added: Dict[int, Dict[str, int]] = {c: {"hard": 0, "soft": 0} for c in range(num_classes)}

    for cls in range(num_classes):
        base_count = base_class_counts.get(cls, 0)
        need = int(max_base_count - base_count)
        if need <= 0:
            continue

        # 先从硬样本池取
        hard_pool = extended_hard_pool.get(cls, [])
        if hard_pool:
            take_h = min(len(hard_pool), need)
            if take_h > 0:
                pick_h = rng.choice(hard_pool, size=take_h, replace=False).tolist()
                selected_extended_indices.extend(pick_h)
                per_class_added[cls]["hard"] += take_h
                need -= take_h

        # 若仍不足且允许取软，则从软样本池再补
        if (need > 0) and (not pick_from_hard_only):
            soft_pool = extended_soft_pool.get(cls, [])
            if soft_pool:
                take_s = min(len(soft_pool), need)
                if take_s > 0:
                    pick_s = rng.choice(soft_pool, size=take_s, replace=False).tolist()
                    selected_extended_indices.extend(pick_s)
                    per_class_added[cls]["soft"] += take_s
                    need -= take_s

    # 4) 构建“可训练”的 extended 子集
    if hasattr(extended_dataset, "subset"):
        extended_subset = extended_dataset.subset(selected_extended_indices)
    else:
        from torch.utils.data import Subset
        extended_subset = Subset(extended_dataset, selected_extended_indices)

    # 5) 统一输出结构：把 base 也适配为 6元组
    base_adapted = HardLabeledAdapter(base_dataset, num_classes=num_classes)

    # 6) 合并
    merged_dataset = ConcatDataset([base_adapted, extended_subset])

    # 7) 合并后的每类计数
    merged_labels: DefaultDict[int, int] = defaultdict(int)
    for y in base_targets:
        merged_labels[int(y)] += 1

    if hasattr(extended_subset, "hard_labels") and hasattr(extended_subset, "soft_labels"):
        hard2 = extended_subset.hard_labels
        soft2 = extended_subset.soft_labels
        argmax2 = torch.argmax(soft2, dim=1)
        labels2 = torch.where(hard2 >= 0, hard2, argmax2).tolist()
    else:
        labels2 = extract_targets_generic(extended_subset, num_classes)

    for y in labels2:
        merged_labels[int(y)] += 1

    # 8) 打印每类从 extended 加入的硬/软样本数量
    if verbose:
        total_hard = sum(d["hard"] for d in per_class_added.values())
        total_soft = sum(d["soft"] for d in per_class_added.values())
        print("[Merge] 从 extended 加入的样本统计：")
        print(f"  总计：hard={total_hard}, soft={total_soft}")
        for c in range(num_classes):
            h = per_class_added[c]["hard"]
            s = per_class_added[c]["soft"]
            if h > 0 or s > 0:
                print(f"  Class {c:03d}: hard={h}, soft={s}")

    return merged_dataset, selected_extended_indices, dict(merged_labels), per_class_added
def unforce_balance_after_merge(base_dataset, extended_dataset, test_dataset, device, save):
    # 1. 统计 base_dataset 中每类样本数量
    base_targets = extract_targets(base_dataset)
    base_class_counts = defaultdict(int)
    for label in base_targets:
        base_class_counts[label] += 1
    max_base_count = max(base_class_counts.values())
    logging.info(f"[INFO] base_dataset 每类样本数: {dict(base_class_counts)}")

    # 2) 统计 extended 中每类的“可选索引”
    ext_labels = extract_targets(extended_dataset)                  # ✅ 不再依赖 extended_dataset.samples
    extended_class_to_indices = defaultdict(list)
    for idx, y in enumerate(ext_labels):
        extended_class_to_indices[int(y)].append(idx)

    # 3. 从 extended_dataset 中选择样本用于补充 base 中的类别
    selected_extended_indices = []
    for cls, base_count in base_class_counts.items():
        need = int((max_base_count - base_count))  # 需要补充的数量
        if need <= 0:
            continue  # 这个类已经够了

        extended_indices = extended_class_to_indices.get(cls, [])
        if not extended_indices:
            logging.info(f"[WARN] 类 {cls} 在 extended 中没有可用样本")
            continue

        selected_count = min(len(extended_indices), need)
        # if len(extended_indices) < need:
        #     raise ValueError(f"错误：类 {cls} 在 extended_dataset 中没有足够可用样本，无法补充！")
        sampled_indices = np.random.choice(extended_indices, size=selected_count, replace=False)
        logging.info(
            f"[INFO] 补充类别 {cls}：base中有{base_count}个，目标{need}个，从extended中取{selected_count}个样本")
        selected_extended_indices.extend(sampled_indices)

    # 4. 构建新 extended 子集（仅包含被选中的）
    extended_subset = Subset(extended_dataset, selected_extended_indices)
    # 6. 再次统计合并后的每类样本索引
    merged_dataset = ConcatDataset([base_dataset, extended_subset])
    # fid, mmd = evaluate_fid_mmd(base_loader, ext_loader, device=device)
    class_to_indices = defaultdict(list)
    offset = len(base_dataset)

    # base 部分
    for idx, label in enumerate(base_targets):
        class_to_indices[label].append(idx)

    # extended 部分（注意偏移）
    for i, global_idx in enumerate(selected_extended_indices):
        label = extended_dataset.samples[global_idx][1]
        class_to_indices[label].append(i + offset)

    # 7. 找到合并后最多类别的样本数 K
    class_counts = {cls: len(indices) for cls, indices in class_to_indices.items()}
    max_count = max(class_counts.values())
    logging.info(f"[INFO] 合并后各类样本数: {class_counts}")

    logging.info(f"[INFO] 最终总样本数: {len(class_counts)}")

    return merged_dataset, 0, 0

def unforce_balance_after_merge_05(base_dataset, extended_dataset, test_dataset, device, save):
    # 1. 统计 base_dataset 中每类样本数量
    base_targets = extract_targets(base_dataset)
    base_class_counts = defaultdict(int)
    for label in base_targets:
        base_class_counts[label] += 1
    max_base_count = max(base_class_counts.values())
    logging.info(f"[INFO] base_dataset 每类样本数: {dict(base_class_counts)}")

    # 2) 统计 extended 中每类的“可选索引”
    ext_labels = extract_targets(extended_dataset)                  # ✅ 不再依赖 extended_dataset.samples
    extended_class_to_indices = defaultdict(list)
    for idx, y in enumerate(ext_labels):
        extended_class_to_indices[int(y)].append(idx)

    # 3. 从 extended_dataset 中选择样本用于补充 base 中的类别
    selected_extended_indices = []
    for cls, base_count in base_class_counts.items():
        need = int((max_base_count - base_count))//5  # 需要补充的数量
        if need <= 0:
            continue  # 这个类已经够了

        extended_indices = extended_class_to_indices.get(cls, [])
        if not extended_indices:
            logging.info(f"[WARN] 类 {cls} 在 extended 中没有可用样本")
            continue

        selected_count = min(len(extended_indices), need)
        # if len(extended_indices) < need:
        #     raise ValueError(f"错误：类 {cls} 在 extended_dataset 中没有足够可用样本，无法补充！")
        sampled_indices = np.random.choice(extended_indices, size=selected_count, replace=False)
        logging.info(
            f"[INFO] 补充类别 {cls}：base中有{base_count}个，目标{need}个，从extended中取{selected_count}个样本")
        selected_extended_indices.extend(sampled_indices)

    # 4. 构建新 extended 子集（仅包含被选中的）
    extended_subset = Subset(extended_dataset, selected_extended_indices)
    # 6. 再次统计合并后的每类样本索引
    merged_dataset = ConcatDataset([base_dataset, extended_subset])
    # fid, mmd = evaluate_fid_mmd(base_loader, ext_loader, device=device)
    class_to_indices = defaultdict(list)
    offset = len(base_dataset)

    # base 部分
    for idx, label in enumerate(base_targets):
        class_to_indices[label].append(idx)

    # extended 部分（注意偏移）
    for i, global_idx in enumerate(selected_extended_indices):
        label = extended_dataset.samples[global_idx][1]
        class_to_indices[label].append(i + offset)

    # 7. 找到合并后最多类别的样本数 K
    class_counts = {cls: len(indices) for cls, indices in class_to_indices.items()}
    max_count = max(class_counts.values())
    logging.info(f"[INFO] 合并后各类样本数: {class_counts}")

    logging.info(f"[INFO] 最终总样本数: {len(class_counts)}")

    return merged_dataset, 0, 0

def force_balance_after_merge(base_dataset, extended_dataset):
    # 1. 统计 base_dataset 中每类样本数量
    base_targets = np.array(base_dataset.targets)
    base_class_counts = defaultdict(int)
    for label in base_targets:
        base_class_counts[label] += 1
    max_base_count = max(base_class_counts.values())
    print(f"[INFO] base_dataset 每类样本数: {dict(base_class_counts)}")

    # 2. 准备 extended_dataset 中的各类样本索引
    extended_class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(extended_dataset.samples):
        extended_class_to_indices[label].append(idx)

    # 3. 从 extended_dataset 中选择样本用于补充 base 中的类别
    selected_extended_indices = []
    for cls, base_count in base_class_counts.items():
        need = max_base_count - base_count  # 需要补充的数量
        if need <= 0:
            continue  # 这个类已经够了

        extended_indices = extended_class_to_indices.get(cls, [])
        if not extended_indices:
            print(f"[WARN] 类 {cls} 在 extended 中没有可用样本")
            continue

        selected_count = min(len(extended_indices), need)
        sampled_indices = np.random.choice(extended_indices, size=selected_count, replace=False)
        print(
            f"[INFO] 补充类别 {cls}：base中有{base_count}个，目标{max_base_count}个，从extended中取{selected_count}个样本")
        selected_extended_indices.extend(sampled_indices)

    # 4. 构建新 extended 子集（仅包含被选中的）
    extended_subset = Subset(extended_dataset, selected_extended_indices)

    # 5. 合并 base 和补充后的 extended 数据
    merged_dataset = ConcatDataset([base_dataset, extended_subset])

    # 6. 再次统计合并后的每类样本索引
    class_to_indices = defaultdict(list)
    offset = len(base_dataset)

    # base 部分
    for idx, label in enumerate(base_dataset.targets):
        class_to_indices[label].append(idx)

    # extended 部分（注意偏移）
    for i, global_idx in enumerate(selected_extended_indices):
        label = extended_dataset.samples[global_idx][1]
        class_to_indices[label].append(i + offset)

    # 7. 找到合并后最多类别的样本数 K
    class_counts = {cls: len(indices) for cls, indices in class_to_indices.items()}
    max_count = max(class_counts.values())
    print(f"[INFO] 合并后各类样本数: {class_counts}")
    print(f"[INFO] 将进行上采样使所有类别达到 {max_count} 个样本")

    # 8. 对所有类别做上采样（或保留）
    final_indices = []
    for cls, indices in class_to_indices.items():
        if len(indices) < max_count:
            sampled = np.random.choice(indices, size=max_count, replace=True)  # 上采样
        else:
            sampled = np.random.choice(indices, size=max_count, replace=False)
        final_indices.extend(sampled)

    # 9. 构建最终平衡数据集
    balanced_dataset = Subset(merged_dataset, final_indices)

    # 统计 balanced_dataset 中的标签
    final_labels = []

    for i in range(len(balanced_dataset)):
        idx = balanced_dataset.indices[i]  # 获取 merged_dataset 中的索引
        if idx < len(base_dataset):
            label = base_dataset.targets[idx]
        else:
            real_idx = selected_extended_indices[idx - len(base_dataset)]
            label = extended_dataset.samples[real_idx][1]
        final_labels.append(label)

    # 统计标签频率
    label_counts = Counter(final_labels)
    classes = sorted(label_counts.keys())
    counts = [label_counts[c] for c in classes]

    # 可视化
    plt.figure(figsize=(10, 4))
    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Sample Count")
    plt.title("Final Balanced Dataset (after Oversampling)")
    plt.xticks(classes)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("final_balanced_distribution_from_balanced_dataset.png")
    plt.close()

    print(f"[INFO] 最终平衡后每类样本数: {dict(label_counts)}")
    print(f"[INFO] 平衡后数据集总样本数: {len(balanced_dataset)}")

    return balanced_dataset

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                                 download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb;

    pdb.set_trace()