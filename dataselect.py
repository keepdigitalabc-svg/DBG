# proto_pseudo_label.py
import os
from typing import Tuple, List, Optional, Dict, Any
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as T

# ----------------------
# 公共工具
# ----------------------
def _pair_list_from_dataset(ds):
    """返回 [(path, label), ...]（兼容 ImageFolder / 自定义 Dataset）"""
    if hasattr(ds, 'samples'):
        return ds.samples
    if hasattr(ds, 'imgs'):
        return ds.imgs
    if hasattr(ds, 'filepaths') and hasattr(ds, 'labels'):
        return list(zip(ds.filepaths, ds.labels))
    raise AttributeError("Dataset需暴露 .samples / .imgs / (filepaths, labels) 之一。")

@torch.no_grad()
def _select_feat_from_output(out, feat_idx: int) -> torch.Tensor:
    """
    期望 forward 返回 (logits, [feat1,...]) 或 直接返回 [feat1,...]
    支持4D特征自动 GAP 到向量。
    """
    if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], (list, tuple)):
        feats_list = out[1]
    elif isinstance(out, (list, tuple)) and all(hasattr(x, 'shape') for x in out):
        feats_list = out
    else:
        raise ValueError("模型输出需为 (logits, [feat1,...]) 或 [feat1,...]。")

    if feat_idx < 0:
        feat_idx = len(feats_list) + feat_idx
    assert 0 <= feat_idx < len(feats_list), f"feat_idx {feat_idx} 超出范围(共有 {len(feats_list)} 个特征)"

    feat = feats_list[feat_idx]
    if feat.dim() == 4:
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
    return feat

def pairwise_distance_matrix(X: torch.Tensor, Y: torch.Tensor, metric: str, already_normed=False) -> torch.Tensor:
    """
    X: [N,D], Y: [C,D] -> 返回 [N,C] 的距离/相似度映射
    metric='cosine'：返回 1 - cos，相当于“距离”
    metric='euclidean'：L2
    """
    if metric == 'cosine':
        if not already_normed:
            X = F.normalize(X, dim=1)
            Y = F.normalize(Y, dim=1)
        cos = X @ Y.t()  # [N,C]
        return (1.0 - cos).clamp(0.0, 2.0)
    elif metric == 'euclidean':
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2x·y
        XX = (X*X).sum(dim=1, keepdim=True)   # [N,1]
        YY = (Y*Y).sum(dim=1, keepdim=True).t()  # [1,C]
        dist2 = XX + YY - 2 * (X @ Y.t())
        return dist2.clamp_min(0.0).sqrt()
    else:
        raise ValueError(f"Unknown metric: {metric}")

def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float('nan')
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return float('nan')
    return float(torch.quantile(x, q))

# ----------------------
# 核心：构建伪标签 extend 数据集
# ----------------------

@torch.no_grad()
def build_pseudo_labeled_extend_dataset(
    base_dataset: Dataset,
    extend_dataset: Dataset,
    model: torch.nn.Module,
    num_classes: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size: int = 256,
    num_workers: int = 4,
    feat_idx: int = -1,
    feat_norm: bool = True,
    metric: str = "cosine",              # 'cosine' 或 'euclidean'
    q_low: float = 5.0,                  # 每类下分位（%）
    q_high: float = 95.0,                # 每类上分位（%）
    min_samples_thr: int = 10,           # 类内样本少时回退到全局分位
    iqr_k: float = 1.5,                  # IQR 裁剪参数
    eps_margin: float = 1e-6,            # 避免 low==high
    widen_if_tight: float = 1e-3,        # 区间太窄时加宽
    temperature: float = 0.1,            # 软标签 softmax 温度（越小越“硬”）
    eval_transform: Optional[T.Compose] = None,
    progress_desc_prefix: str = "ProtoPseudo"
) -> "PseudoLabeledExtendDataset":
    """
    返回：PseudoLabeledExtendDataset
      - 对通过区间筛选的样本：硬标签（保持原伪标签），soft label = one-hot
      - 对未通过的样本：软标签 = softmax(-dist/τ)，保留参与训练
    """
    assert 0.0 <= q_low < q_high <= 100.0

    # 缺省 eval transform（不做强增强）
    if eval_transform is None:
        eval_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
        ])

    model = model.to(device)
    model.eval()

    # ---- 备份原 transform，并临时换成 eval_transform 提特征
    _base_tf_bak = getattr(base_dataset, "transform", None)
    _ext_tf_bak  = getattr(extend_dataset, "transform", None)
    if hasattr(base_dataset, "transform"):
        base_dataset.transform = eval_transform
    if hasattr(extend_dataset, "transform"):
        extend_dataset.transform = eval_transform

    # ---- 提取 base 特征
    base_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    base_feats, base_labels = [], []
    for images, targets in base_loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        feats = _select_feat_from_output(out, feat_idx)
        if feat_norm:
            feats = F.normalize(feats, dim=1)
        base_feats.append(feats.cpu())
        base_labels.append(targets.cpu())
    base_feats  = torch.cat(base_feats, dim=0)
    base_labels = torch.cat(base_labels, dim=0)

    # ---- 构建类原型
    D = base_feats.shape[1]
    prototypes = []
    for c in range(num_classes):
        idx = (base_labels == c)
        if idx.any():
            proto = base_feats[idx].mean(dim=0)
            if feat_norm:
                proto = F.normalize(proto, dim=0)
        else:
            proto = torch.zeros(D)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes, dim=0)  # [C,D]

    # ---- 计算每类区间阈值（IQR + 分位 + 回退）
    low_thr = torch.zeros(num_classes, dtype=torch.float32)
    high_thr = torch.zeros(num_classes, dtype=torch.float32)

    per_class_d = []
    for c in range(num_classes):
        idx = (base_labels == c)
        if idx.any():
            d = pairwise_distance_matrix(base_feats[idx], prototypes[c].unsqueeze(0), metric, already_normed=feat_norm).squeeze(1)
            if metric == 'cosine':
                d = d.clamp_min(0.0).clamp_max(2.0)
            d = d[torch.isfinite(d)]
            per_class_d.append((c, d))
        else:
            per_class_d.append((c, torch.empty(0)))

    global_pool = torch.cat([d for _, d in per_class_d if d.numel() > 0], dim=0)
    if global_pool.numel() == 0:
        low_thr[:] = -1e9
        high_thr[:] = 1e9
    else:
        g_low = _safe_quantile(global_pool, q_low/100.0)
        g_high = _safe_quantile(global_pool, q_high/100.0)
        if not torch.isfinite(torch.tensor(g_low)):  g_low = 0.0
        if not torch.isfinite(torch.tensor(g_high)): g_high = float(global_pool.median())

        for c, d in per_class_d:
            if d.numel() < min_samples_thr:
                low_c, high_c = g_low, g_high
            else:
                q1 = _safe_quantile(d, 0.25)
                q3 = _safe_quantile(d, 0.75)
                if torch.isfinite(torch.tensor(q1)) and torch.isfinite(torch.tensor(q3)):
                    iqr = max(q3 - q1, 0.0)
                    lo_clip = q1 - iqr_k * iqr
                    hi_clip = q3 + iqr_k * iqr
                    d_clip = d[(d >= lo_clip) & (d <= hi_clip)]
                    d_use = d_clip if d_clip.numel() >= max(5, int(0.5 * d.numel())) else d
                else:
                    d_use = d

                low_c = _safe_quantile(d_use, q_low/100.0)
                high_c = _safe_quantile(d_use, q_high/100.0)
                if not torch.isfinite(torch.tensor(low_c)):  low_c = g_low
                if not torch.isfinite(torch.tensor(high_c)): high_c = g_high

            if high_c <= low_c + eps_margin:
                center = 0.5 * (low_c + high_c)
                span = max(widen_if_tight, eps_margin)
                low_c, high_c = center - span, center + span

            low_thr[c] = low_c
            high_thr[c] = high_c

    # ---- 提取 extend 特征
    ext_loader = DataLoader(extend_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    ext_feats, ext_labels = [], []
    for images, targets in ext_loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        feats = _select_feat_from_output(out, feat_idx)
        if feat_norm:
            feats = F.normalize(feats, dim=1)
        ext_feats.append(feats.cpu())
        ext_labels.append(targets.cpu())
    ext_feats  = torch.cat(ext_feats, dim=0)        # [N,D]
    ext_labels = torch.cat(ext_labels, dim=0).long()# [N]

    # ---- 恢复原 transform
    if hasattr(base_dataset, "transform"):
        base_dataset.transform = _base_tf_bak
    if hasattr(extend_dataset, "transform"):
        extend_dataset.transform = _ext_tf_bak

    # ---- 软标签分配需要到所有类原型的距离
    # 先计算 [N,C] 距离矩阵
    dis_nc = pairwise_distance_matrix(ext_feats, prototypes, metric, already_normed=feat_norm)  # [N,C]
    # 负距离做logit，按温度 softmax 得到 soft label
    logits = -dis_nc / max(1e-8, float(temperature))
    soft_all = torch.softmax(logits, dim=1)  # [N,C]

    # ---- 按所属标签的区间做“硬/软”判定
    N = ext_feats.size(0)
    is_hard = torch.zeros(N, dtype=torch.uint8)  # 1=硬标签，0=软标签
    hard_labels = torch.full((N,), -1, dtype=torch.long)
    soft_labels = soft_all.clone()

    kept_indices: List[int] = []
    removed_indices: List[int] = []
    stats = defaultdict(lambda: {'keep':0, 'rm_small':0, 'rm_large':0, 'total':0})

    valid_mask = (ext_labels >= 0) & (ext_labels < num_classes)
    for c in range(num_classes):
        cls_mask = valid_mask & (ext_labels == c)
        if not torch.any(cls_mask):
            continue
        idx = torch.nonzero(cls_mask).squeeze(1)  # 该类所有索引

        # 与自身类原型的距离
        d_c = dis_nc[idx, c]  # [Nc]
        low_c, high_c = float(low_thr[c]), float(high_thr[c])

        keep_mask = (d_c >= low_c) & (d_c <= high_c)
        small_m   = (d_c < low_c)
        large_m   = (d_c > high_c)

        # 通过者：硬标签 = 原伪标签；soft置为one-hot
        k_idx = idx[keep_mask]
        kept_indices.extend(k_idx.tolist())
        is_hard[k_idx] = 1
        hard_labels[k_idx] = c
        soft_labels[k_idx] = torch.nn.functional.one_hot(torch.full((k_idx.numel(),), c, dtype=torch.long),
                                                         num_classes=num_classes).float()

        # 未通过者：保留，硬标签=-1，soft_labels使用 soft_all（已算好）
        r_idx = idx[small_m | large_m]
        removed_indices.extend(r_idx.tolist())
        # hard_labels[r_idx] 维持 -1，soft_labels[r_idx] 已是 soft_all

        # 统计
        Nc = idx.numel()
        stats[c]['total']    += int(Nc)
        stats[c]['keep']     += int(keep_mask.sum().item())
        stats[c]['rm_small'] += int(small_m.sum().item())
        stats[c]['rm_large'] += int(large_m.sum().item())

    # 对非法标签样本：直接使用soft_all，硬标签=-1
    invalid_idx = torch.nonzero(~valid_mask).squeeze(1)
    if invalid_idx.numel() > 0:
        removed_indices.extend(invalid_idx.tolist())

    # 置信度权重
    weights = soft_labels.max(dim=1).values  # [N]，可直接用于加权训练

    # 返回一个新的用于训练的 Dataset
    pairs = _pair_list_from_dataset(extend_dataset)
    return PseudoLabeledExtendDataset(
        pairs=pairs,
        transform=getattr(extend_dataset, "transform", None),
        num_classes=num_classes,
        hard_labels=hard_labels,
        soft_labels=soft_labels,
        is_hard=is_hard,
        weights=weights,
        meta=dict(
            thresholds=dict(low=low_thr, high=high_thr),
            kept_indices=kept_indices,
            removed_indices=removed_indices,
            per_class_stats=stats,
            metric=metric,
            temperature=temperature
        )
    )

# ----------------------
# 数据集封装：训练时直接用
# ----------------------
class PseudoLabeledExtendDataset(Dataset):
    """
    返回：
      img: Tensor
      hard_label: int（-1 表示使用软标签）
      soft_label: FloatTensor [C]
      is_hard: int(0/1)
      weight: float（=max(soft_label)）
      path: str
    """
    def __init__(
        self,
        pairs: List[Tuple[str, int]],
        transform: Optional[Any],
        num_classes: int,
        hard_labels: torch.Tensor,
        soft_labels: torch.Tensor,
        is_hard: torch.Tensor,
        weights: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.pairs = pairs
        self.transform = transform
        self.num_classes = num_classes

        self.hard_labels = hard_labels
        self.soft_labels = soft_labels
        self.is_hard = is_hard
        self.weights = weights
        self.meta = meta or {}

        # 简单的图片加载（假设 pairs 存的是路径）
        self._pil_loader = _default_pil_loader

    def __len__(self):
        return len(self.pairs)

    def subset(self, indices: List[int]) -> "PseudoLabeledExtendDataset":
        import torch
        # 选择子集时，同时裁剪所有字段，保证信息不丢
        new_pairs = [self.pairs[i] for i in indices]
        new_hard = self.hard_labels[indices]
        new_soft = self.soft_labels[indices]
        new_is_hard = self.is_hard[indices]
        new_weight = self.weights[indices]
        new_meta = dict(self.meta)
        new_meta["subset_from"] = indices
        return PseudoLabeledExtendDataset(
            pairs=new_pairs,
            transform=self.transform,
            num_classes=self.num_classes,
            hard_labels=new_hard,
            soft_labels=new_soft,
            is_hard=new_is_hard,
            weights=new_weight,
            meta=new_meta
        )
    def __getitem__(self, idx):
        path, _ = self.pairs[idx]
        img = self._pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return (
            img,
            int(self.hard_labels[idx].item()),
            self.soft_labels[idx],
            int(self.is_hard[idx].item()),
            float(self.weights[idx].item()),
            path
        )

def _default_pil_loader(path: str):
    from PIL import Image
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# ----------------------
# 导出工具：将筛选后的样本反 transform 并落盘
# ----------------------
from PIL import Image
import os
import copy
import torchvision.transforms.functional as TF

def _build_denorm(mean, std):
    """返回一个将张量从 Normalize(mean,std) 反归一化的函数。"""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    def _denorm(x: torch.Tensor) -> torch.Tensor:
        # x: [C,H,W] in normalized space
        return x * std + mean
    return _denorm

@torch.no_grad()
def save_filtered_images(
    ds: "PseudoLabeledExtendDataset",
    out_dir_hard: str = "extended_hard1",
    out_dir_soft: str = "extened_soft1",   # 注意：按你的拼写要求
    transform_to_apply: Optional[T.Compose] = None,
    # 若 transform_to_apply 内含 Normalize，需要指定其 mean/std 以便反归一化
    denorm_mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465),
    denorm_std: Tuple[float, float, float]  = (0.2023, 0.1994, 0.2010),
    overwrite: bool = False,
    use_soft_topk: int = 1,  # 目前用于决定软样本标签，=1 表示 argmax；可扩展写入topk到文件名
    verbose: bool = True,
):
    """
    根据 PseudoLabeledExtendDataset 的筛选结果，将图片保存到：
      extended_hard/<label>/filename
      extened_soft/<label>/filename

    参数：
      - ds: 由 build_pseudo_labeled_extend_dataset 返回的 Dataset
      - transform_to_apply: 若提供，则会先对原图应用该 transform，再反 Normalize 后保存；
                            若不提供，则直接保存原始图（PIL）；
      - denorm_mean/std:    与 transform_to_apply 中 Normalize 对应；
      - overwrite:          同名是否覆盖；为 False 时会自动加后缀避免覆盖；
    """
    os.makedirs(out_dir_hard, exist_ok=True)
    os.makedirs(out_dir_soft, exist_ok=True)

    # 反归一化与张量->PIL
    denorm = _build_denorm(denorm_mean, denorm_std)
    to_pil = T.ToPILImage()

    num = len(ds)
    hard_cnt = soft_cnt = 0

    for i in range(num):
        path, _ = ds.pairs[i]
        is_h = int(ds.is_hard[i].item()) == 1

        if is_h:
            label = int(ds.hard_labels[i].item())
            root = out_dir_hard
            hard_cnt += 1
        else:
            # 软样本：用 soft argmax 作为保存子目录
            label = int(torch.argmax(ds.soft_labels[i]).item())
            root = out_dir_soft
            soft_cnt += 1

        # 目标子目录
        save_dir = os.path.join(root, str(label))
        os.makedirs(save_dir, exist_ok=True)

        # 文件名与后缀
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        if ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
            ext = ".png"  # 不可识别的就保存成 png

        # 读取原始 PIL
        img_pil: Image.Image = ds._pil_loader(path)

        # 若提供 transform_to_apply：先应用 transform（Tensor 流程），再反 Normalize 保存
        if transform_to_apply is not None:
            # 尽量复制一份 transform，避免外部被修改
            tfm = transform_to_apply

            # 将 PIL 送入 transform；若 transform 以 ToTensor + Normalize 结尾，这里能正确反归一化
            out_img = tfm(img_pil)

            # 如果结果是张量（常见），尝试反 Normalize
            if torch.is_tensor(out_img):
                x = out_img
                # 检查是否需要反归一化：若范围明显在[-3,3]附近，基本可视为已 Normalize
                # 直接按提供的 mean/std 做反归一化更稳妥
                try:
                    x = denorm(x)
                except Exception:
                    pass
                # 裁剪到 [0,1] 以避免浮点误差
                x = x.clamp(0, 1)
                img_save = to_pil(x)
            else:
                # 某些 transform 可能仍返回 PIL
                img_save = out_img
        else:
            # 不做 transform，直接保存原图
            img_save = img_pil

        # 处理重名
        save_path = os.path.join(save_dir, name + ext)
        if (not overwrite) and os.path.exists(save_path):
            k = 1
            while True:
                alt = os.path.join(save_dir, f"{name}_{k}{ext}")
                if not os.path.exists(alt):
                    save_path = alt
                    break
                k += 1

        # 保存
        try:
            img_save.save(save_path)
        except Exception as e:
            if verbose:
                print(f"[WARN] 保存失败: {save_path} ({e})")

    if verbose:
        print(f"[DONE] 硬样本保存: {hard_cnt} 到 {out_dir_hard}/<label>/")
        print(f"[DONE] 软样本保存: {soft_cnt} 到 {out_dir_soft}/<label>/")
