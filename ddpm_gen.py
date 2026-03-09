
import copy
import json
import os
import warnings
from classify.resnet import resnet32, resnet50
import torchvision
from absl import app, flags
import torchvision.transforms as T
import random
import shutil
#import cv2
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import trange
from diffusion import *
from model.model import UNet
from score.both import get_inception_and_fid_score
from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10, get_dataset224_longtail
from score.fid import get_fid_score
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from utils.augmentation import KarrasAugmentationPipeline
import sys

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
# parser = argparse.ArgumentParser(description="Training with imbalanced data")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# parser.add_argument('--dataset', type=str, default='cifar-100')
# parser.add_argument('--net', type=str, default='resnet32')
# parser.add_argument('--imb', type=float, default=0.01)
FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', True, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 8, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs_cifar-100_0.01', help='log directory')
flags.DEFINE_string('cond_logdir', './logs/DDPM_NIH', help='cond log directory')
flags.DEFINE_string('uncond_logdir', './logs/DDPM_NIH', help='uncond log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
flags.DEFINE_float('w', 1.5, help='Guided rate')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 500000, help='the number of generated images for evaluation')
flags.DEFINE_integer('num_images_per_class', 10000, help='the number of generated images for evaluation per class')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='fid cache')
flags.DEFINE_integer('ckpt_step',100000,help="chekpoint step")
flags.DEFINE_integer('ckpt_step_uncond',-1,help="chekpoint step")
flags.DEFINE_integer('specific_class',-1,help="generate specific class -1 for not utilization")
flags.DEFINE_bool('balanced_dat',False,help="using bal dataset")
flags.DEFINE_integer('ddim_skip_step',10,help="ddim step")
flags.DEFINE_integer('cut_time',1001,help="cut time")
flags.DEFINE_integer('num_class', 100, help='number of class of the pretrained model')
flags.DEFINE_string('sample_method', 'ddim_target', help='sampling method')
flags.DEFINE_bool('conditional', True, help='conditional generation')
flags.DEFINE_bool('weight', False, help='reweight')
flags.DEFINE_bool('cotrain', False, help='cotrain with an adjusted classifier or not')
flags.DEFINE_bool('logit', False, help='use logit adjustment or not')
flags.DEFINE_bool('augm', False, help='whether to use ADA augmentation')
flags.DEFINE_bool('cfg', False, help='whether to train unconditional generation with with 10\%  probability')
flags.DEFINE_string('dataset', 'cifar-100', help='')
flags.DEFINE_string('imbalance', '0.01', help='')
FLAGS(sys.argv)
num_class = FLAGS.num_class
dataset = FLAGS.dataset
imbalance = FLAGS.imbalance
if dataset == 'cifar-100':
    mu, sigma = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
else:  # cifar-10
    mu, sigma = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mu, sigma),
    # transforms.Resize([32, 32])
])
transform_test = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mu, sigma)
])
device = torch.device('cuda')
# 加载 CIFAR 数据
net = resnet32(num_classes=num_class)
net.load_state_dict(torch.load(f'./classify/best_resnet32_{dataset}_imbalanced{imbalance}.pth', map_location='cpu'))
net = net.to(device)
net.eval()

@torch.no_grad()
def compute_class_stats(phi0, train_loader, device):
    phi0.eval()
    feats_by_c = {}
    for x, y in train_loader:
        x = x.to(device); y = y.to(device)
        _,f = phi0(x)  # [B, D]
        f=f[3].detach()
        for fi, yi in zip(f, y):
            feats_by_c.setdefault(int(yi), []).append(fi.cpu())
    centers, df = {}, {}
    for c, vecs in feats_by_c.items():
        F = torch.stack(vecs)               # [Nc, D]
        mu = F.mean(0)
        centers[c] = mu
        # 最大两点距离的近似：与中心的最远距离×2 的保守估计
        # 或者精确一点：F.cdist(F).max()（大类会慢）
        r = (F - mu).norm(dim=1).max()
        df[c] = r.item()
    return {k: v for k, v in centers.items()}, {k: v for k, v in df.items()}

@torch.no_grad()
def classify_id_aid_ood(phi0, imgs, labels, centers, df_map, device):
    phi0.eval()
    imgs = imgs.to(device)
    labels = labels.to(device)
    _,f = phi0(imgs)
    f = f[3].detach().cpu()
    types = []  # 'ID' / 'AID' / 'OOD'
    for i in range(f.shape[0]):
        c = int(labels[i])
        mu = centers[c]
        df = df_map[c]
        d = torch.norm(f[i] - mu).item()
        if d <= df: types.append('ID')
        elif d <= 2*df: types.append('AID')
        else: types.append('OOD')
    return types

def _counts_to_map(train_counts, num_classes=None):
    """
    支持 dict / list / tuple / np.ndarray / torch.Tensor
    统一转成 {class_id: count} 的字典
    """
    if isinstance(train_counts, dict):
        return {int(k): int(v) for k, v in train_counts.items()}

    # torch.Tensor → list
    if isinstance(train_counts, torch.Tensor):
        train_counts = train_counts.detach().cpu().tolist()

    # numpy → list
    if isinstance(train_counts, np.ndarray):
        train_counts = train_counts.tolist()

    if isinstance(train_counts, (list, tuple)):
        if num_classes is not None and len(train_counts) != num_classes:
            # 不是致命问题，但提示你检查一下配置
            print(f"[warn] len(train_counts)={len(train_counts)} != num_classes={num_classes}")
        return {i: int(n) for i, n in enumerate(train_counts)}

    raise TypeError(f"Unsupported type for train_counts: {type(train_counts)}")


def plan_counts(train_counts, N_t: int, num_classes: int = None):
    """
    输入可为 list/dict/ndarray/tensor，返回 {class_id: need_num}
    need_num = max(0, N_t - real_count)
    """
    counts_map = _counts_to_map(train_counts, num_classes=num_classes)
    return {c: max(0, int(N_t) - int(n)) for c, n in counts_map.items()}

@torch.no_grad()
def generate_balanced(
    sampler,               # 你的 GaussianDiffusionSamplerOld 实例
    num_classes:int,
    per_class_need:dict,   # 来自 plan_counts
    batch_size:int,
    img_size:int,
    device,
    phi0=None, centers=None, df_map=None,  # 过滤所需，可为 None 跳过
    keep_types=('ID','AID'),
    method='ddim', skip=1, eta=0.0,
):
    images_out, labels_out = [], []
    for c in tqdm(range(num_classes)):
        need = per_class_need.get(c, 0)
        while need > 0:
            b = min(batch_size, need)
            x_T = torch.randn((b, 3, img_size, img_size), device=device)
            y   = torch.full((b,), c, dtype=torch.long, device=device)
            x0  = sampler(
                x_T.to(device),
                y,
                method="ddim",
                skip=FLAGS.ddim_skip_step,
                w_neg=0.5
            ).cpu()

            x0  = ((x0 + 1) / 2).clamp(0, 1).cpu()
            y_cpu = y.cpu()

            if phi0 is not None:
                x0, y_cpu, _ = filter_keep_types(x0, y_cpu, keep=keep_types,
                                                 phi0=phi0, centers=centers, df_map=df_map, device=device)
            images_out.append(x0); labels_out.append(y_cpu)
            need -= x0.shape[0]
    return torch.cat(images_out,0), torch.cat(labels_out,0)

def filter_keep_types(imgs, labels, keep=('ID','AID'), **kw):
    # kw: centers, df_map, phi0, device
    types = classify_id_aid_ood(kw['phi0'], imgs, labels, kw['centers'], kw['df_map'], kw['device'])
    mask = torch.tensor([t in keep for t in types])
    return imgs[mask], labels[mask], types

if dataset == 'cifar-100':
    train_dataset = IMBALANCECIFAR100(
        root='./data',  # 数据集存储路径
        train=True,  # 使用训练集
        download=True,  # 自动下载数据集
        transform=transform_train,  # 数据预处理
        imb_type='exp',  # 不平衡类型: 'exp'(指数)/'step'(阶梯)
        imb_factor=float(FLAGS.imbalance),  # 不平衡因子(最小类样本比例)
        rand_number=42  # 随机种子(确保可复现性)
    )
    trainloader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4)

    # trainset = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=True, download=True, transform=transform_train)
    # trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=24, shuffle=False, num_workers=4)
elif dataset == 'cifar-10':
    train_dataset = IMBALANCECIFAR10(
        root='./data',              # 数据集存储路径
        train=True,                  # 使用训练集
        download=True,               # 自动下载数据集
        transform=transform_train,         # 数据预处理
        imb_type='exp',              # 不平衡类型: 'exp'(指数)/'step'(阶梯)
        imb_factor=float(FLAGS.imbalance),             # 不平衡因子(最小类样本比例)
        rand_number=42               # 随机种子(确保可复现性)
    )
    trainloader = DataLoader(train_dataset, batch_size=24, shuffle=False, num_workers=4)

    # trainset = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=True, download=True, transform=transform_train)
    # trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=24, shuffle=False, num_workers=4)
else:
    train_dataset, testset, classnames, per_cls = get_dataset224_longtail(
        data_root="./tiny-imagenet-200",
        # label_json_path="/data/lhz/data/imagenet100/ImageNet_class_index.json",  # 你的 JSON 路径
        imb_type="exp",  # 'exp' | 'step' | 'none'
        imb_factor=float(FLAGS.imbalance),  # 尾部/头部比例
        rand_number=0
    )
    trainloader = DataLoader(train_dataset, batch_size=24, shuffle=False, num_workers=6)

    # trainset = torchvision.datasets.CIFAR10(root='../data/cifar-10', train=True, download=True, transform=transform_train)
    # trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=24, shuffle=False, num_workers=6)

    #
    # def _find_normalize(tfms):
    #     if tfms is None:
    #         return None, None
    #     if isinstance(tfms, T.Normalize):
    #         return tfms.mean, tfms.std
    #     if isinstance(tfms, T.Compose):
    #         for t in tfms.transforms:
    #             if isinstance(t, T.Normalize):
    #                 return t.mean, t.std
    #     return None, None
    #
    #
    # def _denorm_if_needed(x, mean, std):
    #     """x: (B,C,H,W) tensor in torch.float"""
    #     if mean is not None and std is not None:
    #         mean = torch.as_tensor(mean, device=x.device).view(1, -1, 1, 1)
    #         std = torch.as_tensor(std, device=x.device).view(1, -1, 1, 1)
    #         x = x * std + mean
    #     elif x.min() < 0:  # 可能在 [-1,1]
    #         x = (x + 1) / 2
    #     return x.clamp(0, 1)
    #
    #
    # # 取一个 batch
    # imgs, labels = next(iter(testloader))  # imgs: (B,C,H,W), labels: (B,)
    #
    # # 反归一化到 [0,1]
    # mean, std = _find_normalize(getattr(testset, 'transform', None))
    # vis = _denorm_if_needed(imgs[:1].clone(), mean, std)  # 只看第1张，形状保持 (1,C,H,W)
    #
    # # —— 方式2：存成文件（不依赖图形界面）—— #
    # save_image(vis, "sample_debug.png")  # 会保存为 ./sample_debug.png
    # print("Saved to sample_debug.png")
def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y

N_CLASS=10


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x,y

def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

def evaluate(sampler, model, save_dir="gen_images", save=True, use_eval=True, save_intermediate=False):
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        desc = "generating images"
        if FLAGS.sample_method == 'ddim':
            for i in trange(100, desc=desc):
                classes = torch.arange(100, device=device)
                classes = classes[classes != i]  # [99]
                x_T = torch.randn((10, 3, FLAGS.img_size, FLAGS.img_size))
                # 3) 生成 990 个标签：每个类重复 n 次
                y = classes.repeat_interleave(10).to(torch.long)  # [99*n] = [990]

                # 4) 把 x 按类数复制并拼接成 990 个样本（两种方式：repeat 或 expand）
                # （需要独立内存就用 repeat；想省内存且只读可用 expand+reshape）
                x_T = x_T.unsqueeze(0).repeat(len(classes), 1, 1, 1, 1).reshape(-1, 3, FLAGS.img_size, FLAGS.img_size)
                y_cond = torch.full((x_T.shape[0],), i, device=device, dtype=torch.long)
                # 采样生成图像
                batch_images = sampler(
                    x_T.to(device),
                    y_cond,
                    y_uncond=y,
                    method=FLAGS.sample_method,
                    skip=FLAGS.ddim_skip_step,
                    w_neg=0.5
                ).cpu()
                batch_images = ((batch_images + 1) / 2).clamp(0, 1)
                for idx, (img_tensor, neg_label) in enumerate(zip(batch_images, y.cpu().numpy())):
                    # 转换为PIL
                    img = transforms.ToPILImage()(img_tensor)

                    # 保存到标签对应的子文件夹
                    label_dir = os.path.join(save_dir, str(i))
                    os.makedirs(label_dir, exist_ok=True)

                    img.save(os.path.join(label_dir, f"img_{idx//10}+{neg_label}.png"))
        elif FLAGS.sample_method == 'ddim_diff':
            # 1) 先离线得到 centers/df_map
            centers, df_map = compute_class_stats(net, trainloader, device)

            # 2) 统计训练集各类样本数，规划补齐
            per_class_need = plan_counts(train_counts=train_dataset.get_cls_num_list(), N_t=500, num_classes=FLAGS.num_class)  # CIFAR100-LT 示例

            # 3) 生成（严格复现：ddim skip=1, eta=0 或 ddpm）
            gen_imgs, gen_labels = generate_balanced(
                sampler, num_classes=FLAGS.num_class, per_class_need=per_class_need,
                batch_size=FLAGS.batch_size, img_size=FLAGS.img_size, device=device,
                phi0=net, centers=centers, df_map=df_map, keep_types=('ID', 'AID'),
                method=FLAGS.sample_method, skip=FLAGS.ddim_skip_step, eta=0.0
            )
            for idx, (img_tensor, neg_label) in enumerate(zip(gen_imgs, gen_labels.cpu().numpy())):
                # 转换为PIL
                img = transforms.ToPILImage()(img_tensor)

                # 保存到标签对应的子文件夹
                label_dir = os.path.join(save_dir, str(neg_label))
                os.makedirs(label_dir, exist_ok=True)

                img.save(os.path.join(label_dir, f"img_{idx}.png"))
        else:
            i = 0
            for inputs, labels in tqdm(trainloader):
                i+=1
                if FLAGS.sample_method == 'ddim':
                    x_T = torch.randn((64, 3, FLAGS.img_size, FLAGS.img_size))
                    y = torch.randint(FLAGS.num_class, size=(x_T.shape[0],), device=device)
                    # 采样生成图像
                    batch_images = sampler(
                        x_T.to(device),
                        y,
                        method=FLAGS.sample_method,
                        skip=FLAGS.ddim_skip_step,
                        w_neg=0.5
                    ).cpu()
                    batch_images = ((batch_images + 1) / 2).clamp(0, 1)
                elif FLAGS.sample_method == 'ddim_remove':
                    x0 = inputs.to(device)  # 值域要与训练一致（通常 [-1,1]）
                    y = labels.to(device)  # 目标类（要“去掉”的）
                    skip = FLAGS.ddim_skip_step
                    w_neg = 5
                    eta = 0

                    t_max = int(0.6 * (sampler.T - 1))
                    t_start = int(0.5 * (sampler.T - 1))
                    x_hi, reached_t = sampler.phase1_forward_remove_class(
                        x_init=x0,
                        y_target=y,
                        t_start=t_start,
                        skip=skip,
                        w_neg=w_neg,
                        t_max=t_max,
                        label=y,
                        # net=net.to(device)
                    )

                    # ---------- Phase 2: 无条件去噪（降序 t） ----------
                    x_out = sampler.phase2_denoise_uncond(
                        x_t=x_hi,
                        start_t=reached_t,
                        skip=skip,
                        eta=eta,
                        y_target=None
                    )

                    # 可视化（若输出域为 [-1,1]）
                    batch_images = ((x_out + 1) / 2).clamp(0, 1).detach().cpu()
                    for idx, (img_tensor, neg_label) in enumerate(zip(batch_images, y.cpu().numpy())):
                        # 转换为PIL
                        img = transforms.ToPILImage()(img_tensor)

                        # 保存到标签对应的子文件夹
                        label_dir = os.path.join(save_dir, str(neg_label))
                        os.makedirs(label_dir, exist_ok=True)

                        img.save(os.path.join(label_dir, f"img_{i}+{idx}.png"))

                elif FLAGS.sample_method == 'ddim_target':
                    # 假设已知 num_classes (int)
                    # 先在这个 batch 内收集
                    pseudo_img_list = []
                    pseudo_lbl_list = []
                    pseudo_lab_list = []
                    # 预先准备一个所有类别的列表（放在与labels相同的设备上/类型）
                    all_classes = torch.arange(num_class, device=labels.device, dtype=labels.dtype)
                    logits, _ = net(inputs.to(device))
                    K = min(num_class//3, num_class - 1)
                    for idx, (input, label) in enumerate(zip(inputs, labels)):

                        # 把真实标签对应的 logit 置为 -inf，确保不会被选入 topk
                        masked_logits = logits[idx].clone().to(device)
                        masked_logits[label] = -float('inf')
                        topk_idx = torch.topk(masked_logits, k=K, dim=0).indices

                        dup_imgs = input.to(device, non_blocking=True).unsqueeze(0).repeat(topk_idx.numel(), 1, 1, 1)
                        dup_labs = label.unsqueeze(0).repeat(topk_idx.numel(), )

                        pseudo_img_list.append(dup_imgs)
                        pseudo_lbl_list.append(topk_idx.to(device=device, dtype=labels.dtype))
                        pseudo_lab_list.append(dup_labs)
                    pseudo_images = torch.cat(pseudo_img_list, dim=0) if pseudo_img_list else torch.empty(0,
                                                                                                          *inputs.shape[
                                                                                                           1:],
                                                                                                          device=inputs.device)
                    pseudo_labels = torch.cat(pseudo_lbl_list, dim=0) if pseudo_lbl_list else torch.empty(0,
                                                                                                          dtype=labels.dtype,
                                                                                                          device=labels.device)
                    pseudo_labs = torch.cat(pseudo_lab_list, dim=0) if pseudo_lab_list else torch.empty(0,
                                                                                                          dtype=labels.dtype,
                                                                                                          device=labels.device)
                    img_out_chunks = []
                    for x0, y, lab in zip(
                            torch.split(pseudo_images.to(device), 512, dim=0),
                            torch.split(pseudo_labels.to(device), 512, dim=0),
                            torch.split(pseudo_labs.to(device), 512, dim=0)
                    ):
                        # x0 = pseudo_images.to(device)  # 值域要与训练一致（通常 [-1,1]）
                        # y = pseudo_labels.to(device)  # 目标类
                        skip = FLAGS.ddim_skip_step
                        w_neg = 5
                        eta = 0

                        t_max = int(0.6 * (sampler.T - 1))
                        t_start = int(0.5 * (sampler.T - 1))
                        x_hi, reached_t = sampler.phase1_forward_remove_class(
                            x_init=x0,
                            y_target=y,
                            t_start=t_start,
                            skip=skip,
                            w_neg=w_neg,
                            t_max=t_max,
                            label=lab
                            # net=net.to(device)
                        )

                        # ---------- Phase 2: 无条件去噪（降序 t） ----------
                        x_out = sampler.phase2_denoise_uncond(
                            x_t=x_hi,
                            start_t=reached_t,
                            skip=skip,
                            eta=eta,
                            y_target=y
                        )
                        img_out_chunks.append(((x_out + 1) / 2).clamp(0, 1).detach().cpu())
                        # 可视化（若输出域为 [-1,1]）

                    batch_images = torch.cat(img_out_chunks, dim=0)
                    for idx, (img_tensor, neg_label,label) in enumerate(zip(batch_images, pseudo_labels.cpu().numpy(),pseudo_labs.cpu().numpy())):
                        # 转换为PIL
                        img = transforms.ToPILImage()(img_tensor)

                        # 保存到标签对应的子文件夹
                        label_dir = os.path.join(save_dir, str(neg_label))
                        os.makedirs(label_dir, exist_ok=True)

                        img.save(os.path.join(label_dir, f"img_{i}+{idx}_{label}.png"))

    print(f"所有生成的图片已保存到 {save_dir}/[label]/img_xxx.png 目录下")

    print(f"所有生成的图片已保存到 {save_dir}/[label]/img_xxx.png 目录下")


def eval():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        cond=FLAGS.conditional, augm=FLAGS.augm, num_class=FLAGS.num_class)

    sampler = GaussianDiffusionSamplerOld(
            model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
             var_type=FLAGS.var_type, w=FLAGS.w,cond = True).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    if FLAGS.ckpt_step >= 0:
        ckpt = torch.load(os.path.join(FLAGS.logdir, f'ckpt_{FLAGS.ckpt_step}.pt'))
    else:
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
    state = ckpt['net_model']
    missing, unexpected = model.load_state_dict(state, strict=False)
    # model.load_state_dict(ckpt['net_model'])

    # (IS, IS_std), FID, samples = evaluate(sampler, model)
    # print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    # save_image(
    #    torch.tensor(samples[:256]),
    #    os.path.join(FLAGS.logdir, 'samples.png'),
    #    nrow=16)

    model.load_state_dict(ckpt['ema_model'], strict=False)
    evaluate(sampler, model, save_dir=f"./{FLAGS.dataset}_{FLAGS.imbalance}_testbounduray0.1fused", save=True, use_eval=True, save_intermediate=False)
    # print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    # save_image(
    #     torch.tensor(samples[:256]),
    #     os.path.join(FLAGS.logdir, 'samples_ema_{}.png'.format(FLAGS.specific_class)),
    #     nrow=16)













def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()


    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
