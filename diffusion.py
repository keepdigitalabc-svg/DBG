# from dd_code.backdoor.benchmarks.pytorch-ddpm.main import self
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
from tqdm import tqdm
import ipdb
from functools import partial

from torchvision.utils import save_image
import string

# class GaussianDiffusionSamplerOld(nn.Module):
#     def __init__(self, model, beta_1, beta_T, T, img_size=32,
#                  mean_type='epsilon', var_type='fixedlarge', w=2, cond=False):
#         assert mean_type in ['xprev', 'xstart', 'epsilon']  # ← 加上逗号
#         assert var_type in ['fixedlarge', 'fixedsmall']
#         super().__init__()
#
#         self.model = model
#         self.T = T
#         self.img_size = img_size
#         self.mean_type = mean_type
#         self.var_type = var_type
#         self.cond = cond
#         self.w = w
#         print(f"current guidance rate is {w}")
#
#         self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
#         alphas = 1. - self.betas
#         alphas_bar = torch.cumprod(alphas, dim=0)
#         alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
#
#         self.register_buffer('alphas_bar', alphas_bar)  # ← 你已有的名字
#         alphas = 1. - self.betas
#         alphas_bar = torch.cumprod(alphas, dim=0)
#
#
#         self.register_buffer(
#             'sqrt_alphas_bar', torch.sqrt(alphas_bar))
#         self.register_buffer(
#             'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
#         self.register_buffer(
#             'sigma_tsq', 1./alphas_bar-1.)
#         self.register_buffer('sigma_t',torch.sqrt(self.sigma_tsq))
#
#     def forward(self, x_0, y_0, augm=None,fix_t=None):
#         """
#         Algorithm 1.
#         """
#         # original codes
#         if fix_t is None:
#             t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
#         else:
#             t = torch.full((x_0.shape[0], ),fix_t)
#         noise = torch.randn_like(x_0)
#         ini_noise = noise
#
#         x_t = (
#             extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
#             extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
#
#         if self.cfg or self.cb:
#             if torch.rand(1)[0] < 1/10:
#                 y_0 = None
#         h,temp_mid = self.model(x_t, t, y=y_0, augm=augm)
#
#
#         return h,temp_mids


class GaussianDiffusionSamplerOld(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='epsilon', var_type='fixedlarge',w=2,cond=False):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.cond = cond
        self.w=w
        print(f"current guidance rate is {w}")
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer(
            'alphas_bar', alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t,
                        method='ddpm',
                        skip=1,
                        eps=None):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        if method == 'ddim':
            assert (eps is not None)
            skip_time = torch.clamp(t - skip, 0, self.T)
            posterior_mean_coef1 = torch.sqrt(extract(self.alphas_bar, t, x_t.shape))
            posterior_mean_coef2 = torch.sqrt(1-extract(self.alphas_bar, t, x_t.shape))
            posterior_mean_coef3 = torch.sqrt(extract(self.alphas_bar, skip_time, x_t.shape))
            posterior_mean_coef4 = torch.sqrt(1-extract(self.alphas_bar, skip_time, x_t.shape))
            posterior_mean = (
                posterior_mean_coef3 / posterior_mean_coef1 * x_t +
                (posterior_mean_coef4 - 
                posterior_mean_coef3 * posterior_mean_coef2 / posterior_mean_coef1) * eps
            )
        else:
            posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)

        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, y,y_uncond,method, skip):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t, y)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t ,y)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            if self.cond:
                eps = self.model(x_t, t ,y)
                eps_g=self.model(x_t, t ,y_uncond)
                eps=eps-(0.1)*(eps-eps_g)
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t, method, skip, eps)
            else:
                #ipdb.set_trace()
                eps = self.model(x_t, t)
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t, method, skip, eps)
                #print("un conditional!")
        else:
            raise NotImplementedError(self.mean_type)
        #x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def scale_model_input(self, x, t):
        return x

    @torch.no_grad()
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: int | torch.LongTensor):
        """
        x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * noise
        与你给出的 add_noise 语义一致
        """
        abar = self.alphas_bar.to(x0.device, dtype=x0.dtype)  # (T,)
        if isinstance(t, torch.Tensor):
            abar_t = abar[t].view(-1, 1, 1, 1)
        else:
            abar_t = abar[t].view(1, 1, 1).to(x0)
        return abar_t.sqrt() * x0 + (1.0 - abar_t).sqrt() * noise

    @torch.no_grad()
    def con_inverse_step(  # == 你参考代码里的“正向去类一步”
            self,
            model_output: torch.FloatTensor,  # ε_neg(x_t, t)
            timestep: int,  # t
            sample: torch.FloatTensor,  # x_t
            next_timestep: int,  # t_plus = t + skip（已裁到 [0, T-1]）
            clip_x0: bool = True,
    ):
        abar = self.alphas_bar.to(sample.device)

        abar_t = abar[timestep]
        abar_next = abar[next_timestep]

        # 条件 DDIM 反演（确定性，σ=0）的系数
        coef_x = (abar_next / abar_t).sqrt()
        coef_eps = (1.0 - abar_next).sqrt() - coef_x * (1.0 - abar_t).sqrt()

        # 直接线性一步：x_{t+} = coef_x * x_t + coef_eps * ε_cond
        next_sample = coef_x * sample + coef_eps * model_output

        # 可选：给出 x0_hat（便于外部监控/判定），与 DDIM 式(12)一致
        x0_hat = (sample - (1.0 - abar_t).sqrt() * model_output) / abar_t.sqrt()
        if clip_x0:
            x0_hat = x0_hat.clamp(-1, 1)

        return next_sample, x0_hat
    @torch.no_grad()
    # def con_inverse_step(  # == 你参考代码里的“正向去类一步”
    #         self,
    #         model_output: torch.FloatTensor,  # ε(x_t, t) —— 你传什么就用什么
    #         timestep: int,  # t
    #         sample: torch.FloatTensor,  # x_t
    #         next_timestep: int,  # t_plus = t + skip（已裁到 [0, T-1]）
    #         clip_x0: bool = True,
    # ):
    #     abar = self.alphas_bar.to(sample.device)
    #     abar_t = abar[timestep]
    #     abar_next = abar[next_timestep]
    #
    #     # 先按 DDIM 式(12)复原 x0_hat
    #     x0_hat = (sample - (1.0 - abar_t).sqrt() * model_output) / abar_t.sqrt()
    #     if clip_x0:
    #         x0_hat = x0_hat.clamp(-1, 1)
    #
    #     # === 新增：像素/特征控制（相对于原图锚点回拉） ===
    #     # 需要在 phase1/phase2 开始前设置一次：self._x_init_anchor = x_init（见下）
    #     x_init_anchor = getattr(self, "_x_init_anchor", x0_hat.detach())
    #     x0_ctrl = self._control_x0(x0_hat, x_init_anchor)
    #
    #     # === 关键：重算当前步的一致噪声 z_t，再用它合成下一步，保持 forward 关系一致 ===
    #     z_t = (sample - abar_t.sqrt() * x0_ctrl) / (1.0 - abar_t).sqrt()
    #     next_sample = abar_next.sqrt() * x0_ctrl + (1.0 - abar_next).sqrt() * z_t
    #
    #     return next_sample, x0_ctrl  # 返回受控后的 x0


    @torch.no_grad()
    def ddim_step(self, model_output: torch.FloatTensor, t_cur: int, t_prev: int, sample: torch.FloatTensor,
                  eta: float = 0.0):
        """
        显式 t_cur -> t_prev 的 DDIM 反向一步；model_output 是 ε(x_{t_cur}, t_cur, ·)
        """
        abar = self.alphas_bar.to(sample.device)
        abar_t = abar[t_cur]
        abar_tp = abar[t_prev]

        # x0_hat
        x0_hat = (sample - (1.0 - abar_t).sqrt() * model_output) / abar_t.sqrt()

        # DDIM 噪声项（η=0 即确定性）
        if eta > 0.0 and t_prev > 0:
            sigma = eta * torch.sqrt((1.0 - abar_tp) / (1.0 - abar_t) * (1.0 - abar_t / abar_tp + 1e-12))
            z = torch.randn_like(sample)
        else:
            sigma = sample.new_zeros(())
            z = torch.zeros_like(sample)

        # 合成 x_{t_prev}
        dir_xt = (1.0 - abar_tp).sqrt() * model_output
        x_prev = abar_tp.sqrt() * x0_hat + dir_xt + sigma * z
        return x_prev

    # ---------- 两个便捷循环：与“示例代码的两段”一一对应 ----------
    @torch.no_grad()
    def _control_x0(self, x0_hat: torch.Tensor, x_init_anchor: torch.Tensor) -> torch.Tensor:
        """
        对 x0_hat 施加像素级与特征级的回拉限制，返回 x0_ctrl 。
        依赖（可选）属性：
          - self.ctrl_pix_alpha ∈ [0,1]         # 像素回拉强度（越大越靠近锚点）
          - self.ctrl_pix_step_max (float|None)  # 单步像素最大扰动（L∞ 上限）
          - self.ctrl_mask (B,1,H,W)|None        # 1=可改, 0=冻结（可为软掩码）
          - self.ctrl_use_ema_anchor (bool)      # 是否用 EMA 的像素锚
          - self.ctrl_ema_rho ∈ (0,1)            # EMA 衰减
          - self.feat_encoder (nn.Module|None)   # 冻结的特征网络（如 CLIP/VGG），输入域 [0,1]
          - self.ctrl_feat_sim_thr ∈ (0,1)       # 特征相似度阈
          - self.ctrl_feat_alpha ∈ [0,1]         # 特征回拉强度
        """
        # ---------- 像素锚点 ----------
        anchor = getattr(self, "_pix_anchor", None)
        if anchor is None:
            anchor = x_init_anchor
        if getattr(self, "ctrl_use_ema_anchor", False):
            rho = getattr(self, "ctrl_ema_rho", 0.9)
            anchor = rho * anchor + (1 - rho) * x0_hat.detach()
        self._pix_anchor = anchor

        # ---------- A. 像素回拉 ----------
        alpha = getattr(self, "ctrl_pix_alpha", 0)  # 0=不回拉
        x0_ctrl = (1 - alpha) * x0_hat + alpha * anchor

        # ---------- B. 单步最大扰动 ----------
        x0_prev = getattr(self, "_x0_prev", x0_ctrl.detach())
        tau = getattr(self, "ctrl_pix_step_max", None)
        if tau is not None:
            delta = x0_ctrl - x0_prev
            scale = torch.clamp(
                tau / (delta.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8), max=1.0
            )
            x0_ctrl = x0_prev + scale * delta
        self._x0_prev = x0_ctrl.detach()

        # ---------- C. 掩码限制（冻结非 ROI） ----------
        M = getattr(self, "ctrl_mask", None)
        if M is not None:
            x0_ctrl = M * x0_ctrl + (1 - M) * anchor

        # ---------- D. 特征级回拉（无梯度） ----------
        feat_net = getattr(self, "feat_encoder", None)
        if feat_net is not None:
            with torch.no_grad():
                to01 = lambda z: (z + 1) * 0.5
                f_new = feat_net(to01(x0_ctrl))
                if not hasattr(self, "_feat_anchor"):
                    self._feat_anchor = feat_net(to01(x_init_anchor)).detach()
                f_ref = self._feat_anchor
                sim = torch.nn.functional.cosine_similarity(f_new, f_ref, dim=1, eps=1e-6).view(-1, 1, 1, 1)
                sim_thr = getattr(self, "ctrl_feat_sim_thr", 0.85)
                alpha_f = getattr(self, "ctrl_feat_alpha", 0.0)
                w = torch.clamp((sim_thr - sim).relu() / max(1e-6, sim_thr), 0, 1) * alpha_f
                x0_ctrl = (1 - w) * x0_ctrl + w * anchor

        return x0_ctrl

    # @torch.no_grad()
    # def phase1_forward_remove_class(self, x_init: torch.Tensor, y_target: torch.LongTensor,
    #                                 skip: int, w_neg: float, t_start: int, t_max: int | None = None, bootstrap_if_zero=True,
    #                                 bootstrap_ratio=0.5,label=None):
    #     """
    #     第一段：按升序 t=0..t_max 做“正向去类”：
    #     ε_neg = ε_uncond - w_neg * (ε_cond - ε_uncond)；然后 con_inverse_step 到 t+skip
    #     """
    #     B = x_init.shape[0]
    #     x = x_init
    #     T = self.T
    #     if t_max is None:
    #         t_max = T - 1
    #     # 与 skip 对齐
    #     t_max = (int(t_max) // skip) * skip
    #     t_start = (int(t_start) // skip) * skip
    #     # --- 关键改动：若 t_start=0，先“自举”到一个中间步 ---
    #     if t_start == 0 and bootstrap_if_zero:
    #         t_boot = (int(bootstrap_ratio * (T - 1)) // skip) * skip  # 例如 0.5*(T-1)
    #         z0 = torch.randn_like(x_init)
    #         x = self.add_noise(x_init, z0, torch.full((B,), t_boot, device=x_init.device, dtype=torch.long))
    #         t_begin = t_boot
    #     else:
    #         if t_start > 0:
    #             z0 = torch.randn_like(x_init)
    #             x = self.add_noise(x_init, z0, torch.full((B,), t_start, device=x_init.device, dtype=torch.long))
    #         else:
    #             x = x_init
    #         t_begin = t_start
    #     for t in range(t_begin, t_max + 1, skip):
    #         t_plus = min(t + skip, T - 1)
    #
    #         t_vec = torch.full((B,), t, device=x.device, dtype=torch.long)
    #
    #         # 负向 CFG：需要“无条件分支”；你训练时见过 None，这里就用 None
    #         eps_cond = self.model(x, t_vec, y_target)  # 有条件
    #         eps_uncond = self.model(x, t_vec, label)  # 要去掉的类
    #         eps_un = self.model(x, t_vec, None)  # 无条件
    #         # eps_neg = eps_un + self.w * (eps_cond - eps_un) - self.w * (eps_uncond - eps_un)
    #         eps_neg = eps_uncond
    #         x, _ = self.con_inverse_step(eps_neg, t, x, next_timestep=t_plus)
    #
    #         if t == t_max:
    #             break  # 已经到达目标最嘈步
    #     return x, t_max
    @torch.no_grad()
    def phase1_forward_remove_class(self, x_init: torch.Tensor, y_target: torch.LongTensor,
                                    skip: int, w_neg: float, t_start: int, t_max: int | None = None,
                                    bootstrap_if_zero=True,
                                    bootstrap_ratio=0.5, label=None):
        """
        第一段：按升序 t=0..t_max 做“正向去类”
        （保留你的参数名/调用方式不变；内部增加像素/特征控制与噪声一致性）
        """
        B = x_init.shape[0]
        x = x_init
        T = self.T
        if t_max is None:
            t_max = T - 1
        t_max = (int(t_max) // skip) * skip
        t_start = (int(t_start) // skip) * skip

        # 设定全程的锚点（用于像素/特征回拉）
        self._x_init_anchor = x_init.detach()
        self._pix_anchor = x_init.detach()
        self._x0_prev = x_init.detach()
        # 特征锚会在 _control_x0 内部懒加载

        if t_start == 0 and bootstrap_if_zero:
            t_boot = (int(bootstrap_ratio * (T - 1)) // skip) * skip
            z0 = torch.randn_like(x_init)
            x = self.add_noise(x_init, z0, torch.full((B,), t_boot, device=x_init.device, dtype=torch.long))
            t_begin = t_boot
        else:
            if t_start > 0:
                z0 = torch.randn_like(x_init)
                x = self.add_noise(x_init, z0, torch.full((B,), t_start, device=x_init.device, dtype=torch.long))
            else:
                x = x_init
            t_begin = t_start

        for t in range(t_begin, t_max + 1, skip):
            t_plus = min(t + skip, T - 1)
            t_vec = torch.full((B,), t, device=x.device, dtype=torch.long)

            # 你原本的选择（这里保留）：用哪个 ε 就传哪个
            eps_cond = self.model(x, t_vec, y_target)  # 条件
            eps_uncond = self.model(x, t_vec, label)  # 你传的 label（可能是“要去掉的类”）
            eps_un = self.model(x, t_vec, None)  # 无条件

            # 你可以继续用自己的组合；这里只沿用你最后一行：“eps_neg = eps_uncond”
            eps_neg = eps_uncond

            # 注意：con_inverse_step 内部已做控制与一致性重建
            x, _ = self.con_inverse_step(eps_neg, t, x, next_timestep=t_plus)

            if t == t_max:
                break
        return x, t_max
    @torch.no_grad()
    def forward(
            self,
            x_T,
            y,  # 目标类
            method: str = '',
            skip: int = 1,
            return_intermediate: bool = False,
            w_neg: float = 0,
            eta: float = 0.0,
            y_uncond=None,
    ):
        x_t = x_T
        if return_intermediate:
            xt_list = []

        for time_step in reversed(range(0, self.T, skip)):
            if method == 'ddim_remove':
                x_t = self.con_inverse_step_remove_class(
                    model=self.model,
                    sample=x_t,
                    timestep=time_step,
                    y_target=y,
                    y_uncond=y_uncond,
                    w_neg=w_neg,
                    eta=eta,
                    skip=skip,  # ← 关键：传进单步函数
                    generator=None,
                    return_pred_x0=False,
                )
            elif method == 'ddim':
                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y,y_uncond=y_uncond, method='ddim', skip=skip)
                x_t = mean
            else:  # 'ddpm'
                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y, method='ddpm', skip=skip)
                noise = torch.randn_like(x_t) if time_step > 0 else 0
                x_t = mean + torch.exp(0.5 * log_var) * noise

            if return_intermediate:
                xt_list.append(x_t.detach().cpu())

        x_0 = x_t
        return (torch.clip(x_0, -1, 1), xt_list) if return_intermediate else torch.clip(x_0, -1, 1)
    @torch.no_grad()
    def phase2_denoise_uncond(self, x_t: torch.Tensor, start_t: int, skip: int, eta: float = 0.0, y_target=None):
        """
        第二段：从 start_t 开始按降序无条件 DDIM 去噪到 0
        （我们在这里也加入像素/特征控制与噪声一致性；保留参数名）
        """
        B = x_t.shape[0]
        x = x_t

        # 设定锚点（若 phase1 已设，这里只是覆盖成当前批的原图锚）
        # 如果你希望锚点始终是 phase1 的 x_init，可把下面两行注释掉
        # self._x_init_anchor = x_t.detach()  # 通常不需要；保持 phase1 的锚点更贴近“与原图差距小”
        # self._pix_anchor = getattr(self, "_pix_anchor", x_t.detach())

        abar = self.alphas_bar.to(x.device)

        for t in range(start_t, -1, -skip):
            t_prev = max(t - skip, 0)
            t_vec = torch.full((B,), t, device=x.device, dtype=torch.long)

            # 你的无条件 + CFG（你原代码写的是 cond/uncond 混合）
            x_scaled = self.scale_model_input(x, t)
            epsuncond = self.model(x_scaled, t_vec, None)
            epscond = self.model(x_scaled, t_vec, y_target)
            eps = epsuncond + self.w * (epscond - epsuncond)

            # ---- 经典 DDIM 复原 x0_hat ----
            abar_t = abar[t]
            x0_hat = (x - (1.0 - abar_t).sqrt() * eps) / abar_t.sqrt()
            x0_hat = x0_hat.clamp(-1, 1)

            # ---- 像素/特征控制 ----
            x0_ctrl = self._control_x0(x0_hat, getattr(self, "_x_init_anchor", x0_hat.detach()))

            # ---- 噪声一致性重建，再合成到 t_prev （Deterministic DDIM；eta 忽略或自行加 sigma）----
            abar_prev = abar[t_prev]
            z_t = (x - abar_t.sqrt() * x0_ctrl) / (1.0 - abar_t).sqrt()

            # 若你想用 eta，引入 sigma_t（可选）
            if eta and t_prev != t:
                # DDIM sigma_t 公式
                sigma_t = eta * ((1 - abar_prev) / (1 - abar_t)).sqrt() * (1 - abar_t / abar_prev).sqrt()
                noise = torch.randn_like(x)
                # 先把 eps 部分替换为与一致性 z_t 对应的项：√(1-abar_prev)*z_t
                # 再为随机性空出 sigma_t：√(1-abar_prev - sigma_t^2)*z_t + sigma_t*noise
                coef_eps = (1.0 - abar_prev - sigma_t ** 2).clamp(min=0.0).sqrt()
                x = abar_prev.sqrt() * x0_ctrl + coef_eps * z_t + sigma_t * noise
            else:
                # 确定性：保持 forward 关系
                x = abar_prev.sqrt() * x0_ctrl + (1.0 - abar_prev).sqrt() * z_t

            if t == 0:
                break
        return x
    # def forward(self, x_T, y, method='ddim', skip=10,return_intermediate=False):
    #     """
    #     Algorithm 2.
    #         - method: sampling method, default='ddpm'
    #         - skip: decrease sampling steps from T/skip, default=1
    #     """
    #     x_t = x_T
    #     if return_intermediate:
    #         xt_list = []
    #
    #     for time_step in reversed(range(0, self.T,skip)):
    #         t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
    #         mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y, method=method, skip=skip)
    #         # no noise when t == 0
    #         if time_step > 0:
    #             noise = torch.randn_like(x_t)
    #         else:
    #             noise = 0
    #
    #         if method == 'ddim':
    #             # ODE for DDIM
    #             x_t = mean
    #         else:
    #             # SDE for DDPM
    #             x_t = mean + torch.exp(0.5 * log_var) * noise
    #             # # delete this line
    #             # x_t_Guided=mean_Guided + torch.exp(0.5 * log_var_Guided) * noise
    #         if return_intermediate:
    #             xt_list.append(x_t.cpu())
    #
    #         # update guidance in every step
    #         #x_t = mean + torch.exp(0.5 * log_var) * noise
    #     x_0 = x_t
    #     if return_intermediate:
    #         return torch.clip(x_0, -1, 1),xt_list
    #     return torch.clip(x_0, -1, 1)



def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def uniform_sampling(n, N, k):
    return np.stack([np.random.randint(int(N/n)*i, int(N/n)*(i+1), k) for i in range(n)])


def dist(X, Y):
    sx = torch.sum(X**2, dim=1, keepdim=True)
    sy = torch.sum(Y**2, dim=1, keepdim=True)
    return torch.sqrt(-2 * torch.mm(X, Y.T) + sx + sy.T)


def topk(y, all_y, K):
    dist_y = dist(y, all_y)
    return torch.topk(-dist_y, K, dim=1)[1]


class GaussianDiffusionTrainer(nn.Module):
    # def __init__(self,
    #              model, beta_1, beta_T, T, dataset,
    #              num_class, cfg, weight,transfer_x0=True,
    #              mixing=False,transfer_mode='full',transfer_only_uncond=False,
    #              transfer_label=False,transfer_tr_tau=False,label_weight_tr = None,
    #              count=False,cut_time=-1,transfer_only_cond=False,uncond_flag_from_out=False,
    #              double_transfer=False):
    #     super().__init__()
    def __init__(self,
                 model, beta_1, beta_T, T, dataset,
                 num_class, cfg, weight,transfer_x0=True,
                 mixing=False,transfer_mode='full',transfer_only_uncond=False,
                 transfer_label=False,transfer_tr_tau=False,label_weight_tr = None,
                 count=False,cut_time=-1,transfer_only_cond=False,uncond_flag_from_out=False,
                 double_transfer=False, cond=False,
                 # --- NEW ---
                 clf=None, adv_train=False, adv_weight=0.1, adv_topk=3, adv_eta=0.05, adv_reconstruct=False):
        super().__init__()

        self.model = model
        self.T = T
        self.dataset = dataset
        self.num_class = num_class
        self.cfg = cfg
        self.transfer_mode = transfer_mode
        self.weight = weight
        self.transfer_x0 = transfer_x0
        self.transfer_label=transfer_label
        self.transfer_only_uncond = transfer_only_uncond
        self.transfer_tr_tau = transfer_tr_tau
        self.label_weight_tr = label_weight_tr
        self.mixing = mixing
        self.count = count
        self.cut_time = cut_time
        self.cond = cond
        # --- NEW: adversarial config ---
        self.clf = clf
        self.adv_train = adv_train
        self.adv_weight = float(adv_weight)
        self.adv_topk = int(adv_topk)
        self.adv_eta = float(adv_eta)
        self.adv_reconstruct = bool(adv_reconstruct)
        if count:
            self.total_count = np.zeros(T)
            self.transfer_count = np.zeros(T)
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).cuda()


        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(self.alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - self.alphas_bar))
        self.register_buffer(
            'sigma_tsq', 1./self.alphas_bar-1.)
        self.register_buffer('sigma_t',torch.sqrt(self.sigma_tsq))

    def forward(self, x_0, y_0, augm=None,uncond_flag_out=False):
        """
        Algorithm 1.
        """
        # original codes
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0) 
        ini_noise = noise

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        uncond_flag = False
        y_l = y_0

        if self.cfg:
            if torch.rand(1)[0] < 1/10:
                y_l = None
                uncond_flag = True
            else:
                y_l = y_0

        h = self.model(x_t, t, y=y_l, augm=augm)
        if self.transfer_x0:
            cx_t = x_0 + extract(self.sigma_t, t, x_0.shape) * noise
            if self.transfer_tr_tau:
                noise = self.do_transfer_x0_with_y(x_t,cx_t,x_0,t,y_0,self.label_weight_tr)
            else:
                noise,_ = self.do_transfer_x0(x_t,cx_t,x_0,t,y_0,return_transfer_label=True)

        loss = F.mse_loss(h, noise, reduction='none')
        loss_reg = loss_com = torch.tensor(0).to(x_t.device)
        # Adversarial margin loss (DeepFool-style, batch)
        # ==========================================
        adv_loss = x_0.new_tensor(0.0)

        if self.adv_train and (self.clf is not None) and (self.adv_weight > 0):
            # 1) 取和 ddpm 主损同一批次/同一时刻的 x_t、eps_pred，得到去噪估计 x0_pred
            # 假定你已有 t ~ Uniform{1..T}，eps ~ N(0,I)，以及 x_t = sqrt(a_bar[t])*x_0 + sqrt(1-a_bar[t])*eps
            # 如果你的代码里这些变量名不同，请用你原有的。
            with torch.enable_grad():
                # 若上面你对 eps_pred/x_t 做了 no_grad()，这里要重新 forward 一次以建图
                x_t_for_adv = x_t.detach().clone().requires_grad_(True)
                eps_pred_adv = self.model(x_t_for_adv, t, y_0 if self.cond else None)
                # 估计 x0_pred
                a_bar_t = self.alphas_bar[t] if isinstance(self.alphas_bar, torch.Tensor) else self.alphas_bar[t.item()]
                if not torch.is_tensor(a_bar_t):
                    a_bar_t = torch.tensor(a_bar_t, device=x_t_for_adv.device, dtype=x_t_for_adv.dtype)
                a_bar_t = a_bar_t.view(-1, *([1] * (x_t_for_adv.dim() - 1)))  # broadcast to NCHW
                x0_pred = (x_t_for_adv - torch.sqrt(1 - a_bar_t) * eps_pred_adv) / torch.sqrt(a_bar_t)
                x0_pred = x0_pred.to(torch.float32)
                # 2) 分类器 logits
                logits = self.clf(x0_pred)
                N, C = logits.shape

                # 原类：如果是条件扩散，用数据标签 y_0；否则用预测最大类作为“当前类”
                if getattr(self.model, 'cond', False):
                    cur_cls = y_0.clone()
                else:
                    cur_cls = logits.argmax(1)

                # 3) 选择 rival 类：对每个样本取 top-(K+1)，剔除 cur_cls，选 margin 最小者
                K = min(max(int(self.adv_topk), 1), C - 1)
                topv, topi = torch.topk(logits, k=K + 1, dim=1)  # (N, K+1)
                # 去掉本类
                mask_not_cur = topi != cur_cls.unsqueeze(1)
                # 对每行保留前 K 个非本类候选
                cand_idx = []
                for i in range(N):
                    cand = topi[i][mask_not_cur[i]]
                    if cand.numel() == 0:
                        cand = torch.arange(C, device=topi.device)
                        cand = cand[cand != cur_cls[i]][:K]
                    else:
                        cand = cand[:K]
                    cand_idx.append(cand)
                # 拼成一个 (sum_k, ) 的索引批以便一次性求梯度
                # 计算 margin = f_c - f_c' （希望尽量小/为负：跨界）
                rows = torch.arange(N, device=logits.device)
                logit_c = logits[rows, cur_cls]  # (N,)
                # 为了向量化：构造 rival logits，逐样本 gather
                max_rival = []
                for i in range(N):
                    l_rivals = logits[i, cand_idx[i]]  # (K_i,)
                    # 近似 DeepFool：选使 margin 最小的 rival
                    margins_i = logit_c[i] - l_rivals
                    arg = torch.argmin(margins_i)
                    max_rival.append(l_rivals[arg])
                logit_rival = torch.stack(max_rival, dim=0)

                # margin: f_c - f_c'，我们想把它变小 -> loss = +margin
                margin = logit_c - logit_rival
                adv_margin_loss = margin.mean()  # 越小越好

                # ========== 可选：构造 x0_adv 并做重建一致性 ==========
                if self.adv_reconstruct:
                    # 对 margin 做一次 DeepFool/FGSM 风格的内层更新
                    # d(margin)/dx0 = ∇(f_c - f_c')
                    # 这里简单用 sign 或标准化梯度，你也可以换成更精细的 DeepFool 步长
                    with torch.no_grad():
                        grad = torch.autograd.grad(adv_margin_loss, x0_pred, retain_graph=False)[0]
                        # 归一化以避免尺度问题
                        grad_norm = grad.view(N, -1).norm(dim=1).clamp_min(1e-8).view(N, 1, 1, 1)
                        step = self.adv_eta * (grad / grad_norm)
                        x0_adv = (x0_pred - step).detach()  # 减小 margin（推向跨界）
                        # 用同一 eps 和同一 t，把 x0_adv 正向加噪，形成 x_t_adv
                        eps_used = (x_t_for_adv - torch.sqrt(a_bar_t) * x_0) / torch.sqrt(1 - a_bar_t) \
                            if 'eps' not in locals() else 2.0  # 若你已有 eps，就直接用
                        # 更稳妥：显式重算 eps
                        eps_from_xt = (x_t_for_adv - torch.sqrt(a_bar_t) * x_0) / torch.sqrt(1 - a_bar_t)
                        x_t_adv = torch.sqrt(a_bar_t) * x0_adv + torch.sqrt(1 - a_bar_t) * eps_from_xt
                        x_t_adv = x_t_adv.float()
                        # 让模型在 x_t_adv 上仍能还原 eps（重建一致性）
                        eps_pred_on_adv = self.model(x_t_adv, t, y_0 if self.cond else None)
                        adv_recon_loss = torch.mean((eps_pred_on_adv - eps_from_xt) ** 2)
                        adv_loss = self.adv_weight * (adv_margin_loss + adv_recon_loss)
                else:
                    adv_loss = self.adv_weight * adv_margin_loss

        # 把对抗损失加到总损失
        loss = loss.mean() + adv_loss
        return loss, loss_reg + 1/4 * loss_com

    def do_transfer_x0(self,x_t,cx_t,x_0,t,y,return_transfer_label=False,mode=None,x_ref=None):
        '''
        new item for this function:
        restrict the transfer direction from long to tail or tail to long.
        '''
        if mode is not None:
            this_mode = mode
        else:
            this_mode = self.transfer_mode
        with torch.no_grad():
            bs,ch,h,w = x_0.shape
            ### here we should change the defination of the x_t

            x_t1 = cx_t.reshape(len(x_t),-1)
            x_01 = x_0.reshape(len(x_0),-1)
            '''
            here we should decay the initial signal by sqrt{alpha_t}
            '''
            com_dis = x_t1.unsqueeze(1) - x_01
            gt_distance = torch.sum((x_t1.unsqueeze(1) - x_01)**2,dim=[-1])
            normalize_distance = 2*extract(self.sigma_tsq, t, gt_distance.shape)

            #distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            gt_distance = - gt_distance / normalize_distance
            distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            distance = torch.exp(distance)
            # add y prior knowledge
            # self-normalize the per-sample weight of reference batch
            weights = distance / (torch.sum(distance, dim=1, keepdim=True))

            new_ind = torch.multinomial(weights,1)
            # here we wanted to record the transfer probability


            new_ind = new_ind.squeeze(); ini_ind = torch.arange(x_0.shape[0]).cuda()
            transfer_label = y[new_ind]
            old_prob = self.weight.squeeze().cuda().gather(0,y)
            new_prob = self.weight.squeeze().cuda().gather(0,transfer_label)
            #here add the restriction item, just make judgement!
            if this_mode == 't2h':
                # ipdb.set_trace()
                # here we implement the long to tail transfer
                # firstly we should obtain the y label to the corresponding images
                # initial label is the y 
                if self.cut_time < 0:
                    new_ind_f = torch.where(new_prob>=old_prob,new_ind,ini_ind)
                else:
                    new_ind_f1 = torch.where(new_prob>=old_prob ,new_ind,ini_ind)
                    new_ind_f = torch.where(t < self.cut_time,new_ind_f1,ini_ind)
            elif this_mode == 'h2t':
                if self.cut_time < 0:
                    new_ind_f = torch.where(new_prob<=old_prob,new_ind,ini_ind)
                else:
                    new_ind_f1 = torch.where(new_prob<=old_prob,new_ind,ini_ind)
                    new_ind_f = torch.where(t < self.cut_time,new_ind_f1,ini_ind)
            elif this_mode == 'none':
                # 不做任何迁移，始终保持初始索引
                new_ind_f = ini_ind
            elif this_mode == 'full':
                new_ind_f = new_ind
            else:
                raise NotImplementedError
            x_n0 = x_0[new_ind_f]

            new_epsilon =  (x_t - extract(self.sqrt_alphas_bar, t, x_0.shape)*x_n0) / extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)
                        # then record the things 
            if self.count:
                reference = list(range(len(new_ind_f)))
                for i in range(len(reference)):
                    if new_ind_f[i] == reference[i]:
                        self.total_count[t[i].item()] +=1
                    else:
                        self.total_count[t[i].item()] +=1
                        self.transfer_count[t[i].item()]+=1

            if return_transfer_label:
                return new_epsilon,transfer_label
            return new_epsilon

    def do_transfer_x0_with_y(self,x_t,cx_t,x_0,t,y,weight_label):
        '''
        new item for this function:
        restrict the transfer direction from long to tail or tail to long.
        '''
        with torch.no_grad():
            bs,ch,h,w = x_0.shape
            ### here we should change the defination of the x_t

            x_t1 = cx_t.reshape(len(x_t),-1)
            x_01 = x_0.reshape(len(x_0),-1)
            '''
            here we should decay the initial signal by sqrt{alpha_t}
            '''
            com_dis = x_t1.unsqueeze(1) - x_01
            gt_distance = torch.sum((x_t1.unsqueeze(1) - x_01)**2,dim=[-1])
            normalize_distance = 2*extract(self.sigma_tsq, t, gt_distance.shape)

            #distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            gt_distance = - gt_distance / normalize_distance
            distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
            wl = weight_label.cuda()
            reweight = torch.gather(wl[y],1,y.unsqueeze(0).repeat(bs,1))
                         #distance = torch.exp(distance) * weight_label
            distance = reweight * torch.exp(distance)#distance
            # self-normalize the per-sample weight of reference batch
            weights = distance / (torch.sum(distance, dim=1, keepdim=True))
            new_ind = torch.multinomial(weights,1)
            new_ind = new_ind.squeeze(); ini_ind = torch.arange(x_0.shape[0]).cuda()

            new_ind_f = new_ind
            x_n0 = x_0[new_ind_f]
            new_epsilon =  (x_t - extract(self.sqrt_alphas_bar, t, x_0.shape)*x_n0) / extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape)

                        # then record the things 
            return new_epsilon





class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, num_class, img_size=32, var_type='fixedlarge'):
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.num_class = int(num_class)
        self.img_size = img_size
        self.var_type = var_type
        
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer(
            'alphas_bar', alphas_bar)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps): 
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    ### May change it to cg mode.
    

    def p_mean_variance(self, x_t, t, y=None, omega=0.0, method='free'):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped}[self.var_type]

        model_log_var = extract(model_log_var, t, x_t.shape)
        unc_eps = None
        augm = torch.zeros((x_t.shape[0], 9)).to(x_t.device)

        # Mean parameterization
        eps = self.model(x_t, t, y=y, augm=augm)
        if omega > 0 and (method == 'cfg'):
            unc_eps = self.model(x_t, t, y=None, augm=None)
            guide = eps - unc_eps
            eps = eps + omega * guide
        
        x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
        model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def forward(self, x_T, omega=0.0, method='cfg'):
        """
        Algorithm 2.
        """
        x_t = x_T.clone()
        y = None

        if method == 'uncond':
            y = None
        else:
            y = torch.randint(0, self.num_class, (len(x_t),)).to(x_t.device)

        with torch.no_grad():
            for time_step in tqdm(reversed(range(0, self.T)), total=self.T):
                t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, log_var = self.p_mean_variance(x_t=x_t, t=t, y=y,
                                                     omega=omega, method=method)

                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                
                x_t = mean + torch.exp(0.5 * log_var) * noise

        return torch.clip(x_t, -1, 1), y



class GaussianDiffusionSamplerCond(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge',w=2,cond=False):
        # assert mean_type in ['xprev' 'xstart', 'eps']
        # assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.cond = cond
        self.w=w
        print(f"current guidance rate is {w}")
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer(
            'alphas_bar', alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t,
                        method='ddpm',
                        skip=1,
                        eps=None):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)

        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape)

    def p_mean_variance(self, x_t, t, y,method, skip):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)


        eps = self.model(x_t, t, y)
        x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
        #x_0 = x_0.clamp(-1,1)
        model_mean, _ = self.q_mean_variance(x_0, x_t, t, method='ddpm', skip=10, eps=eps)
        #x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var,x_0,eps



    def condition_score(self, cond_fn, x_0, x, t, y,method='ddim',skip=10):
        """
        Borrow from guided diffusion "Diffusion Beat Gans in Image Synthesis"
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract(self.alphas_bar, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, x_0)
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, y)


        # out = p_mean_var.copy()
        # out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        # out["mean"], _, _ = self.q_posterior_mean_variance(
        #     x_start=out["pred_xstart"], x_t=x, t=t
        # )
        cond_x0 = self.predict_xstart_from_eps(x, t, eps)
        cond_mean, _ = self.q_mean_variance(cond_x0, x, t,
                        method='ddpm',
                        skip=skip,
                        eps=eps)

        return cond_x0,cond_mean


    def forward(self, x_T, y, method='ddim', skip=10,cond_fn=None):
        """
        Algorithm 2.
            - method: sampling method, default='ddpm'
            - skip: decrease sampling steps from T/skip, default=1
        """
        x_t = x_T

        for time_step in reversed(range(0, self.T,skip)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            t = t.to(x_t.device)
            mean, log_var, pred_x0, eps = self.p_mean_variance(x_t=x_t, t=t, y=y, method='ddpm', skip=skip)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            if method == 'ddim':
                #x_t = mean
                # ODE for DDIM
                pred_x0, cond_mean = self.condition_score(cond_fn, pred_x0, mean, t, y,method='ddpm',skip=skip)
                eps = self._predict_eps_from_xstart(x_t,t,pred_x0)
                # x_t, _ = self.q_mean_variance(pred_x0, x_t, t,method='ddim',skip=skip,eps=eps)
                
                assert (eps is not None)
                skip_time = torch.clamp(t - skip, 0, self.T)
                posterior_mean_coef1 = torch.sqrt(extract(self.alphas_bar, t, x_t.shape))
                posterior_mean_coef2 = torch.sqrt(1-extract(self.alphas_bar, t, x_t.shape))
                posterior_mean_coef3 = torch.sqrt(extract(self.alphas_bar, skip_time, x_t.shape))
                posterior_mean_coef4 = torch.sqrt(1-extract(self.alphas_bar, skip_time, x_t.shape))
                x_t = (
                    posterior_mean_coef3 / posterior_mean_coef1 * x_t +
                    (posterior_mean_coef4 - 
                    posterior_mean_coef3 * posterior_mean_coef2 / posterior_mean_coef1) * eps
                )


            else:
                # SDE for DDPM
                x_t = mean + torch.exp(0.5 * log_var) * noise
                # # delete this line
                # x_t_Guided=mean_Guided + torch.exp(0.5 * log_var_Guided) * noise

            # update guidance in every step
            #x_t = mean + torch.exp(0.5 * log_var) * noise

        x_0 = x_t

        return torch.clip(x_0, -1, 1),y






