from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer

from mint.config.base import Config


@dataclass
class AdamConfig(Config):
    adamw_beta1: float = 0.8
    adamw_beta2_lm_head: float = 0.96
    adamw_beta2_embedding: float = 0.995
    adamw_beta2_scalar: float = 0.95
    adamw_eps: float = 1e-10


@dataclass
class MuonConfig(Config):
    muon_momentum: float = 0.95
    muon_beta2: float = 0.9
    muon_ns_steps: int = 5


@dataclass
class MuonAdamWConfig(AdamConfig, MuonConfig):
    pass


POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class MuonAdamW(Optimizer):
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})

        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._fused_adam = torch.compile(
            self._adamw_step_fused,
            fullgraph=True,
            dynamic=False,
        )

        self._fused_muon = torch.compile(
            self._muon_step_fused,
            fullgraph=True,
            dynamic=False,
        )

    @staticmethod
    def _adamw_step_fused(
        p: Tensor,
        grad: Tensor,
        exp_avg: Tensor,  # First moment (same shape as p)
        exp_avg_sq: Tensor,  # Second moment (same shape as p)
        step_t: Tensor,  # 0-D CPU tensor: step count
        lr_t: Tensor,  # 0-D CPU tensor: learning rate
        beta1_t: Tensor,  # 0-D CPU tensor: beta1
        beta2_t: Tensor,  # 0-D CPU tensor: beta2
        eps_t: Tensor,  # 0-D CPU tensor: epsilon
        wd_t: Tensor,  # 0-D CPU tensor: weight decay
    ) -> None:
        p.mul_(1 - lr_t * wd_t)

        exp_avg.lerp_(grad, 1 - beta1_t)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)

        bias1 = 1 - beta1_t**step_t
        bias2 = 1 - beta2_t**step_t

        denom = (exp_avg_sq / bias2).sqrt() + eps_t
        step_size = lr_t / bias1
        p.add_(exp_avg / denom, alpha=-step_size.item())

    @staticmethod
    def _muon_step_fused(
        stacked_grads: Tensor,  # (N, H, W) - stacked gradients
        stacked_params: Tensor,  # (N, H, W) - stacked parameters
        momentum_buffer: Tensor,  # (N, H, W) - momentum buffer
        second_momentum_buffer: Tensor,  # (N, H, 1) or (N, 1, W) - variance buffer
        momentum_t: Tensor,  # 0-D CPU tensor: momentum coefficient
        lr_t: Tensor,  # 0-D CPU tensor: learning rate
        wd_t: Tensor,  # 0-D CPU tensor: weight decay
        beta2_t: Tensor,  # 0-D CPU tensor: beta2 for variance
        ns_steps: int,  # Number of Newton-Schulz iterations
        red_dim: int,  # Reduction dimension (-1 or -2)
    ) -> None:

        # Nesterov momentum
        momentum = momentum_t.to(stacked_grads.dtype)
        momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        g = stacked_grads.lerp_(momentum_buffer, momentum)

        # Polar Express orthogonalization
        # Use bfloat16 for speed when available (fp16 is unstable here)
        X = g.bfloat16() if g.dtype == torch.bfloat16 else g
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)

        if g.size(-2) > g.size(-1):  # Tall matrix
            for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
                A = X.mT @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:  # Wide matrix
            for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
                A = X @ X.mT
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        g = X

        # Variance reduction (NorMuon)
        beta2 = beta2_t.to(g.dtype)
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
        v_norm = v_norm_sq.sqrt()

        second_momentum_buffer.lerp_(v_mean.to(second_momentum_buffer.dtype), 1 - beta2)
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
        g = g * final_scale.to(g.dtype)

        # Cautious weight decay + parameter update
        # Only apply weight decay when gradient aligns with parameter
        lr = lr_t.to(g.dtype)
        wd = wd_t.to(g.dtype)
        mask = (g * stacked_params) >= 0
        stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

    def _step_adamw(self, group: dict) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            # Lazy state initialization
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1

            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])

            self._fused_adam(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        params = group["params"]
        if not params:
            return

        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                num_params, *shape, dtype=dtype, device=device
            )
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(
                state_shape, dtype=dtype, device=device
            )

        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)

        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        self._fused_muon(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        return loss
