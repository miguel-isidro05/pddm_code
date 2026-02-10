"""
Synthetic EEG Channel Generation using Score-Based Generative Modeling (SGM / ScoreNet)

Drop-in replacement for `ddpm.py`:
- Same conditional setup: model input is `torch.cat([conditioning_inputs, x_t], dim=1)`
- Same public function signatures: `train_diffusion_model(...)` and `sample_from_model(...)`

This version follows Score-Based Generative Modeling (Song et al.) with an explicit SDE:
- Continuous time t in (0, 1] (sampled as eps + (1-eps)*U[0,1])
- Network outputs the score: s_theta(x, t) with same shape as x
- Denoising Score Matching (DSM) loss: E[ || s_theta(x_t,t)*sigma(t) + z ||^2 ]
- EMA of parameters for stable sampling/metrics
- Predictor–Corrector (PC) sampler:
  - Predictor: Euler–Maruyama over reverse-time SDE
  - Corrector: Langevin dynamics with step size from fixed SNR

Hyperparams to report/ablate (explicit defaults):
- snr = 0.1
- num_steps = 2000
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Set device (works on macOS as well)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Keep for API-compatibility with ddpm.py (unused here).
def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


# DDPM-like sinusoidal embedding (kept for familiarity / compatibility).
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# Official-style time embedding for ScoreNet (Gaussian Fourier features).
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense1D(nn.Module):
    """A fully connected layer that reshapes outputs to 1D feature maps."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)[..., None]


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _best_group_count(num_channels: int, max_groups: int = 32) -> int:
    max_groups = min(max_groups, num_channels)
    for g in range(max_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def _group_norm(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(_best_group_count(num_channels, max_groups=max_groups), num_channels=num_channels)


def marginal_prob_std(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Standard deviation of the perturbation kernel p_{0t}(x(t) | x(0)) for VE SDE.
    Matches the formulation used in the official ScoreNet tutorial.
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device)
    t = t.to(device)
    return torch.sqrt((sigma ** (2.0 * t) - 1.0) / (2.0 * np.log(sigma)))


def diffusion_coeff(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Diffusion coefficient g(t) for the VE SDE."""
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device)
    t = t.to(device)
    return sigma**t


# -----------------------
# VP-SDE helpers (beta linear) — better scaled for z-scored EEG
# -----------------------


def vp_beta(t: torch.Tensor, *, beta_min: float, beta_max: float) -> torch.Tensor:
    return float(beta_min) + t * (float(beta_max) - float(beta_min))


def vp_int_beta(t: torch.Tensor, *, beta_min: float, beta_max: float) -> torch.Tensor:
    return float(beta_min) * t + 0.5 * (float(beta_max) - float(beta_min)) * (t**2)


def vp_mean_coeff(t: torch.Tensor, *, beta_min: float, beta_max: float) -> torch.Tensor:
    return torch.exp(-0.5 * vp_int_beta(t, beta_min=beta_min, beta_max=beta_max))


def vp_marginal_std(t: torch.Tensor, *, beta_min: float, beta_max: float) -> torch.Tensor:
    int_b = vp_int_beta(t, beta_min=beta_min, beta_max=beta_max)
    return torch.sqrt(1.0 - torch.exp(-int_b))


def vp_diffusion(t: torch.Tensor, *, beta_min: float, beta_max: float) -> torch.Tensor:
    return torch.sqrt(vp_beta(t, beta_min=beta_min, beta_max=beta_max))


class EMA:
    """EMA for model parameters (skip non-float buffers safely)."""

    def __init__(self, beta: float = 0.9999):
        self.beta = float(beta)

    @torch.no_grad()
    def update(self, ema_model: nn.Module, model: nn.Module) -> None:
        b = self.beta
        ema_sd = ema_model.state_dict()
        sd = model.state_dict()

        for k in ema_sd.keys():
            dst = ema_sd[k]
            src = sd[k].detach()

            if torch.is_floating_point(dst):
                dst.mul_(b).add_(src.to(dtype=dst.dtype), alpha=(1.0 - b))
            else:
                # e.g. BatchNorm.num_batches_tracked (Long)
                if dst.shape == src.shape and dst.dtype == src.dtype:
                    dst.copy_(src)
                else:
                    ema_sd[k] = src.clone()

        ema_model.load_state_dict(ema_sd, strict=False)


# Conditional 1D Score U-Net (keeps name `UNet` for drop-in compatibility)
class UNet(nn.Module):
    """
    Score-based conditional 1D U-Net.

    Input:
      - x: (B, in_channels, L) where in_channels = conditioning_channels + 1 (noisy target)
      - t: (B,) continuous time in (0, 1]
    Output:
      - score: (B, out_channels, L) (typically out_channels=1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_feat: int = 64,
        time_emb_dim: int = 256,
        sigma: float = 25.0,
    ):
        super().__init__()
        self.sigma = float(sigma)

        # Time embedding: official-style Gaussian Fourier features (+ linear), swish activation.
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoding path (1D)
        self.conv1 = nn.Conv1d(in_channels, n_feat, kernel_size=3, stride=1, padding=1, bias=False)
        self.dense1 = Dense1D(time_emb_dim, n_feat)
        self.gnorm1 = _group_norm(n_feat)

        self.conv2 = nn.Conv1d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.dense2 = Dense1D(time_emb_dim, n_feat * 2)
        self.gnorm2 = _group_norm(n_feat * 2)

        self.conv3 = nn.Conv1d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.dense3 = Dense1D(time_emb_dim, n_feat * 4)
        self.gnorm3 = _group_norm(n_feat * 4)

        self.conv4 = nn.Conv1d(n_feat * 4, n_feat * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.dense4 = Dense1D(time_emb_dim, n_feat * 8)
        self.gnorm4 = _group_norm(n_feat * 8)

        # Decoding path (1D)
        # kernel_size=4, stride=2, padding=1 -> length doubles (common 1D upsample)
        self.tconv4 = nn.ConvTranspose1d(n_feat * 8, n_feat * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.dense5 = Dense1D(time_emb_dim, n_feat * 4)
        self.tgnorm4 = _group_norm(n_feat * 4)

        self.tconv3 = nn.ConvTranspose1d(
            n_feat * 8, n_feat * 2, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.dense6 = Dense1D(time_emb_dim, n_feat * 2)
        self.tgnorm3 = _group_norm(n_feat * 2)

        self.tconv2 = nn.ConvTranspose1d(
            n_feat * 4, n_feat, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.dense7 = Dense1D(time_emb_dim, n_feat)
        self.tgnorm2 = _group_norm(n_feat)

        self.tconv1 = nn.Conv1d(n_feat * 2, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device)
        t = t.to(device=x.device, dtype=torch.float32)

        # Embed time
        emb = _swish(self.embed(t))

        # Encoding
        h1 = self.conv1(x)
        h1 = h1 + self.dense1(emb)
        h1 = _swish(self.gnorm1(h1))

        h2 = self.conv2(h1)
        h2 = h2 + self.dense2(emb)
        h2 = _swish(self.gnorm2(h2))

        h3 = self.conv3(h2)
        h3 = h3 + self.dense3(emb)
        h3 = _swish(self.gnorm3(h3))

        h4 = self.conv4(h3)
        h4 = h4 + self.dense4(emb)
        h4 = _swish(self.gnorm4(h4))

        # Decoding (with skip connections)
        h = self.tconv4(h4)
        h = h + self.dense5(emb)
        h = _swish(self.tgnorm4(h))

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = h + self.dense6(emb)
        h = _swish(self.tgnorm3(h))

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = h + self.dense7(emb)
        h = _swish(self.tgnorm2(h))

        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Output interpreted as score s_theta(x,t)
        return h


# Training function (drop-in signature)
def train_diffusion_model(
    model,
    optimizer,
    scheduler,
    dataloader,
    timesteps,
    betas,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    epochs=100,
    sigma: float = 25.0,
    *,
    eps: float = 1e-5,
    use_ema: bool = True,
    ema_beta: float = 0.9999,
    likelihood_weighting: bool = False,
    grad_clip: float = 1.0,
    sde_type: str = "vp",
    beta_min: float = 0.1,
    beta_max: float = 20.0,
):
    """
    Score matching training for VP-SDE (default) or VE-SDE (continuous time).

    Notes:
    - `betas`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod` are accepted for API compatibility
      but are not used by score-based training.
    - Continuous time t in (eps, 1] is sampled directly: t = eps + (1-eps)*U.
    - Loss (DSM): E[ || s_theta(x_t,t)*std(t) + z ||^2 ]
      Optional weighting (likelihood_weighting) to reduce dominance of extreme noise.
    """
    del betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

    sde_type = str(sde_type).lower().strip()
    if sde_type not in {"vp", "ve"}:
        raise ValueError(f"Unsupported sde_type: {sde_type}")

    # If the model was constructed with a different sigma, prefer that (VE only).
    if hasattr(model, "sigma"):
        sigma = float(getattr(model, "sigma"))

    model.train()
    ema = EMA(beta=float(ema_beta))
    ema_model = None
    if bool(use_ema):
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _, (batch_inputs, batch_targets) in tqdm(
            enumerate(dataloader), total=len(dataloader), leave=False
        ):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_size = batch_inputs.size(0)

            # Continuous time in (eps, 1]
            t = float(eps) + (1.0 - float(eps)) * torch.rand(batch_size, device=device)

            z = torch.randn_like(batch_targets)
            if sde_type == "vp":
                mean_c = vp_mean_coeff(t, beta_min=beta_min, beta_max=beta_max).view(batch_size, 1, 1)
                std = vp_marginal_std(t, beta_min=beta_min, beta_max=beta_max).view(batch_size, 1, 1)
                x_t = mean_c * batch_targets + std * z
            else:
                std = marginal_prob_std(t, sigma).view(batch_size, 1, 1)
                x_t = batch_targets + z * std

            model_input = torch.cat([batch_inputs, x_t], dim=1)

            optimizer.zero_grad()
            score = model(model_input, t)

            # score ≈ -z / std  ->  (score * std + z) ≈ 0
            per_ex = torch.mean((score * std + z) ** 2, dim=(1, 2))  # (B,)

            if bool(likelihood_weighting):
                # Common weighting: g(t)^2
                if sde_type == "vp":
                    g = vp_diffusion(t, beta_min=beta_min, beta_max=beta_max)
                else:
                    g = diffusion_coeff(t, sigma)
                w = (g**2).detach()
                loss = torch.mean(per_ex * w)
            else:
                loss = torch.mean(per_ex)

            loss.backward()
            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

            epoch_loss += float(loss.item())

            if ema_model is not None:
                ema.update(ema_model, model)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
        if scheduler is not None:
            scheduler.step()

    # store EMA model for sampling/metrics
    if ema_model is not None:
        try:
            setattr(model, "ema_model", ema_model)
        except Exception:
            pass


# Sampling function (drop-in signature)
def sample_from_model(
    model,
    conditioning_inputs,
    alphas,
    alphas_cumprod,
    betas,
    sqrt_one_minus_alphas_cumprod,
    shape,
    sigma: float = 25.0,
    eps: float = 1e-5,
    *,
    # NOTE: 20k steps suele ser excesivo para EEG (tiempo + acumulación numérica).
    # 2k es un default defendible; sube si necesitas mejor calidad.
    num_steps: int = 2000,
    snr: float = 0.1,
    n_steps_each: int = 1,
    use_ema: bool = True,
    # "Noise removal" debe aplicarse SOLO al final (último paso), no en cada iteración,
    # de lo contrario anula el término estocástico del predictor.
    noise_removal: bool = True,
    sde_type: str = "vp",
    beta_min: float = 0.1,
    beta_max: float = 20.0,
):
    """
    Predictor–Corrector sampling for VP-SDE (default) or VE-SDE.

    Notes:
    - `alphas`, `alphas_cumprod`, `betas`, `sqrt_one_minus_alphas_cumprod` are accepted for API compatibility.
      We do NOT use them here; sampling is defined by the explicit SDE + num_steps.
    - `conditioning_inputs` must be on the same device as the model.
    """
    del alphas, alphas_cumprod, sqrt_one_minus_alphas_cumprod

    sde_type = str(sde_type).lower().strip()
    if sde_type not in {"vp", "ve"}:
        raise ValueError(f"Unsupported sde_type: {sde_type}")

    # Prefer EMA model for sampling if available.
    if bool(use_ema) and hasattr(model, "ema_model"):
        try:
            model_s = getattr(model, "ema_model")
        except Exception:
            model_s = model
    else:
        model_s = model

    # If the model was constructed with a different sigma, prefer that.
    if hasattr(model_s, "sigma"):
        sigma = float(getattr(model_s, "sigma"))

    model_s.eval()
    batch_size = int(shape[0])
    seq_length = int(shape[2])
    num_steps = int(max(2, num_steps))
    eps = float(eps)
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must satisfy 0 < eps < 1. Got eps={eps}")

    conditioning_inputs = conditioning_inputs.to(device)

    # Initial sample from the prior
    if sde_type == "vp":
        x = torch.randn((batch_size, 1, seq_length), device=device)
    else:
        t_init = torch.ones(batch_size, device=device)
        init_std = marginal_prob_std(t_init, sigma).view(batch_size, 1, 1)
        x = torch.randn((batch_size, 1, seq_length), device=device) * init_std

    # Time grid: integrate from 1 -> eps (reverse time), dt < 0
    t_seq = torch.linspace(1.0, eps, num_steps, device=device)
    dt = float(t_seq[1] - t_seq[0])  # negative

    with torch.no_grad():
        last_idx = int(t_seq.numel()) - 1
        for idx, t_scalar in enumerate(t_seq):
            t = torch.full((batch_size,), float(t_scalar.item()), device=device)

            # Corrector: Langevin dynamics (n_steps_each)
            for _ in range(int(n_steps_each)):
                if sde_type == "vp":
                    g = vp_diffusion(t, beta_min=beta_min, beta_max=beta_max).view(batch_size, 1, 1)
                else:
                    g = diffusion_coeff(t, sigma).view(batch_size, 1, 1)
                model_input = torch.cat([conditioning_inputs, x], dim=1)
                score = model_s(model_input, t)

                noise = torch.randn_like(x)
                # Usar mediana (más robusto a outliers) en vez de promedio
                grad_norms = torch.norm(score.reshape(batch_size, -1), dim=1)
                noise_norms = torch.norm(noise.reshape(batch_size, -1), dim=1)
                grad_norm = float(torch.median(grad_norms).item())
                noise_norm = float(torch.median(noise_norms).item())

                step_size = (float(snr) * noise_norm / (grad_norm + 1e-12)) ** 2 * 2.0
                # Clamp conservador para evitar explosiones en señales
                step_size = float(min(step_size, 0.01))

                x = x + step_size * score + torch.sqrt(torch.tensor(2.0 * step_size, device=device)) * noise

            # Predictor: Euler–Maruyama over reverse-time SDE
            model_input = torch.cat([conditioning_inputs, x], dim=1)
            score = model_s(model_input, t)

            if sde_type == "vp":
                beta_t = vp_beta(t, beta_min=beta_min, beta_max=beta_max).view(batch_size, 1, 1)
                drift = -0.5 * beta_t * x
                diffusion = torch.sqrt(beta_t)
                rev_drift = drift - (diffusion**2) * score
                x_mean = x + rev_drift * dt
                noise = torch.randn_like(x)
                x = x_mean + diffusion * torch.sqrt(torch.tensor(-dt, device=device)) * noise
            else:
                g = diffusion_coeff(t, sigma).view(batch_size, 1, 1)
                # Reverse-time VE-SDE: dx = -g(t)^2 * score dt + g dW
                x_mean = x - (g**2) * score * dt
                noise = torch.randn_like(x)
                x = x_mean + g * torch.sqrt(torch.tensor(-dt, device=device)) * noise

            if bool(noise_removal):
                # Solo al final (último paso) para mantener la trayectoria estocástica
                if idx == last_idx:
                    x = x_mean

    return x
