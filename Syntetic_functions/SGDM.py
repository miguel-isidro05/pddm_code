"""
Synthetic EEG Channel Generation using Score-Based Generative Modeling (SGM / ScoreNet)

Drop-in replacement for `ddpm.py`:
- Same conditional setup: model input is `torch.cat([conditioning_inputs, x_t], dim=1)`
- Same public function signatures: `train_diffusion_model(...)` and `sample_from_model(...)`

This version follows the "official" ScoreNet-style ideas:
- Continuous time embedding via Gaussian random Fourier features
- Score matching loss for the VE SDE
- Euler–Maruyama reverse-time sampling (predictor)
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Set device (works on macOS as well)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Keep for API-compatibility with ddpm.py (even if unused here).
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

        # Normalize output (official trick): score / std(t)
        std = marginal_prob_std(t, self.sigma).view(-1, 1, 1).to(x.device)
        h = h / std
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
):
    """
    Score matching training for VE SDE using the same outer-loop structure as ddpm.py.

    Notes:
    - `betas`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod` are accepted for API compatibility
      but are not used by score-based training.
    - Discrete `t` in [0..timesteps-1] is mapped to continuous time in (0, 1].
    """
    del betas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

    # If the model was constructed with a different sigma, prefer that.
    if hasattr(model, "sigma"):
        sigma = float(getattr(model, "sigma"))

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _, (batch_inputs, batch_targets) in tqdm(
            enumerate(dataloader), total=len(dataloader), leave=False
        ):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_size = batch_inputs.size(0)

            # Discrete index -> continuous time in (0, 1]
            t_idx = torch.randint(0, timesteps, (batch_size,), device=device).long()
            t = (t_idx.float() + 1.0) / float(timesteps)

            z = torch.randn_like(batch_targets)
            std = marginal_prob_std(t, sigma).view(batch_size, 1, 1)
            x_t = batch_targets + z * std

            model_input = torch.cat([batch_inputs, x_t], dim=1)

            optimizer.zero_grad()
            score = model(model_input, t)

            # score ≈ -z / std  ->  (score * std + z) ≈ 0
            loss = torch.mean((score * std + z) ** 2)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
        if scheduler is not None:
            scheduler.step()


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
    eps: float = 1e-3,
):
    """
    Reverse-time sampling for VE SDE using Euler–Maruyama (predictor only).

    Notes:
    - `alphas`, `alphas_cumprod`, `betas`, `sqrt_one_minus_alphas_cumprod` are accepted for API compatibility.
      We only use `betas` to infer the number of discretization steps (len(betas)).
    - `conditioning_inputs` must be on the same device as the model.
    """
    del alphas, alphas_cumprod, sqrt_one_minus_alphas_cumprod

    # If the model was constructed with a different sigma, prefer that.
    if hasattr(model, "sigma"):
        sigma = float(getattr(model, "sigma"))

    model.eval()
    batch_size = shape[0]
    seq_length = shape[2]
    num_steps = int(len(betas)) if betas is not None else 1000
    num_steps = max(num_steps, 2)

    conditioning_inputs = conditioning_inputs.to(device)

    # Initial sample ~ N(0, std(t=1)^2)
    t_init = torch.ones(batch_size, device=device)
    init_std = marginal_prob_std(t_init, sigma).view(batch_size, 1, 1)
    x = torch.randn((batch_size, 1, seq_length), device=device) * init_std

    with torch.no_grad():
        # Time grid: t in [1, eps]
        step_size = (1.0 - eps) / float(num_steps - 1)
        sqrt_step = torch.sqrt(torch.tensor(step_size, device=device))
        for i in reversed(range(num_steps)):
            t_scalar = eps + i * step_size
            t = torch.full((batch_size,), float(t_scalar), device=device)

            g = diffusion_coeff(t, sigma).view(batch_size, 1, 1)
            model_input = torch.cat([conditioning_inputs, x], dim=1)
            score = model(model_input, t)

            # Euler–Maruyama with dt < 0 (reverse time)
            dt = -step_size
            x_mean = x + (g**2) * score * dt

            if i > 0:
                noise = torch.randn_like(x)
                x = x_mean + g * sqrt_step * noise
            else:
                x = x_mean

    return x
