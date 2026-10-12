"""Patch-wise asymmetric flow for Level-of-Token Diffusion.

The velocity algebra is LakonLab's ``AsymFlowMixin.asymflow_velocity`` with
scale ``s = 1`` and calibration ``k = 1``. That special case is equation 9 of
Nakayama et al. 2026. LoT does not use AsymFlow's per-extent timestep
calibration (eq. 18). Extent scales are applied to the clean latent instead
(eq. 16), which keeps a single timestep for every token.

LakonLab is loaded from the ``common.py`` file directly. Importing the
``lakonlab`` package pulls in ``mmcv``, which this path does not need.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import torch

from layout import LotLayout, TokenRect


def _lakonlab_common_path() -> Path:
    override = os.environ.get("LAKONLAB_ROOT")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    repo = Path(__file__).resolve().parents[2]
    candidates.append(repo.parent / "LakonLab")
    for root in candidates:
        path = root / "lakonlab" / "models" / "architectures" / "asymflow" / "common.py"
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"LakonLab asymflow/common.py not found. Looked in: {searched}")


@lru_cache(maxsize=1)
def _asymflow():
    path = _lakonlab_common_path()
    spec = importlib.util.spec_from_file_location("lakonlab_asymflow_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _VelocityKernel(module.AsymFlowMixin):
        """One Procrustes basis, otherwise the LakonLab velocity step."""

        def __init__(self, basis: torch.Tensor):
            self.proj_buffer = basis
            self.sigma_min = 1e-6
            self.training = True

    return module.AsymFlowCalibration, _VelocityKernel, path


def mean_basis(token_dim: int, extent: tuple[int, int, int]) -> torch.Tensor:
    """Semi-orthonormal lift that averages the uniform tokens inside a patch.

    ``A`` has shape ``(et*eh*ew*D, D)`` and ``A^T A = I``. For a unit extent
    this is the identity, so a dense layout is unchanged. This is the
    stand-in used before a Procrustes fit; it is not the paper's data-fit basis.
    """
    et, eh, ew = extent
    sites = et * eh * ew
    if token_dim < 1 or sites < 1:
        raise ValueError("token dim and extent must be positive")
    eye = torch.eye(token_dim)
    return eye.repeat(sites, 1) / math_sqrt(sites)


def math_sqrt(value: int) -> float:
    return value ** 0.5


def fit_procrustes(dense: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthogonal Procrustes basis and RMS scale, eqs. 14–15.

    ``dense`` is ``(N, D_e)`` and ``reference`` is ``(N, D)``, the aligned
    multi-scale VAE tokens. The SVD matches
    ``LakonLab/tools/asymflow_subspace_procrustes.py``: ``A = V U^T`` from
    ``Z^T X = U S V^T``. ``A`` is ``(D_e, D)`` with ``A^T A = I``, and ``s``
    matches the RMS of ``A^T x`` to the RMS of ``z``.
    """
    if dense.ndim != 2 or reference.ndim != 2 or dense.shape[0] != reference.shape[0]:
        raise ValueError("dense and reference must be (N, D_e) and (N, D)")
    if reference.shape[1] > dense.shape[1]:
        raise ValueError("reference dimension exceeds the dense patch dimension")
    dense64 = dense.detach().double()
    reference64 = reference.detach().double()
    cross = reference64.T @ dense64
    pixel_gram = dense64.T @ dense64
    latent_norm_sq = (reference64 * reference64).sum().clamp(min=1e-12)
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    rank = cross.shape[0]
    basis = (vh[:rank].T @ u[:, :rank].T).contiguous()
    projected = torch.trace(basis.T @ pixel_gram @ basis)
    scale = (projected / latent_norm_sq).clamp(min=1e-12).sqrt()
    return basis.float(), scale.float().reshape(())


def compress_patches(dense: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Eq. 7. ``dense`` is ``(..., D_e)`` and the result is ``(..., D)``."""
    return dense @ basis.to(dtype=dense.dtype, device=dense.device)


def asymmetric_target(clean: torch.Tensor, noise: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Eq. 8, row-vector form: ``P eps - x0`` with ``P = A A^T``."""
    projector = basis @ basis.T
    return noise @ projector.to(dtype=noise.dtype, device=noise.device) - clean


def recover_dense_velocity(
    u_a: torch.Tensor,
    x_t: torch.Tensor,
    basis: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """Recover a full-rank patch velocity from an asymmetric prediction (eq. 9).

    ``u_a`` and ``x_t`` share shape ``(..., D_e)``. ``sigma`` is a scalar or a
    batch vector of length ``u_a.shape[0]``. The denominator is clamped at
    ``1e-6``, as in the paper.
    """
    if u_a.shape != x_t.shape:
        raise ValueError(f"u_a shape {tuple(u_a.shape)} != x_t shape {tuple(x_t.shape)}")
    if basis.ndim != 2 or basis.shape[0] != u_a.shape[-1]:
        raise ValueError("basis must be (D_e, D) for this patch")
    sigma_t = _broadcast_sigma(sigma, u_a)
    basis_f = basis.detach().to(device=u_a.device, dtype=torch.float32)
    if os.environ.get("LOT_NO_LAKONLAB") == "1":
        return asymflow_velocity_torch(u_a, x_t, basis_f, sigma_t).to(dtype=u_a.dtype)
    try:
        calibration_cls, kernel_cls, _path = _asymflow()
    except FileNotFoundError:
        # No LakonLab checkout (for example inside ComfyUI): the same s = k = 1 arithmetic.
        return asymflow_velocity_torch(u_a, x_t, basis_f, sigma_t).to(dtype=u_a.dtype)
    ones = torch.ones((), device=u_a.device, dtype=torch.float32)
    calibration = calibration_cls(s=ones, k=ones, timestep=ones, sigma=sigma_t)
    recovered = kernel_cls(basis_f).asymflow_velocity(u_a, x_t, calibration)
    return recovered.to(dtype=u_a.dtype)


def asymflow_velocity_torch(
    u_a: torch.Tensor,
    x_t: torch.Tensor,
    basis: torch.Tensor,
    sigma: torch.Tensor,
    sigma_min: float = 1e-6,
) -> torch.Tensor:
    """LakonLab ``asymflow_velocity`` with ``s = k = 1``, in plain torch (eq. 9).

    ``sub(v) = v A A^T`` and ``comp(v) = v - sub(v)``; the result is
    ``sub(u_a) + (comp(x_t) + comp(u_a)) / max(sigma, sigma_min)``. Matches the
    kernel's training-mode ``sigma_min``, which is what ``_VelocityKernel`` sets.
    """
    u = u_a.float()
    x = x_t.float()
    projector = basis @ basis.T
    u_sub = u @ projector
    x_sub = x @ projector
    sig = sigma.float().clamp(min=sigma_min)
    return u_sub + ((x - x_sub) + (u - u_sub)) / sig


def _broadcast_sigma(sigma: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), dtype=torch.float32)
    sigma = sigma.to(device=like.device, dtype=torch.float32)
    if sigma.ndim == 0:
        return sigma
    if sigma.ndim == 1:
        if sigma.shape[0] != like.shape[0]:
            raise ValueError(f"sigma batch {sigma.shape[0]} != {like.shape[0]}")
        return sigma.reshape(like.shape[0], *([1] * (like.ndim - 1)))
    if sigma.shape == like.shape[: sigma.ndim]:
        return sigma.reshape(*sigma.shape, *([1] * (like.ndim - sigma.ndim)))
    raise ValueError(f"cannot broadcast sigma shape {tuple(sigma.shape)} onto {tuple(like.shape)}")


def sample_noisy(
    y0: torch.Tensor,
    sigma: torch.Tensor | float,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``y_t = (1 - sigma) y0 + sigma eps`` on the scaled lattice."""
    if noise is None:
        noise = torch.randn_like(y0)
    sig = _broadcast_sigma(sigma, y0).to(dtype=y0.dtype)
    return (1 - sig) * y0 + sig * noise, noise


def clean_from_velocity(
    y_t: torch.Tensor,
    velocity: torch.Tensor,
    sigma: torch.Tensor | float,
) -> torch.Tensor:
    """``y0_hat = y_t - sigma u``."""
    sig = _broadcast_sigma(sigma, y_t).to(dtype=y_t.dtype)
    return y_t - sig * velocity


def lot_clean_loss(
    y0_hat: torch.Tensor,
    y0: torch.Tensor,
    sigma: torch.Tensor | float,
    sigma_floor: float = 0.05,
) -> torch.Tensor:
    """Weighted clean-data loss, eq. 17.

    The paper writes a squared Euclidean norm inside the expectation. This
    returns the mean over elements of that weighted square, which has the
    same minimizer on a fixed grid and does not grow with the lattice size.
    """
    sig = _broadcast_sigma(sigma, y0).clamp(min=sigma_floor)
    weight = 1.0 / sig.square()
    return ((y0_hat - y0).float().square() * weight).mean()



def lot_h3_clean_loss(
    out: torch.Tensor,
    y_t: torch.Tensor,
    y0: torch.Tensor,
    sigma: torch.Tensor | float,
    sigma_floor: float = 0.05,
) -> torch.Tensor:
    """Eq. 17 for H3's head, which predicts ``x0 - eps`` (H3's scheduler: ``x0 = x_t + sigma * v``).

    The clean estimate is ``y_t + sigma * out``, not ``clean_from_velocity`` (that
    is the ``eps - x0`` convention). For ``sigma >= sigma_floor`` this equals
    ``mean((out - (y0 - eps))^2)``: the ordinary H3 velocity MSE, taken in y-space.
    """
    sig = _broadcast_sigma(sigma, y0).to(dtype=torch.float32)
    y0_hat = y_t.float() + sig * out.float()
    weight = 1.0 / sig.clamp(min=sigma_floor).square()
    return ((y0_hat - y0.float()).square() * weight).mean()

def _one_extent(rects: list[TokenRect]) -> tuple[int, int, int]:
    if not rects:
        raise ValueError("expected at least one rectangle")
    extent = rects[0].extent
    for rect in rects[1:]:
        if rect.extent != extent:
            raise ValueError(f"rectangles mix extents {extent} and {rect.extent}")
    return extent


def _windows(rects: list[TokenRect], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast indices ``(n, et, eh, ew)`` for every rectangle of one extent."""
    et, eh, ew = _one_extent(rects)
    t = torch.tensor([rect.t for rect in rects], dtype=torch.long, device=device)
    u = torch.tensor([rect.u for rect in rects], dtype=torch.long, device=device)
    v = torch.tensor([rect.v for rect in rects], dtype=torch.long, device=device)
    dt = torch.arange(et, device=device)
    du = torch.arange(eh, device=device)
    dv = torch.arange(ew, device=device)
    tt = t[:, None, None, None] + dt[None, :, None, None]
    uu = u[:, None, None, None] + du[None, None, :, None]
    vv = v[:, None, None, None] + dv[None, None, None, :]
    return tt, uu, vv


def apply_extent_scales(
    latent: torch.Tensor,
    layout: LotLayout,
    scales: dict[tuple[int, int, int], torch.Tensor],
    *,
    invert: bool = False,
) -> torch.Tensor:
    """Divide each patch by its extent scale. ``invert=True`` multiplies it back.

    One index write per extent. A unit scale hides a swapped multiply, so tests
    use a scale other than 1.
    """
    out = latent.clone()
    buckets: dict[tuple[int, int, int], list[TokenRect]] = {}
    for rect in layout.rects:
        buckets.setdefault(rect.extent, []).append(rect)
    for extent, rects in buckets.items():
        scale = scales[extent].to(device=latent.device, dtype=latent.dtype)
        tt, uu, vv = _windows(rects, latent.device)
        region = out[:, tt, uu, vv, :]
        region = region * scale if invert else region / scale
        out[:, tt, uu, vv, :] = region
    return out


def gather_extent(tokens: torch.Tensor, rects: list[TokenRect]) -> torch.Tensor:
    """Pack patches of one extent to ``(B, n, D_e)``.

    Site order is C-order over ``(et, eh, ew, D)``, with the token channel
    fastest. Procrustes rows must use this same order. One index read for the
    whole extent.
    """
    tt, uu, vv = _windows(rects, tokens.device)
    patches = tokens[:, tt, uu, vv, :]
    return patches.reshape(tokens.shape[0], len(rects), -1)


def scatter_extent(canvas: torch.Tensor, values: torch.Tensor, rects: list[TokenRect]) -> None:
    """Write ``(B, n, D_e)`` patches back onto a fresh ``(B, T, H, W, D)`` canvas."""
    tt, uu, vv = _windows(rects, canvas.device)
    et, eh, ew = _one_extent(rects)
    patches = values.reshape(canvas.shape[0], len(rects), et, eh, ew, canvas.shape[-1])
    canvas[:, tt, uu, vv, :] = patches


def euler_step(y_t: torch.Tensor, velocity: torch.Tensor, t: float, t_next: float) -> torch.Tensor:
    """One explicit step of ``dy/dt = u`` from ``t`` to ``t_next``."""
    return y_t + (float(t_next) - float(t)) * velocity
