"""Linear filtering in NumPy: convolution, Gaussian blur, Sobel gradients.

Written without OpenCV on purpose (Phase 0 rule: implement it, then compare).
tests/test_filters.py checks every function against cv2 / scipy.

Conventions
  * images are float arrays, (H, W) or (H, W, C); uint8 inputs are converted.
  * `convolve2d` is true convolution (the kernel is flipped). Correlation is
    what most "conv" layers and cv2.filter2D actually compute; the difference
    only matters for asymmetric kernels such as Sobel, and it is the classic
    sign bug in hand-written gradient code.
  * borders are handled by padding: "reflect" (…c b | a b c | b a…, the
    OpenCV default BORDER_REFLECT_101), "edge" or "constant" (zeros).
"""
from __future__ import annotations

import numpy as np

_PAD_MODES = {"reflect": "reflect", "edge": "edge", "constant": "constant"}


def _as_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) if img.dtype != np.float64 else img


def correlate2d(img: np.ndarray, kernel: np.ndarray, border: str = "reflect") -> np.ndarray:
    """out[y, x] = sum_{i,j} kernel[i, j] * img[y + i - kh//2, x + j - kw//2]

    Vectorised with a sliding-window view: for a k×k kernel this does k² whole-
    image multiply-adds instead of H·W small dot products, which is the same
    trick im2col uses, without materialising the (H·W, k²) matrix.
    """
    img = _as_float(img)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel must be 2-D with odd sides")
    if img.ndim == 3:
        return np.stack([correlate2d(img[..., c], kernel, border) for c in range(img.shape[2])],
                        axis=-1)
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    padded = np.pad(img, ((py, py), (px, px)), mode=_PAD_MODES[border])
    out = np.zeros_like(img)
    h, w = img.shape
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j] != 0:
                out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


def convolve2d(img: np.ndarray, kernel: np.ndarray, border: str = "reflect") -> np.ndarray:
    """True 2-D convolution: correlation with the kernel rotated by 180°."""
    return correlate2d(img, np.asarray(kernel)[::-1, ::-1], border)


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """Normalised 1-D Gaussian. Default radius 3σ keeps >99.7% of the mass."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = int(np.ceil(3 * sigma)) if radius is None else radius
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_kernel2d(sigma: float, radius: int | None = None) -> np.ndarray:
    k = gaussian_kernel1d(sigma, radius)
    return np.outer(k, k)


def gaussian_blur(img: np.ndarray, sigma: float, border: str = "reflect") -> np.ndarray:
    """Separable Gaussian blur: a row pass then a column pass.

    A 2-D Gaussian is the outer product of two 1-D Gaussians, so two passes of
    length k cost 2k per pixel instead of k². At σ=3 (k=19) that is 38 vs 361.
    """
    k = gaussian_kernel1d(sigma)
    tmp = correlate2d(img, k[None, :], border)
    return correlate2d(tmp, k[:, None], border)


# Sobel = smoothing [1, 2, 1] in one direction x central difference [-1, 0, 1]
# in the other. Written as correlation kernels (d/dx is positive left-to-right).
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T.copy()


def to_gray(img: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma (the weights OpenCV's RGB2GRAY uses)."""
    img = _as_float(img)
    if img.ndim == 2:
        return img
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def sobel(img: np.ndarray, border: str = "reflect") -> tuple[np.ndarray, np.ndarray]:
    """(gx, gy) image gradients of the grayscale image."""
    gray = to_gray(img)
    return correlate2d(gray, SOBEL_X, border), correlate2d(gray, SOBEL_Y, border)


def gradient_magnitude(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(magnitude, orientation in radians) from Sobel gradients."""
    gx, gy = sobel(img)
    return np.hypot(gx, gy), np.arctan2(gy, gx)
