# RIFE-mojo

RIFE **v4.25** video frame interpolation running on **Modular MAX** with custom
**Mojo** GPU/CPU kernels — a from-scratch port of the PyTorch RIFE (IFNet) model
to the MAX Graph API, targeting **Apple-Silicon GPU (Metal)** and CPU.

Given two frames it synthesises an intermediate frame at any timestep `t ∈ (0,1)`.

```
frame0.png ─┐
            ├─►  RIFE (MAX Graph + Mojo kernels)  ─►  middle.png
frame1.png ─┘
```

## Why a port (and why custom kernels)

The model is pure MAX Graph ops **except** three primitives that MAX has no
Apple-GPU kernel for. Each is implemented here as a Mojo custom op that runs on
both CPU and Metal via `foreach[..., target=target]` (verified against PyTorch,
`max|Δ| ≤ ~1e-5`):

| Kernel | File | Why it's custom |
|---|---|---|
| `grid_sample_border` | `max_port/kernels/grid_sample_pkg/` | No native `grid_sample` op exists in MAX (backward warp). |
| `resize_bilinear_acfalse` | `max_port/kernels/resize_pkg/` | Native `resize_linear` is still CPU-only (host ping-pong buffers) as of MAX 26.6. |
| `conv_transpose_k4s2p1` | `max_port/kernels/convt_pkg/` | Native `conv2d_transpose` GPU path is cuDNN/NVIDIA-only (GEX-2043). |

Everything else (Head, 5× IFBlock coarse-to-fine flow, ResConv, warp, sigmoid
mask blend) is built from stock `max.graph.ops`. See `max_port/graph/`.

## Requirements

- **Apple Silicon** (M1–M5) for the GPU path, or any CPU.
- Python **3.13**, [`uv`](https://docs.astral.sh/uv/).
- Modular MAX / Mojo **≥ 26.6.0.dev2026081105 / 1.1.0.dev** (pulled from the
  Modular nightly index, wired up in `pyproject.toml`).

Inference depends only on `modular`, `numpy`, and `pillow` — **no PyTorch**.

## Install & run

```bash
uv sync                       # installs MAX + numpy + pillow

# interpolate the midpoint between two frames
uv run python run_rife.py frame0.png frame1.png -o middle.png

# quarter-point, on the Apple-Silicon GPU
uv run python run_rife.py frame0.png frame1.png -o q.png --timestep 0.25 --device gpu

# no images handy? random-frame self-test + timing
uv run python run_rife.py --selftest --device gpu
```

Input frames may be any size; they are edge-padded to a multiple of 64
internally (RIFE requirement) and cropped back. The first call compiles the
graph (and the Mojo kernels) for that resolution and caches it.

### Programmatic use

```python
import numpy as np
from max_port.rife import RIFEModel

rife = RIFEModel(device="gpu")            # or "cpu"
# frames: (1, 3, H, W) float32 in [0, 1]
mid = rife.interpolate(frame0, frame1, timestep=0.5)
```

## Tests

The kernel parity tests compare each Mojo kernel against PyTorch (dev-only dep):

```bash
uv sync --group dev
uv run python -m max_port.kernels.test_grid_sample
uv run python -m max_port.kernels.test_resize
uv run python -m max_port.kernels.test_convt
```

Each reports CPU **and** GPU error vs the reference.

## Performance

Reference numbers at 256×256, cached graph, on Apple Silicon:

| Device | ms / frame |
|---|---|
| CPU | ~42 |
| GPU (Metal) | ~216 |

> The GPU path is currently slower: the three custom kernels are naive
> `simd_width=1` scalar implementations written correctness-first. Vectorising /
> tiling them is the main open optimisation.

## Weights

`max_port/weights/rife_v4_25.safetensors` (5.66 M params, ~22 MB) was extracted
from the released RIFE v4.25 `flownet.pkl` and re-laid-out for MAX Graph
(conv = RSCF, convT = (kh,kw,Cout,Cin)). See `max_port/weights/README.md`.

## Layout

```
max_port/
  rife.py                 # RIFEModel: pad → run → crop, graph cache
  graph/
    layers.py             # NCHW op builders (conv, warp, resize, convT…)
    ifnet.py              # full IFNet assembled from the layers
  kernels/                # 3 Mojo custom-op packages + parity tests
  weights/                # rife_v4_25.safetensors
run_rife.py               # CLI entry point
```

## Credits

RIFE (Real-Time Intermediate Flow Estimation) — Huang et al.,
[Practical-RIFE](https://github.com/hzwer/Practical-RIFE). This repo ports the
v4.25 IFNet to Modular MAX + Mojo.
