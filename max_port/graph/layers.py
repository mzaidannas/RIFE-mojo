"""MAX graph builders for RIFE v4.25 sub-modules (P5).

Canonical internal layout is **NCHW** (matches PyTorch, `ops.resize`, and the
grid_sample kernel). `conv2d`/`conv_transpose` transpose to NHWC only for the
conv op itself. Weights come from `max_port/weights/rife_v4_25.safetensors`,
already in MAX filter layout (RSCF), keyed as documented in P3.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from max.graph import ops, TensorValue, Weight, Graph, TensorType
from max.dtype import DType

# Custom-op packages (grid_sample warp + bilinear resize). Pass CUSTOM_EXTENSIONS
# to every Graph that uses these ops. Both run on CPU and Apple GPU; MAX's native
# `ops.resize` is CPU-only on the Apple GPU in 26.5, hence the resize kernel.
_KDIR = Path(__file__).resolve().parents[1] / "kernels"
GRID_SAMPLE_PKG = _KDIR / "grid_sample_pkg"
RESIZE_PKG = _KDIR / "resize_pkg"
CONVT_PKG = _KDIR / "convt_pkg"
CUSTOM_EXTENSIONS = [GRID_SAMPLE_PKG, RESIZE_PKG, CONVT_PKG]


def load_weights(path="max_port/weights/rife_v4_25.safetensors"):
    """Return {key: np.float32 array} for all tensors."""
    from safetensors.numpy import load_file
    return {k: np.ascontiguousarray(v, np.float32) for k, v in load_file(path).items()}


# Weights are added as graph `Weight` placeholders (NOT baked constants) because
# conv2d_transpose fails `num_groups` inference on a constant filter. Pass the
# same dict as `weights_registry=` to `session.load`. `_w` caches per graph so a
# weight referenced twice (e.g. Head applied to both frames) is added once.
_REG: dict = {}
_WCACHE: dict = {}


def set_weights(W: dict):
    """Register the weight dict used by builders and clear the per-graph cache."""
    global _REG
    _REG = W
    _WCACHE.clear()


def _w(name, dev):
    g = Graph.current
    key = (id(g), name)
    if key not in _WCACHE:
        arr = _REG[name]
        _WCACHE[key] = g.add_weight(Weight(name, DType.float32, tuple(arr.shape), dev))
    return _WCACHE[key]


def _dims(x: TensorValue):
    """Return shape dims as ints where static, else symbolic Dim (for symbolic
    H/W graphs). Channel dims are always static so `int()`/`//` on them work."""
    out = []
    for d in x.shape:
        try:
            out.append(int(d))
        except Exception:
            out.append(d)
    return tuple(out)


def leaky(x, slope=0.2):
    return ops.max(x, x * slope)


def conv2d(x, W, prefix, dev, stride, padding):
    """NCHW in/out. Filter already RSCF; bias inline (verified OK for conv2d)."""
    xn = ops.permute(x, [0, 2, 3, 1])                     # NCHW -> NHWC
    filt = _w(prefix + ".weight", dev)
    bias = _w(prefix + ".bias", dev)
    y = ops.conv2d(xn, filt, stride=stride, padding=padding, bias=bias)
    return ops.permute(y, [0, 3, 1, 2])                   # NHWC -> NCHW


def conv_transpose(x, W, prefix, dev):
    """NCHW in/out ConvTranspose2d(k4,s2,p1) via the custom kernel (CPU+GPU).
    Weight is (kh,kw,Cout,Cin); bias added over the NCHW channel axis. Replaces
    ops.conv2d_transpose, which aborts on the Apple GPU (cudnn) in 26.5."""
    N, _, H, Wd = _dims(x)
    filt = _w(prefix + ".weight", dev)
    cout = _REG[prefix + ".weight"].shape[2]        # (kh,kw,Cout,Cin)
    y = ops.custom(
        name="conv_transpose_k4s2p1", device=dev, values=[x, filt],
        out_types=[TensorType(DType.float32, [N, cout, 2 * H, 2 * Wd], dev)],
    )[0].tensor
    b = ops.reshape(_w(prefix + ".bias", dev), [1, cout, 1, 1])
    return y + b


def pixel_shuffle(x, r):
    """NCHW PixelShuffle, C = Co*r*r. Requires static C; H/W may be symbolic ints."""
    N, C, H, Wd = _dims(x)
    Co = C // (r * r)
    y = ops.reshape(x, [N, Co, r, r, H, Wd])
    y = ops.permute(y, [0, 1, 4, 2, 5, 3])
    return ops.reshape(y, [N, Co, H * r, Wd * r])


def resize_bilinear(x, out_h, out_w, dev):
    """NCHW bilinear resize (align_corners=False) via the custom kernel (CPU+GPU;
    ops.resize is CPU-only on Apple GPU in 26.5)."""
    N, C, _, _ = _dims(x)
    return ops.custom(
        name="resize_bilinear_acfalse", device=dev, values=[x],
        out_types=[TensorType(DType.float32, [N, C, out_h, out_w], dev)],
    )[0].tensor


# ------------------------------------------------------------------ Head
def head(x, W, dev):
    """Head/encode: (N,3,H,W) -> (N,4,H,W). cnn0 s2 down, cnn3 convT s2 up."""
    N, _, H, Wd = _dims(x)
    x0 = conv2d(x, W, "head.cnn0", dev, (2, 2), (1, 1, 1, 1))
    x = leaky(x0)
    x1 = conv2d(x, W, "head.cnn1", dev, (1, 1), (1, 1, 1, 1))
    x = leaky(x1)
    x2 = conv2d(x, W, "head.cnn2", dev, (1, 1), (1, 1, 1, 1))
    x = leaky(x2)
    y = conv_transpose(x, W, "head.cnn3", dev)
    # stride-2 down then stride-2 up gives spatial 2*((H+1)//2); for ÷64 (even)
    # inputs that equals H — assert it so symbolic-H graphs keep a clean (N,4,H,W).
    return ops.rebind(y, [N, 4, H, Wd])


# ------------------------------------------------------------------ ResConv
def resconv(x, W, prefix, dev):
    """relu(conv(x) * beta + x), conv 3x3 s1 p1."""
    c = conv2d(x, W, prefix + ".conv", dev, (1, 1), (1, 1, 1, 1))
    beta = _w(prefix + ".beta", dev)
    return leaky(c * beta + x)


# ------------------------------------------------------------------ IFBlock
def ifblock(x, flow, scale, W, block, dev):
    """One IFBlock. x: (N, in-4, H, W) if flow given else (N, in, H, W).
    Returns (flow_out, mask, feat). Mirrors IFBlock.forward."""
    N, _, H, Wd = _dims(x)
    s = int(scale)
    dh, dw = H // s, Wd // s                                  # floor, like F.interpolate; Dim//int ok
    x = resize_bilinear(x, dh, dw, dev)
    if flow is not None:
        fl = resize_bilinear(flow, dh, dw, dev) * (1.0 / scale)
        x = ops.concat([x, fl], axis=1)
    feat = conv2d(x, W, f"{block}.conv0.0", dev, (2, 2), (1, 1, 1, 1))
    feat = leaky(feat)
    feat = conv2d(feat, W, f"{block}.conv0.1", dev, (2, 2), (1, 1, 1, 1))
    feat = leaky(feat)
    for i in range(8):
        feat = resconv(feat, W, f"{block}.convblock.{i}", dev)
    tmp = conv_transpose(feat, W, f"{block}.lastconv", dev)
    tmp = pixel_shuffle(tmp, 2)
    # up back to the block-input size (H, Wd); == tmp_size*scale for ÷64 inputs
    tmp = resize_bilinear(tmp, H, Wd, dev)
    flow_out = ops.slice_tensor(tmp, [slice(None), slice(0, 4)]) * scale
    mask = ops.slice_tensor(tmp, [slice(None), slice(4, 5)])
    feat_out = ops.slice_tensor(tmp, [slice(None), slice(5, None)])
    return flow_out, mask, feat_out
