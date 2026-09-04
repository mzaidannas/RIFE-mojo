"""Full RIFE v4.25 IFNet forward as a MAX graph (P6).

Mirrors `train_log.IFNet_HDv3.IFNet.forward` on the inference path
(training=False, fastmode=True, ensemble=False) and `Model.inference`:
5-block coarse-to-fine flow refinement with backward warps, then the final
sigmoid-mask blend. Returns the interpolated frame `merged[-1]`.

Warp uses the P4 `grid_sample_border` custom op; the graph must be built with
`custom_extensions=[KERNELS]`.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from max.graph import ops, TensorType
from max.dtype import DType

import max_port.graph.layers as L

# Both custom-op packages (warp + resize). Pass to Graph(custom_extensions=...).
KERNELS = L.CUSTOM_EXTENSIONS

# scale_list for Model.inference(..., scale=1.0)
SCALE_LIST = [16.0, 8.0, 4.0, 2.0, 1.0]


def _sl(x, a, b):
    return ops.slice_tensor(x, [slice(None), slice(a, b)])


def _identity_grid(N, H, W):
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    ident = np.zeros((N, 2, H, W), dtype=np.float32)
    ident[:, 0, :, :] = xs[None, None, :]
    ident[:, 1, :, :] = ys[None, :, None]
    return ident


def warp(x, flow, ident, norm, dev):
    """Backward warp of x by a 2-ch flow, matching model/warplayer.warp.

    `ident` is the normalized identity grid (1,2,H,W) and `norm` is (1,2,1,1) =
    [2/(W-1), 2/(H-1)] — both host-computed (so this works with symbolic H/W);
    all warps here are at full resolution, so one ident/norm serves them all.
    grid = ident + flow*norm  ->  grid_sample."""
    N, C, Hf, Wf = L._dims(x)
    grid = ops.permute(ident + flow * norm, [0, 2, 3, 1])
    return ops.custom(
        name="grid_sample_border", device=dev, values=[x, grid],
        out_types=[TensorType(DType.float32, [N, C, Hf, Wf], dev)],
    )[0].tensor


def norm_factors(H, W):
    """(1,2,1,1) = [2/(W-1), 2/(H-1)] — flow normalization from warplayer."""
    return np.array([2.0 / (W - 1.0), 2.0 / (H - 1.0)], np.float32).reshape(1, 2, 1, 1)


def ifnet(img0, img1, W, dev, timestep=0.5, ts_value=None,
          ident_value=None, norm_value=None, scale_list=SCALE_LIST):
    """img0, img1: (N,3,H,W). Returns interpolated frame (N,3,H,W).

    Static graphs: pass nothing extra — ts/ident/norm are baked from the static
    shape. Symbolic-H/W graphs: pass `ts_value`, `ident_value` (1,2,H,W) and
    `norm_value` (1,2,1,1) as graph inputs (host-computed per call), since those
    depend on H/W which aren't known at build time.
    """
    N, _, H, Wd = L._dims(img0)
    f0 = L.head(img0, W, dev)
    f1 = L.head(img1, W, dev)
    if ident_value is not None:                      # symbolic path
        ts, ident, norm = ts_value, ident_value, norm_value
    else:                                            # static path: bake constants
        ts = ts_value if ts_value is not None else ops.constant(
            np.full((N, 1, H, Wd), timestep, np.float32), dtype=DType.float32, device=dev)
        ident = ops.constant(_identity_grid(N, H, Wd), dtype=DType.float32, device=dev)
        norm = ops.constant(norm_factors(H, Wd), dtype=DType.float32, device=dev)

    warped0, warped1 = img0, img1
    flow = mask = feat = None
    for i in range(5):
        if flow is None:
            x = ops.concat([img0, img1, f0, f1, ts], axis=1)
            flow, mask, feat = L.ifblock(x, None, scale_list[i], W, "block0", dev)
        else:
            wf0 = warp(f0, _sl(flow, 0, 2), ident, norm, dev)
            wf1 = warp(f1, _sl(flow, 2, 4), ident, norm, dev)
            x = ops.concat([warped0, warped1, wf0, wf1, ts, mask, feat], axis=1)
            fd, mask, feat = L.ifblock(x, flow, scale_list[i], W, f"block{i}", dev)
            flow = flow + fd
        warped0 = warp(img0, _sl(flow, 0, 2), ident, norm, dev)
        warped1 = warp(img1, _sl(flow, 2, 4), ident, norm, dev)

    mask = ops.sigmoid(mask)
    return warped0 * mask + warped1 * (1.0 - mask)
