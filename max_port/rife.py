"""RIFE v4.25 running on MAX — production wrapper (P8).

Loads the extracted weights once, compiles one graph per (padded) resolution and
caches it, pads inputs up to a multiple of 64 (RIFE's requirement) and crops the
result back, and feeds timestep at runtime so a single compiled graph serves any
timestep. Runs on CPU or the Apple GPU.

    rife = RIFEModel(device="gpu")
    mid = rife.interpolate(frame0, frame1, timestep=0.5)   # (1,3,H,W) float32 [0,1]
"""
from __future__ import annotations
import numpy as np
from max import engine, driver
from max.graph import Graph, TensorType, DeviceRef
from max.dtype import DType
from max.experimental.tensor import Tensor

import max_port.graph.layers as L
import max_port.graph.ifnet as IF

_ALIGN = 64  # RIFE requires H,W divisible by 64


def _pad_to(x, mult=_ALIGN):
    """Pad (1,3,H,W) on bottom/right up to a multiple of `mult` (edge mode)."""
    _, _, h, w = x.shape
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="edge")
    return np.ascontiguousarray(x, np.float32), h, w


class RIFEModel:
    def __init__(self, weights_path="max_port/weights/rife_v4_25.safetensors",
                 device="cpu", symbolic=True):
        self.W = L.load_weights(weights_path)
        if device == "gpu":
            if driver.accelerator_count() == 0:
                raise RuntimeError("no accelerator available")
            self.dev, self.dref = driver.Accelerator(), DeviceRef.GPU()
        else:
            self.dev, self.dref = driver.CPU(), DeviceRef.CPU()
        self.sess = engine.InferenceSession(devices=[self.dev])
        # symbolic=True: one graph for any ÷64 size. False: one graph per size.
        self.symbolic = symbolic
        self._cache: dict = {}

    def _input_types(self, H, W):
        d = self.dref
        return [
            TensorType(DType.float32, [1, 3, H, W], d),   # img0
            TensorType(DType.float32, [1, 3, H, W], d),   # img1
            TensorType(DType.float32, [1, 1, H, W], d),   # timestep
            TensorType(DType.float32, [1, 2, H, W], d),   # identity grid
            TensorType(DType.float32, [1, 2, 1, 1], d),   # flow-norm factors
        ]

    def _model(self, H, W):
        """Compile (or fetch cached) the IFNet graph. Symbolic mode builds one
        graph with dynamic H/W; otherwise one per (H, W)."""
        key = "sym" if self.symbolic else (H, W)
        if key not in self._cache:
            dref = self.dref
            sh = ("H", "W") if self.symbolic else (H, W)
            in_types = self._input_types(*sh)

            def forward(img0, img1, ts, ident, norm):
                L.set_weights(self.W)
                return IF.ifnet(img0, img1, self.W, dref,
                                ts_value=ts, ident_value=ident, norm_value=norm)

            g = Graph("rife_ifnet", forward=forward, input_types=in_types,
                      custom_extensions=IF.KERNELS)
            self._cache[key] = self.sess.load(g, weights_registry=self.W)
        return self._cache[key]

    def interpolate(self, frame0, frame1, timestep=0.5):
        """frame0, frame1: (1,3,H,W) float32 in [0,1]. Returns the same shape."""
        f0, h, w = _pad_to(np.asarray(frame0, np.float32))
        f1, _, _ = _pad_to(np.asarray(frame1, np.float32))
        H, Wp = f0.shape[2], f0.shape[3]
        ts = np.full((1, 1, H, Wp), float(timestep), np.float32)
        ident = IF._identity_grid(1, H, Wp)
        norm = IF.norm_factors(H, Wp)
        model = self._model(H, Wp)
        arrays = [f0, f1, ts, ident, norm]
        out = model.execute(*[Tensor.from_dlpack(a).to(self.dev) for a in arrays])[0]
        out = np.asarray(Tensor.from_dlpack(out).to(driver.CPU()).to_numpy())
        return out[:, :, :h, :w]
