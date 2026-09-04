"""Test resize_bilinear_acfalse custom op vs F.interpolate(bilinear, ac=False),
on CPU and Apple GPU."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from max import engine, driver
from max.graph import Graph, TensorType, DeviceRef, ops
from max.dtype import DType
from max.experimental.tensor import Tensor

PKG = Path(__file__).parent / "resize_pkg"
THRESH = 1e-4


def run_on(dev, dref, x, out_hw):
    N, C = x.shape[0], x.shape[1]
    out_shape = [N, C, out_hw[0], out_hw[1]]

    def fwd(xg):
        return ops.custom(name="resize_bilinear_acfalse", device=dref, values=[xg],
                          out_types=[TensorType(DType.float32, out_shape, dref)])[0].tensor

    g = Graph("resize", forward=fwd,
              input_types=[TensorType(DType.float32, x.shape, dref)],
              custom_extensions=[PKG])
    m = engine.InferenceSession(devices=[dev]).load(g)
    o = m.execute(Tensor.from_dlpack(x).to(dev))[0]
    return np.asarray(Tensor.from_dlpack(o).to(driver.CPU()).to_numpy())


CASES = [  # N, C, H, W, Ho, Wo
    (1, 4, 16, 16, 32, 32),   # 2x up
    (1, 8, 32, 32, 8, 8),     # 4x down
    (1, 3, 12, 20, 24, 15),   # non-square, mixed up/down
    (2, 5, 10, 10, 10, 10),   # identity
    (1, 4, 7, 9, 33, 21),     # odd sizes
]

if __name__ == "__main__":
    print("=== resize_bilinear_acfalse vs F.interpolate(bilinear, ac=False) ===")
    have_gpu = driver.accelerator_count() > 0
    ok_all = True
    for (N, C, H, W, Ho, Wo) in CASES:
        rng = np.random.default_rng(N * 100 + H + W + Ho)
        x = np.ascontiguousarray(rng.standard_normal((N, C, H, W)), np.float32)
        ref = F.interpolate(torch.from_numpy(x), size=(Ho, Wo), mode="bilinear",
                            align_corners=False).numpy()
        cpu = run_on(driver.CPU(), DeviceRef.CPU(), x, (Ho, Wo))
        dc = np.abs(cpu - ref).max()
        line = f"[{N}x{C}x{H}x{W}->{Ho}x{Wo}] CPU Δ={dc:.2e} {'P' if dc<THRESH else 'F'}"
        ok_all &= dc < THRESH
        if have_gpu:
            gpu = run_on(driver.Accelerator(), DeviceRef.GPU(), x, (Ho, Wo))
            dg = np.abs(gpu - ref).max()
            line += f"  GPU Δ={dg:.2e} {'P' if dg<THRESH else 'F'}"
            ok_all &= dg < THRESH
        print(line)
    print(f"OVERALL: {'PASS' if ok_all else 'FAIL'}")
