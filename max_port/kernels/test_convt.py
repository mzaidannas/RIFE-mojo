"""Test conv_transpose_k4s2p1 custom op vs torch ConvTranspose2d(k4,s2,p1),
CPU and Apple GPU."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from max import engine, driver
from max.graph import Graph, TensorType, DeviceRef, ops
from max.dtype import DType
from max.experimental.tensor import Tensor

PKG = Path(__file__).parent / "convt_pkg"
THRESH = 1e-4


def run_on(dev, dref, x, w_khkwcoci, out_shape):
    def fwd(xg, wg):
        return ops.custom(name="conv_transpose_k4s2p1", device=dref, values=[xg, wg],
                          out_types=[TensorType(DType.float32, out_shape, dref)])[0].tensor

    g = Graph("convt", forward=fwd, custom_extensions=[PKG],
              input_types=[TensorType(DType.float32, x.shape, dref),
                           TensorType(DType.float32, w_khkwcoci.shape, dref)])
    m = engine.InferenceSession(devices=[dev]).load(g)
    o = m.execute(Tensor.from_dlpack(x).to(dev), Tensor.from_dlpack(w_khkwcoci).to(dev))[0]
    return np.asarray(Tensor.from_dlpack(o).to(driver.CPU()).to_numpy())


CASES = [  # N, Cin, Cout, H, W
    (1, 8, 4, 8, 8),
    (1, 16, 52, 10, 14),   # RIFE-like (lastconv-ish), non-square
    (2, 3, 6, 7, 9),
]

if __name__ == "__main__":
    print("=== conv_transpose_k4s2p1 vs torch ConvTranspose2d(4,2,1) ===")
    have_gpu = driver.accelerator_count() > 0
    ok_all = True
    for (N, Cin, Cout, H, W) in CASES:
        rng = np.random.default_rng(N * 100 + Cin + H)
        x = np.ascontiguousarray(rng.standard_normal((N, Cin, H, W)), np.float32)
        w_torch = np.ascontiguousarray(rng.standard_normal((Cin, Cout, 4, 4)), np.float32)  # IOHW
        ref = F.conv_transpose2d(torch.from_numpy(x), torch.from_numpy(w_torch),
                                 stride=2, padding=1).numpy()
        w_max = np.ascontiguousarray(np.transpose(w_torch, (2, 3, 1, 0)), np.float32)  # (kh,kw,Cout,Cin)
        out_shape = [N, Cout, 2 * H, 2 * W]
        cpu = run_on(driver.CPU(), DeviceRef.CPU(), x, w_max, out_shape)
        dc = np.abs(cpu - ref).max()
        line = f"[{N}x{Cin}->{Cout} {H}x{W}] out={cpu.shape} CPU Δ={dc:.2e} {'P' if dc<THRESH else 'F'}"
        ok_all &= (cpu.shape == ref.shape and dc < THRESH)
        if have_gpu:
            gpu = run_on(driver.Accelerator(), DeviceRef.GPU(), x, w_max, out_shape)
            dg = np.abs(gpu - ref).max()
            line += f"  GPU Δ={dg:.2e} {'P' if dg<THRESH else 'F'}"
            ok_all &= dg < THRESH
        print(line)
    print(f"OVERALL: {'PASS' if ok_all else 'FAIL'}")
