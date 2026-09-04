"""P4 test: custom `grid_sample_border` MAX op vs torch.nn.functional.grid_sample.

Replicates:
    F.grid_sample(input, grid, mode='bilinear', padding_mode='border',
                  align_corners=True)

Kernel signature (pure standard grid_sample, semantics baked in):
    input: (N, C, H, W)   grid: (N, Ho, Wo, 2)  ->  output: (N, C, Ho, Wo)
    grid last dim = (x, y) in normalized [-1, 1].

Runs on CPU and (if available) the Apple GPU (Metal). Includes grid coords
outside [-1, 1] to exercise border clamping. PASS threshold: max|Δ| < 1e-4.

Custom-op registration mechanism (MAX 26.5):
    Mojo: `@extensibility.register("grid_sample_border")` on a struct with a
          `@staticmethod def execute[target: StaticString](output, *inputs, ctx)`
          body using `foreach[fn, target=target, simd_width=1](output, ctx)`.
    Python: `ops.custom(name="grid_sample_border", device=..., values=[...],
             out_types=[TensorType(...)])`, with the kernel package directory
             passed via `Graph(..., custom_extensions=[<pkg dir>])`.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from max import engine, driver
from max.graph import Graph, TensorType, DeviceRef, ops
from max.dtype import DType
from max.experimental.tensor import Tensor  # 26.5: moved from max.driver.Tensor

KERNELS = Path(__file__).parent / "grid_sample_pkg"
THRESH = 1e-4


def torch_ref(x_np, grid_np):
    x = torch.from_numpy(x_np)
    g = torch.from_numpy(grid_np)
    y = F.grid_sample(x, g, mode="bilinear", padding_mode="border",
                      align_corners=True)
    return y.numpy()


def build_graph(x_shape, grid_shape, out_shape, dref):
    def fwd(xg, gg):
        return ops.custom(
            name="grid_sample_border",
            device=dref,
            values=[xg, gg],
            out_types=[TensorType(DType.float32, out_shape, dref)],
        )[0].tensor

    return Graph(
        "grid_sample_border",
        forward=fwd,
        input_types=[
            TensorType(DType.float32, x_shape, dref),
            TensorType(DType.float32, grid_shape, dref),
        ],
        custom_extensions=[KERNELS],
    )


def run_on(dev, dref, x_np, grid_np, out_shape):
    sess = engine.InferenceSession(devices=[dev])
    g = build_graph(x_np.shape, grid_np.shape, out_shape, dref)
    model = sess.load(g)
    xt = Tensor.from_dlpack(x_np).to(dev)
    gt = Tensor.from_dlpack(grid_np).to(dev)
    out = model.execute(xt, gt)[0]
    return np.asarray(Tensor.from_dlpack(out).to(driver.CPU()).to_numpy())


def make_case(N, C, H, W, Ho, Wo, grid_lo, grid_hi, seed):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.standard_normal((N, C, H, W)), dtype=np.float32)
    grid = np.ascontiguousarray(
        rng.uniform(grid_lo, grid_hi, size=(N, Ho, Wo, 2)), dtype=np.float32)
    return x, grid


CASES = [
    # name,               N, C,  H,  W, Ho, Wo, grid range (exercise border)
    ("C2 square in-range", 1, 2, 16, 16, 16, 16, (-1.0, 1.0)),
    ("C4 nonsquare",       1, 4, 12, 20, 18, 10, (-1.0, 1.0)),
    ("C16 out-of-bounds",  1, 16, 24, 16, 20, 22, (-1.6, 1.6)),
    ("N2 batch OOB",       2, 3, 10, 14, 14, 10, (-1.4, 1.4)),
]


def build_grid_from_flow(flow_np):
    """Reproduce model/warplayer.warp's grid construction (host-side).

    identity = linspace(-1,1,W) / linspace(-1,1,H); flow normalized by
    (W-1)/2, (H-1)/2; grid = (identity + norm_flow).permute(0,2,3,1) -> (N,H,W,2).
    """
    N, _, H, W = flow_np.shape
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    ident = np.zeros((N, 2, H, W), dtype=np.float32)
    ident[:, 0, :, :] = xs[None, None, :]
    ident[:, 1, :, :] = ys[None, :, None]
    norm = np.empty_like(flow_np)
    norm[:, 0] = flow_np[:, 0] / ((W - 1.0) / 2.0)
    norm[:, 1] = flow_np[:, 1] / ((H - 1.0) / 2.0)
    grid = np.transpose(ident + norm, (0, 2, 3, 1))  # (N,H,W,2)
    return np.ascontiguousarray(grid, dtype=np.float32)


def test_warp_equivalence():
    """Full RIFE warp: custom op (grid built host-side) vs warplayer.warp."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from model.warplayer import warp

    rng = np.random.default_rng(7)
    N, C, H, W = 1, 4, 24, 32
    x = np.ascontiguousarray(rng.standard_normal((N, C, H, W)), dtype=np.float32)
    flow = np.ascontiguousarray(
        rng.standard_normal((N, 2, H, W)).astype(np.float32) * 4.0)

    ref = warp(torch.from_numpy(x), torch.from_numpy(flow)).numpy()
    grid = build_grid_from_flow(flow)
    got = run_on(driver.CPU(), DeviceRef.CPU(), x, grid, [N, C, H, W])
    d = np.abs(got - ref).max()
    ok = d < THRESH
    print(f"[warp-equiv vs warplayer] shape={ref.shape}  CPU max|Δ|={d:.3e} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=== grid_sample_border custom op vs torch (mode=bilinear, "
          "padding=border, align_corners=True) ===")
    dref_cpu = DeviceRef.CPU()
    dev_cpu = driver.CPU()

    have_gpu = driver.accelerator_count() > 0
    print(f"accelerator_count = {driver.accelerator_count()}\n")

    overall_ok = True
    for name, N, C, H, W, Ho, Wo, (lo, hi) in CASES:
        x, grid = make_case(N, C, H, W, Ho, Wo, lo, hi, seed=hash(name) & 0xFFFF)
        out_shape = [N, C, Ho, Wo]
        ref = torch_ref(x, grid)

        # CPU
        cpu = run_on(dev_cpu, dref_cpu, x, grid, out_shape)
        d_cpu = np.abs(cpu - ref).max()
        ok_cpu = d_cpu < THRESH
        overall_ok &= ok_cpu
        line = f"[{name:22}] shape={tuple(out_shape)}  CPU max|Δ|={d_cpu:.3e} " \
               f"{'PASS' if ok_cpu else 'FAIL'}"

        # GPU
        if have_gpu:
            try:
                gpu = run_on(driver.Accelerator(), DeviceRef.GPU(), x, grid,
                             out_shape)
                d_gpu = np.abs(gpu - ref).max()
                ok_gpu = d_gpu < THRESH
                overall_ok &= ok_gpu
                line += f"   GPU max|Δ|={d_gpu:.3e} {'PASS' if ok_gpu else 'FAIL'}"
            except Exception as e:
                line += f"   GPU ERROR: {type(e).__name__}: {str(e)[:120]}"
                overall_ok = False
        print(line)

    print()
    overall_ok &= test_warp_equivalence()

    print(f"\nOVERALL: {'PASS' if overall_ok else 'FAIL'}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
