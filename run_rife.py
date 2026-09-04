#!/usr/bin/env python3
"""Run RIFE v4.25 frame interpolation on the Mojo/MAX backend.

Two modes:

  # interpolate a middle frame between two images
  python run_rife.py frame0.png frame1.png -o middle.png [--timestep 0.5] [--device cpu|gpu]

  # sanity self-test on random frames (no image files needed)
  python run_rife.py --selftest [--device cpu|gpu]

Inference needs only `modular` + `numpy` (+ `pillow` for image I/O). No PyTorch.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from max_port.rife import RIFEModel


def _load_image(path: str) -> np.ndarray:
    """Read an image as (1, 3, H, W) float32 in [0, 1]."""
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]  # HWC -> 1,C,H,W


def _save_image(arr: np.ndarray, path: str) -> None:
    from PIL import Image

    img = np.clip(arr[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="RIFE v4.25 on Mojo/MAX")
    ap.add_argument("frame0", nargs="?", help="first input image")
    ap.add_argument("frame1", nargs="?", help="second input image")
    ap.add_argument("-o", "--out", default="middle.png", help="output path")
    ap.add_argument("--timestep", type=float, default=0.5,
                    help="interpolation point in (0,1); 0.5 = midpoint")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                    help="cpu, or gpu for Apple-Silicon Metal")
    ap.add_argument("--selftest", action="store_true",
                    help="run on random frames instead of image files")
    args = ap.parse_args()

    rife = RIFEModel(device=args.device)

    if args.selftest:
        h, w = 256, 256
        f0 = np.random.rand(1, 3, h, w).astype(np.float32)
        f1 = np.random.rand(1, 3, h, w).astype(np.float32)
        out = rife.interpolate(f0, f1, args.timestep)  # warm up + compile
        t0 = time.perf_counter()
        for _ in range(10):
            out = rife.interpolate(f0, f1, args.timestep)
        dt = (time.perf_counter() - t0) / 10 * 1e3
        print(f"[{args.device}] selftest {h}x{w}: out={out.shape} "
              f"range=[{out.min():.3f},{out.max():.3f}] {dt:.1f} ms/frame")
        return

    if not (args.frame0 and args.frame1):
        ap.error("provide two image paths, or use --selftest")

    f0 = _load_image(args.frame0)
    f1 = _load_image(args.frame1)
    if f0.shape != f1.shape:
        ap.error(f"frames differ in size: {f0.shape} vs {f1.shape}")

    t0 = time.perf_counter()
    out = rife.interpolate(f0, f1, args.timestep)
    dt = (time.perf_counter() - t0) * 1e3
    _save_image(out, args.out)
    print(f"[{args.device}] {args.frame0} + {args.frame1} @t={args.timestep} "
          f"-> {args.out}  ({out.shape[3]}x{out.shape[2]}, {dt:.0f} ms incl. compile)")


if __name__ == "__main__":
    main()
