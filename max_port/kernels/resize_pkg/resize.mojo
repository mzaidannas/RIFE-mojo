# ===----------------------------------------------------------------------=== #
# Custom MAX op: bilinear resize matching
#   torch.nn.functional.interpolate(x, size=(Ho,Wo), mode='bilinear',
#       align_corners=False)
#
# Signature (NCHW):
#   input: (N, C, H, W)  ->  output: (N, C, Ho, Wo)   (Ho/Wo from output shape)
#
# align_corners=False source mapping (per spatial axis):
#   scale = in/out;  src = scale*(dst+0.5) - 0.5;  if src < 0: src = 0
#   lo = floor(src) (clamped to in-1);  hi = min(lo+1, in-1);  blend by frac.
#
# Native `resize_linear` is still CPU-only (host List ping-pong buffers) as of
# MAX 26.6; this kernel runs on both CPU and GPU via `foreach[..., target=target]`,
# unblocking full-GPU RIFE.
# ===----------------------------------------------------------------------=== #

import extensibility

from max.gpu.host import DeviceContext
from std.math import floor

from extensibility import InputTensor, OutputTensor, foreach

from std.utils.coord import Coord, coord_to_index_list


@extensibility.register("resize_bilinear_acfalse")
struct ResizeBilinearAcFalse:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=4, ...],
        input: InputTensor[dtype = output.dtype, rank=4, ...],
        ctx: DeviceContext,
    ) raises:
        comptime dt = output.dtype

        @parameter
        @always_inline
        def sample[width: Int](out_idx: Coord) -> SIMD[dt, width]:
            var idx = coord_to_index_list(out_idx)
            var n = idx[0]
            var c = idx[1]
            var ho = idx[2]
            var wo = idx[3]

            var H = input.shape()[2]
            var W = input.shape()[3]
            var Ho = output.shape()[2]
            var Wo = output.shape()[3]

            var scale_h = Float32(H) / Float32(Ho)
            var scale_w = Float32(W) / Float32(Wo)

            var ry = scale_h * (Float32(ho) + 0.5) - 0.5
            var rx = scale_w * (Float32(wo) + 0.5) - 0.5
            if ry < 0.0:
                ry = 0.0
            if rx < 0.0:
                rx = 0.0

            var y0f = floor(ry)
            var x0f = floor(rx)
            var y0 = min(Int(y0f), H - 1)
            var x0 = min(Int(x0f), W - 1)
            var y1 = min(y0 + 1, H - 1)
            var x1 = min(x0 + 1, W - 1)

            var wy1 = ry - y0f
            var wy0 = 1.0 - wy1
            var wx1 = rx - x0f
            var wx0 = 1.0 - wx1

            var v00 = rebind[Scalar[dt]](input[n, c, y0, x0]).cast[DType.float32]()
            var v01 = rebind[Scalar[dt]](input[n, c, y0, x1]).cast[DType.float32]()
            var v10 = rebind[Scalar[dt]](input[n, c, y1, x0]).cast[DType.float32]()
            var v11 = rebind[Scalar[dt]](input[n, c, y1, x1]).cast[DType.float32]()

            var top = wx0 * v00 + wx1 * v01
            var bot = wx0 * v10 + wx1 * v11
            var val = (wy0 * top + wy1 * bot).cast[dt]()

            return SIMD[dt, width](val)

        foreach[sample, target=target, simd_width=1](output, ctx)
