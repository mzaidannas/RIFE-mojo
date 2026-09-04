# ===----------------------------------------------------------------------=== #
# Custom MAX op: standard grid_sample (backward warp) matching
#   torch.nn.functional.grid_sample(input, grid,
#       mode='bilinear', padding_mode='border', align_corners=True)
#
# Signature:
#   input: (N, C, H, W)   grid: (N, Ho, Wo, 2)  ->  output: (N, C, Ho, Wo)
#   grid last dim is (x, y) in normalized [-1, 1] coordinates.
#
# align_corners=True unnormalize:  ix = (x+1)/2*(W-1),  iy = (y+1)/2*(H-1)
# border padding: clamp the 4 neighbour indices to [0,W-1] / [0,H-1]
# bilinear: weight the 4 clamped samples.
#
# Runs on both CPU and GPU via `foreach[..., target=target]`.
# ===----------------------------------------------------------------------=== #

import extensibility

from max.gpu.host import DeviceContext
from std.math import floor

from extensibility import InputTensor, OutputTensor, foreach

from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList


@extensibility.register("grid_sample_border")
struct GridSampleBorder:
    @staticmethod
    def execute[
        # "cpu" or "gpu"
        target: StaticString,
    ](
        output: OutputTensor[rank=4, ...],
        input: InputTensor[dtype = output.dtype, rank=4, ...],
        grid: InputTensor[dtype = output.dtype, rank=4, ...],
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

            # Input spatial extents.
            var H = input.shape()[2]
            var W = input.shape()[3]

            # Normalized grid coords: last dim (x, y).
            var gx = rebind[Scalar[dt]](grid[n, ho, wo, 0]).cast[DType.float32]()
            var gy = rebind[Scalar[dt]](grid[n, ho, wo, 1]).cast[DType.float32]()

            # align_corners=True unnormalize.
            var ix = (gx + 1.0) * 0.5 * Float32(W - 1)
            var iy = (gy + 1.0) * 0.5 * Float32(H - 1)

            # Neighbour indices (unclamped) and bilinear weights.
            var x0f = floor(ix)
            var y0f = floor(iy)
            var x0 = Int(x0f)
            var y0 = Int(y0f)
            var x1 = x0 + 1
            var y1 = y0 + 1

            var wx1 = ix - x0f
            var wx0 = 1.0 - wx1
            var wy1 = iy - y0f
            var wy0 = 1.0 - wy1

            # border padding: clamp neighbour indices into range.
            var x0c = max(0, min(x0, W - 1))
            var x1c = max(0, min(x1, W - 1))
            var y0c = max(0, min(y0, H - 1))
            var y1c = max(0, min(y1, H - 1))

            var v00 = rebind[Scalar[dt]](input[n, c, y0c, x0c]).cast[
                DType.float32
            ]()
            var v01 = rebind[Scalar[dt]](input[n, c, y0c, x1c]).cast[
                DType.float32
            ]()
            var v10 = rebind[Scalar[dt]](input[n, c, y1c, x0c]).cast[
                DType.float32
            ]()
            var v11 = rebind[Scalar[dt]](input[n, c, y1c, x1c]).cast[
                DType.float32
            ]()

            var top = wx0 * v00 + wx1 * v01
            var bot = wx0 * v10 + wx1 * v11
            var val = (wy0 * top + wy1 * bot).cast[dt]()

            return SIMD[dt, width](val)

        # simd_width=1: each output element sampled independently (adjacent
        # output pixels map to unrelated input locations, so no vectorization).
        foreach[sample, target=target, simd_width=1](output, ctx)
