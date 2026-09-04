# ===----------------------------------------------------------------------=== #
# Custom MAX op: ConvTranspose2d with kernel=4, stride=2, padding=1 (all of
# RIFE's transpose convs use these), matching torch.nn.ConvTranspose2d.
#
# Signature (NCHW input, no bias — bias is added in the graph):
#   input:  (N, Cin, H, W)
#   weight: (kh=4, kw=4, Cout, Cin)   [torch IOHW.permute(2,3,1,0)]
#   -> output: (N, Cout, 2H, 2W)
#
# Transpose-conv relation (s=2, p=1):
#   out[n,co,oy,ox] = sum_{ci,ky,kx} in[n,ci,(oy+1-ky)/2,(ox+1-kx)/2]
#                                    * weight[ky,kx,co,ci]
#   summed over taps where (oy+1-ky) and (ox+1-kx) are >=0, even, and in range.
#
# Native ops.conv2d_transpose has no Apple-GPU kernel as of MAX 26.6 (GPU path
# is cuDNN/NVIDIA-only, GEX-2043); this kernel runs on CPU and GPU via foreach.
# ===----------------------------------------------------------------------=== #

import extensibility

from max.gpu.host import DeviceContext

from extensibility import InputTensor, OutputTensor, foreach

from std.utils.coord import Coord, coord_to_index_list


@extensibility.register("conv_transpose_k4s2p1")
struct ConvTransposeK4S2P1:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=4, ...],
        input: InputTensor[dtype = output.dtype, rank=4, ...],
        weight: InputTensor[dtype = output.dtype, rank=4, ...],
        ctx: DeviceContext,
    ) raises:
        comptime dt = output.dtype

        @parameter
        @always_inline
        def compute[width: Int](out_idx: Coord) -> SIMD[dt, width]:
            var idx = coord_to_index_list(out_idx)
            var n = idx[0]
            var co = idx[1]
            var oy = idx[2]
            var ox = idx[3]

            var H = input.shape()[2]
            var W = input.shape()[3]
            var Cin = input.shape()[1]

            var acc = Float32(0.0)
            for ky in range(4):
                var ty = oy + 1 - ky
                if ty < 0 or (ty & 1) == 1:
                    continue
                var iy = ty >> 1
                if iy >= H:
                    continue
                for kx in range(4):
                    var tx = ox + 1 - kx
                    if tx < 0 or (tx & 1) == 1:
                        continue
                    var ix = tx >> 1
                    if ix >= W:
                        continue
                    for ci in range(Cin):
                        var iv = rebind[Scalar[dt]](input[n, ci, iy, ix]).cast[
                            DType.float32
                        ]()
                        var wv = rebind[Scalar[dt]](weight[ky, kx, co, ci]).cast[
                            DType.float32
                        ]()
                        acc += iv * wv

            return SIMD[dt, width](acc.cast[dt]())

        foreach[compute, target=target, simd_width=1](output, ctx)
