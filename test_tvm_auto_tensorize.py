from tvm import meta_schedule as ms
from tvm import te
from tvm.ir import assert_structural_equal
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
    print_sketches,
)
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.tensor_intrin.arm_cpu import DP4A_S8S8S32_INTRIN
from tvm.tir.tensor_intrin.x86 import AVX512_DOT_16x4_INTRIN as AVX512_INTRIN
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN


def test_x86_conv2d_nchwc(intrin=VNNI_INTRIN, target="llvm -mcpu=cascadelake -num-cores=4"):
    @T.prim_func
    def conv2d_nchwc(
        placeholder: T.Buffer((1, 4, 56, 56, 16), "uint8"),
        placeholder_1: T.Buffer((16, 4, 1, 1, 4, 16, 4), "int8"),
        conv2d_NCHWc_int8: T.Buffer((1, 16, 56, 56, 16), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 in T.grid(1, 16, 56, 56, 16, 1, 1, 4, 4, 4):
            with T.block("conv2d_NCHWc_int8"):
                (
                    n,
                    oc_chunk,
                    oh,
                    ow,
                    oc_block,
                    kh,
                    kw,
                    ic_outer,
                    ic_f_inner,
                    ic_s_inner,
                ) = T.axis.remap("SSSSSRRRRR", [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9])
                T.reads(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner],
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner],
                )
                T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                with T.init():
                    conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = 0
                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[
                    n, oc_chunk, oh, ow, oc_block
                ] + T.cast(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], "int32"
                ) * T.cast(
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner],
                    "int32",
                )
    mod = conv2d_nchwc
    actual = generate_design_space(
        kind="llvm",
        mod=mod,
        target=Target(target),
        types=None,
        sch_rules=[
            ms.schedule_rule.MultiLevelTilingWithIntrin(
                intrin,
                structure="SSRSRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=ms.schedule_rule.ReuseType(req="may", levels=[1, 2], scope="global"),
            ),
        ],
    )