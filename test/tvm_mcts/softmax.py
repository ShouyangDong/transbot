import tvm
import tvm.testing
from tvm.script import tir as T


@tvm.script.ir_module
class Softmax:
    @T.prim_func
    def main(
        A: T.Buffer((64, 1280), "float32"),
        T_softmax_norm: T.Buffer((64, 1280), "float32"),
    ) -> None:
        T_softmax_maxelem = T.alloc_buffer([64], dtype="float32", scope="local")
        T_softmax_exp = T.alloc_buffer([64, 1280], dtype="float32", scope="local")
        T_softmax_expsum = T.alloc_buffer([64], dtype="float32", scope="local")
        for i0, i1 in T.grid(64, 1280):
            with T.block("T_softmax_maxelem"):
                i0_1, k = T.axis.remap("SR", [i0, i1])
                with T.init():
                    T_softmax_maxelem[i0_1] = T.min_value("float32")
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(64, 1280):
            with T.block("T_softmax_exp"):
                i0_2, i1_1 = T.axis.remap("SS", [i0, i1])
                T_softmax_exp[i0_2, i1_1] = T.exp(
                    A[i0_2, i1_1] - T_softmax_maxelem[i0_2], dtype="float32"
                )
        for i0_3, i1 in T.grid(64, 1280):
            with T.block("T_softmax_expsum"):
                i0_4, k = T.axis.remap("SR", [i0_3, i1])
                with T.init():
                    T_softmax_expsum[i0_4] = T.float32(0)
                T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_exp[i0_4, k]
        for i0_5, i1 in T.grid(64, 1280):
            with T.block("T_softmax_norm"):
                i0_6, i1_2 = T.axis.remap("SS", [i0_5, i1])
                T_softmax_norm[i0_6, i1_2] = (
                    T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]
                )
