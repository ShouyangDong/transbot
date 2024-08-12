import logging
import tempfile

import numpy as np

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.target import Target

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@tvm.script.ir_module
class Softmax:
    @T.prim_func
    def main(
        A: T.Buffer((64, 1280), "float32"),
        T_softmax_norm: T.Buffer((64, 1280), "float32"),
    ) -> None:
        T_softmax_maxelem = T.alloc_buffer([64], dtype="float32")
        T_softmax_exp = T.alloc_buffer([64, 1280], dtype="float32")
        T_softmax_expsum = T.alloc_buffer([64], dtype="float32")
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


def test_tune_softmax_cuda():
    rules = ms.ScheduleRule.create("cuda")
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/nvidia-a100")
        database = ms.tir_integration.tune_tir(
            mod=Softmax,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
            space=ms.space_generator.PostOrderApply(
                sch_rules=rules,
                postprocs=[],
                mutator_probs={},
            ),
        )
        sch = ms.tir_integration.compile_tir(database, Softmax, target)
        assert sch is not None
        sch.mod.show()
        sch.trace.show()


def ref_program(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def test_transformer_softmax_cuda():
    rules = ms.ScheduleRule.create("cuda")
    context = ms.TuneContext(
        mod=Softmax,
        target=Target("nvidia/nvidia-a100", host="llvm"),
        task_name="Double Rules Task",
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=rules,
            postprocs=[],
            mutator_probs={},
        ),
    )
    print("[INFO]**************space: ", context.generate_design_space()[0].mod)
    print("[INFO]**************num: ", len(context.generate_design_space()))
    for space in context.generate_design_space():
        mod = space.mod
        print("[INFO]**************space: ", mod)
        dev = tvm.device("cuda", 0)
        a_np = np.random.uniform(size=(64, 1280)).astype("float32")
        c_np = ref_program(a_np)
        buff_a = tvm.nd.array(a_np, dev)
        buff_c = tvm.nd.array(np.zeros((64, 1280), dtype="float32"), dev)
        myfunc = tvm.build(mod, target="cuda", name="softmax")
        myfunc(buff_a, buff_c)
        tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)


if __name__ == "__main__":
    test_tune_softmax_cuda()
    # test_transformer_softmax_cuda()