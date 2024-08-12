import tempfile

import numpy as np

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.target import Target


@T.prim_func
def sddmm(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [512, 512], "float32")
    B = T.match_buffer(b, [512, 512], "float32")
    C = T.match_buffer(c, [512, 512], "float32")

    for i, j, k in T.grid(512, 512, 512):
        with T.block("sddmm"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_sddmm_cuda():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/nvidia-a100")
        database = ms.tir_integration.tune_tir(
            mod=sddmm,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
        )
        sch = ms.tir_integration.compile_tir(database, sddmm, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()

        dev = tvm.device("cuda", 0)
        a_np = np.random.uniform(size=(512, 512)).astype("float32")
        b_np = np.random.uniform(size=(512, 512)).astype("float32")
        c_np = a_np.dot(b_np)
        buff_a = tvm.nd.array(a_np, dev)
        buff_b = tvm.nd.array(b_np, dev)
        buff_c = tvm.nd.array(np.zeros((512, 512), dtype="float32"), dev)
        myfunc = tvm.build(sch.mod, target="cuda", name="sddmm")
        myfunc(buff_a, buff_b, buff_c)
        tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)


def test_transform_attention_llvm():
    rules = ms.ScheduleRule.create("llvm")
    context = ms.TuneContext(
        mod=sddmm,
        target=Target("llvm --num-cores=16"),
        task_name="Double Rules Task",
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=rules,
            postprocs=[],
            mutator_probs={},
        ),
    )

    print("[INFO]**************space: ", context.generate_design_space()[0].mod)
    print("[INFO]**************num: ", len(context.generate_design_space()))


if __name__ == """__main__""":
    test_sddmm_cuda()
    # test_sddmm_llvm()
