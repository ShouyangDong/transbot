import logging
import tempfile

import tvm
import numpy as np
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.target import Target

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@T.prim_func
def flash_atten(q: T.handle, k: T.handle, v: T.handle, o: T.handle) -> None:
    Q = T.match_buffer(q, [64, 4096, 12, 256], "float32")
    K = T.match_buffer(k, [64, 4096, 12, 256], "float32")
    V = T.match_buffer(v, [64, 4096, 12, 256], "float32")
    O = T.match_buffer(o, [64, 4096, 12, 256], "float32")

    S = T.alloc_buffer([64, 4096, 12, 12], "float32")
    N = T.alloc_buffer([64, 4096, 12, 12], "float32")
    for i, j, k, m, n in T.grid(64, 4096, 12, 12, 256):
        with T.block("flash_atten"):
            vi, vj, vk, vm, vn = T.axis.remap("SSSSR", [i, j, k, m, n])
            with T.init():
                S[vi, vj, vk, vm] = 0.0
            S[vi, vj, vk, vm] = (
                S[vi, vj, vk, vm] + Q[vi, vj, vk, vn] * K[vi, vj, vm, vn]
            )

    for i, j, m, n in T.grid(64, 4096, 12, 12):
        with T.block("norm"):
            vi, vj, vm, vn = T.axis.remap("SSSS", [i, j, m, n])
            N[vi, vj, vm, vn] = S[vi, vj, vm, vn] / 256

    softmax_maxelem = T.alloc_buffer([64, 4096, 12], "float32")
    for i, j, m, n in T.grid(64, 4096, 12, 12):
        with T.block("softmax_maxelem"):
            i_0, j_0, m_0, n_0 = T.axis.remap("SSSR", [i, j, m, n])
            with T.init():
                softmax_maxelem[i_0, j_0, m_0] = T.min_value("float32")
            softmax_maxelem[i_0, j_0, m_0] = T.max(
                softmax_maxelem[i_0, j_0, m_0], N[i_0, j_0, m_0, n_0]
            )

    softmax_exp = T.alloc_buffer([64, 4096, 12, 12], "float32")
    for i, j, m, n in T.grid(64, 4096, 12, 12):
        with T.block("softmax_exp"):
            i_0, j_0, m_0, n_0 = T.axis.remap("SSSS", [i, j, m, n])
            softmax_exp[i_0, j_0, m_0, n_0] = T.exp(
                N[i_0, j_0, m_0, n_0] - softmax_maxelem[i_0, j_0, m_0], dtype="float32"
            )

    softmax_expsum = T.alloc_buffer([64, 4096, 12], "float32")
    for i, j, m, n in T.grid(64, 4096, 12, 12):
        with T.block("softmax_expsum"):
            i_0, j_0, m_0, n_0 = T.axis.remap("SSSS", [i, j, m, n])
            with T.init():
                softmax_expsum[i_0, j_0, m_0] = T.float32(0)
            softmax_expsum[i_0, j_0, m_0] = (
                softmax_expsum[i_0, j_0, m_0] + softmax_exp[i_0, j_0, m_0, n_0]
            )

    softmax_norm = T.alloc_buffer([64, 4096, 12, 12], "float32")
    for i, j, m, n in T.grid(64, 4096, 12, 12):
        with T.block("softmax_norm"):
            i_0, j_0, m_0, n_0 = T.axis.remap("SSSS", [i, j, m, n])
            softmax_norm[i_0, j_0, m_0, n_0] = (
                softmax_exp[i_0, j_0, m_0, n_0] / softmax_expsum[i_0, j_0, m_0]
            )

    for i, j, k, m, n in T.grid(64, 4096, 12, 256, 12):
        with T.block("matmul"):
            vi, vj, vk, vm, vn = T.axis.remap("SSSSR", [i, j, k, m, n])
            with T.init():
                O[vi, vj, vk, vm] = 0.0
            O[vi, vj, vk, vm] = (
                O[vi, vj, vk, vm] + softmax_norm[vi, vj, vk, vn] * V[vi, vj, vn, vm]
            )


def test_flash_atten_cuda():
    # rules = ms.ScheduleRule.create("cuda")
    rules = [
        ms.schedule_rule.AutoInline(
            into_producer=True,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
        ),
        ms.schedule_rule.AutoBind(),
    ]
    context = ms.TuneContext(
        mod=flash_atten,
        target=Target("nvidia/nvidia-a100", host="llvm"),
        task_name="Double Rules Task",
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=rules,
            postprocs=[],
            mutator_probs={},
        ),
    )
    print("[INFO]**************space: ", context.generate_design_space()[0].mod)
    print("[INFO]**************num: ", context.generate_design_space())
    for space in context.generate_design_space():
        mod = space.mod
        print("[IFNO]************************space: ", mod)
        dev = tvm.device("cuda", 0)
        a_np = np.random.uniform(size=(64, 4096, 12, 256)).astype("float32")
        b_np = np.random.uniform(size=(64, 4096, 12, 256)).astype("float32")
        c_np = np.random.uniform(size=(64, 4096, 12, 256)).astype("float32")
        o_np = np.random.uniform(size=(64, 4096, 12, 256)).astype("float32")
        buff_a = tvm.nd.array(a_np, dev)
        buff_b = tvm.nd.array(b_np, dev)
        buff_c = tvm.nd.array(c_np, dev)
        buff_o = tvm.nd.array(o_np, dev)
        try:
            print("[INFO] Attempting to build the CUDA kernel")
            myfunc = tvm.build(mod, target="cuda", name="flashatten")
        except (tvm.TVMError, RuntimeError) as e:
            print(f"[ERROR] Compilation failed for this module: {e}")
            continue

        try:
            print("[INFO] Kernel built successfully, executing...")
            myfunc(buff_a, buff_b, buff_c, buff_o)
            print("Runtime success!")
        except (tvm.TVMError, RuntimeError) as e:
            print(f"[ERROR] Runtime failed for this module: {e}")
            continue


def test_tune_flash_atten_cuda():
    rules = ms.ScheduleRule.create("cuda")
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/nvidia-a100")
        database = ms.tir_integration.tune_tir(
            mod=flash_atten,
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
        sch = ms.tir_integration.compile_tir(database, flash_atten, target)
        assert sch is not None
        sch.mod.show()
        sch.trace.show()


def test_transform_attention_llvm():
    # rules = ms.ScheduleRule.create("llvm")
    rules = get_rules(kind="llvm", types=ms.schedule_rule.AutoInline) + [
        ms.schedule_rule.RandomComputeLocation(),
        ms.schedule_rule.InlineConstantScalars(),
    ]
    context = ms.TuneContext(
        mod=flash_atten,
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
    test_flash_atten_cuda()
    # test_transform_attention_llvm()
    # test_tune_flash_atten_cuda()
