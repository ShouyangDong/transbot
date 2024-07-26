import logging
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import meta_schedule as ms


from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule
from tvm.meta_schedule.testing.space_generation import get_rules

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@T.prim_func
def flash_atten(q: T.handle, k: T.handle, v: T.handle, o: T.handle) -> None:
    Q = T.match_buffer(q, [64, 4096, 12, 256])
    K = T.match_buffer(k, [64, 4096, 12, 256])
    V = T.match_buffer(v, [64, 4096, 12, 256])
    O = T.match_buffer(o, [64, 4096, 12, 256])

    S = T.alloc_buffer([64, 4096, 12, 12])
    N = T.alloc_buffer([64, 4096, 12, 12])
    for i, j, k, m, n in T.grid(64, 4096, 12, 12, 256):
        with T.block("flash_atten"):
            vi, vj, vk, vm, vn = T.axis.remap("SSSSR", [i, j, k, m, n])
            with T.init():
                S[vi, vj, vk, vm] = 0.0
            S[vi, vj, vk, vm] = (
                O[vi, vj, vk, vm] + Q[vi, vj, vk, vn] * K[vi, vj, vm, vn]
            )

    for i, j, m, n in T.grid(64, 4096, 12, 12):
        with T.block("norm"):
            vi, vj, vm, vn = T.axis.remap("SSSS", [i, j, m, n])
            N[vi, vj, vm, vn] = S[vi, vj, vm, vn] / 256

    softmax_maxelem = T.alloc_buffer([64, 4096, 12])
    for i, j, m, n in T.grid(64, 4096, 12, 12):
        with T.block("softmax_maxelem"):
            i_0, j_0, m_0, n_0 = T.axis.remap("SSSR", [i, j, m, n])
            with T.init():
                softmax_maxelem[i_0, j_0, m_0] = T.min_value("float32")
            softmax_maxelem[i_0, j_0, m_0] = T.max(
                softmax_maxelem[i_0, j_0, m_0], N[i_0, j_0, m_0, n_0]
            )

    softmax_exp = T.alloc_buffer([64, 4096, 12, 12])
    for i, j, m, n in T.grid(64, 4096, 12, 12):
        with T.block("softmax_exp"):
            i_0, j_0, m_0, n_0 = T.axis.remap("SSSS", [i, j, m, n])
            softmax_exp[i_0, j_0, m_0, n_0] = T.exp(
                N[i_0, j_0, m_0, n_0] - softmax_maxelem[i_0, j_0, m_0], dtype="float32"
            )

    softmax_expsum = T.alloc_buffer([64, 4096, 12])
    for i, j, m, n in T.grid(64, 4096, 12, 12):
        with T.block("softmax_expsum"):
            i_0, j_0, m_0, n_0 = T.axis.remap("SSSR", [i, j, m, n])
            with T.init():
                softmax_expsum[i_0, j_0, m_0] = T.float32(0)
            softmax_expsum[i_0, j_0, m_0] = (
                softmax_expsum[i_0, j_0, m_0] + softmax_exp[i_0, j_0, m_0, n_0]
            )

    softmax_norm = T.alloc_buffer([64, 4096, 12, 12])
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
                O[vi, vj, vk, vm] + softmax_norm[vi, vj, vk, vn] * V[vi, vj, vk, vm]
            )


def testune_flash_atten_cuda():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/nvidia-a100")
        database = ms.tir_integration.tune_tir(
            mod=flash_atten,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
        )
        sch = ms.tir_integration.compile_tir(database, flash_atten, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()


def test_transform_attention_llvm():
    # rules = ms.ScheduleRule.create("llvm")
    rules = get_rules(kind="cuda", types=ms.schedule_rule.AutoInline) + [ms.schedule_rule.RandomComputeLocation(), ms.schedule_rule.InlineConstantScalars()]
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
    # testune_flash_atten_cuda()
    test_transform_attention_llvm()
