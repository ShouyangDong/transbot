import logging
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.space_generation import get_rules
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule

@T.prim_func
def sddmm(a:T.handle, b: T.handle, c: T.handle) -> None：
    A = T.match_buffer(a, [512, 512], "float32")
    B = T.match_buffer(b, [512, 512], "float32")
    C = T.match_buffer(c, [512, 512], "float32")

    for i, j, k in T.grid(512):
        with T.block("sddmm"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B [vj, vk]

def test_sddmm_cuda():
    rules = ms.ScheduleRule.create("cuda")
    context = ms.TuneContext(
        mod=sddmm,
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


def test_transform_attention_llvm():
    # rules = ms.ScheduleRule.create("llvm")
    rules = get_rules(kind="llvm", types=ms.schedule_rule.AutoInline) + [
        ms.schedule_rule.RandomComputeLocation(),
        ms.schedule_rule.InlineConstantScalars(),
    ]
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
    test_sddmm_llvm()