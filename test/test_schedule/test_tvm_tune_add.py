import logging
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.schedule_rule import ApplyCustomRule
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@tvm.script.ir_module
class Add:
    @T.prim_func
    def add(q: T.handle, k: T.handle, o: T.handle) -> None:
        Q = T.match_buffer(q, [64, 12, 256])
        K = T.match_buffer(k, [64, 12, 256])
        O = T.match_buffer(o, [64, 12, 256])

        for i, j, m in T.grid(64, 12, 256):
            with T.block("add"):
                vi, vj, vm = T.axis.remap("SSS", [i, j, m])
                O[vi, vj, vm] = Q[vi, vj, vm] + K[vi, vj, vm]


def test_tune_add_cuda():
    rules = ms.ScheduleRule.create("cuda")
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/nvidia-a100", host="llvm")
        database = ms.tir_integration.tune_tir(
            mod=Add,
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
        sch = ms.tir_integration.compile_tir(database, Add, target)
        assert sch is not None
        sch.mod.show()
        sch.trace.show()


def test_simple_bind():
    rules = ms.ScheduleRule.create("cuda")
    context = ms.TuneContext(
        mod=Add,
        target=Target("nvidia/nvidia-a100", host="llvm"),
        task_name="Double Rules Task",
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=rules,
            postprocs=[],
            mutator_probs={},
        ),
    )
    mod = context.generate_design_space()[0].mod
    print("[INFO]**************space: ", mod)
    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(64, 12, 256)).astype("float32")
    b_np = np.random.uniform(size=(64, 12, 256)).astype("float32")
    c_np = np.add(a_np, b_np)
    buff_a = tvm.nd.array(a_np, dev)
    buff_b = tvm.nd.array(b_np, dev)
    buff_c = tvm.nd.array(np.zeros((64, 12, 256), dtype="float32"), dev)
    myfunc = tvm.build(mod, target="cuda", name="add")
    myfunc(buff_a, buff_b, buff_c)
    tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)


if __name__ == """__main__""":
    test_tune_add_cuda()
    test_simple_bind()
