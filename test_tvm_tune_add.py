import logging
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.local_rpc import LocalRPC
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir.schedule import BlockRV, Schedule

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

@tvm.script.ir_module
class Add:
    @T.prim_func
    def add(q: T.handle, k: T.handle, o: T.handle) -> None:
        Q = T.match_buffer(q, [64, 4096, 12, 256])
        K = T.match_buffer(k, [64, 4096, 12, 256])
        O = T.match_buffer(o, [64, 4096, 12, 256])


        for i, j, m, n in T.grid(64, 4096, 12, 256):
            with T.block("add"):
                vi, vj, vm, vn = T.axis.remap("SSSS", [i, j, m, n])
                O[vi, vj, vm, vn] = Q[vi, vj, vm, vn] + K[vi, vj, vm, vn]

# def test_tune_add_cuda():
#     rules = ms.ScheduleRule.create("cuda")
#     with tempfile.TemporaryDirectory() as work_dir: 
#         target = Target("nvidia/nvidia-a100")
#         database = ms.tir_integration.tune_tir(
#             mod=add,
#             target=target,
#             work_dir=work_dir,
#             max_trials_global=32,
#             num_trials_per_iter=16,
#             space=ms.space_generator.PostOrderApply(
#                 sch_rules=rules,
#                 postprocs=[],
#                 mutator_probs={},
#             ),
#         )
#         sch = ms.tir_integration.compile_tir(database, add, target)
#         assert sch is not None
#         sch.mod.show()
#         sch.trace.show()

def test_simple_bind():
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
                sch_rules=[ms.schedule_rule.AutoBind()],
                postprocs=[],
                mutator_probs={},
            ),
        )
        sch = ms.tir_integration.compile_tir(database, Add, target)
        assert sch is not None
        sch.mod.show()
        sch.trace.show()

if __name__ == """__main__""":
    # test_tune_add_cuda()
    test_simple_bind()
    # mod = Add
    # context = ms.TuneContext(
    #     mod=mod,
    #     target=Target("nvidia/nvidia-a100", host="llvm"),
    #     task_name="Double Rules Task",
    #     space_generator=ms.space_generator.PostOrderApply(
    #         sch_rules=[ms.schedule_rule.AutoBind()],
    #         postprocs=[],
    #         mutator_probs={},
    #     ),
    # )
    # post_order_apply = context.space_generator
    # schs = post_order_apply.generate_design_space(mod)
    # # space=ms.space_generator.PostOrderApply(
    # #     sch_rules=[ms.schedule_rule.AutoBind()],
    # #     postprocs=[],
    # #     mutator_probs={},
    # # )
    # for sch in schs:
    #     sch.mod.show()
    #     sch.trace.show()