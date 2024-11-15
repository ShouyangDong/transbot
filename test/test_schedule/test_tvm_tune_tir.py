import tempfile

import numpy as np
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.target import Target


@T.prim_func
def two_step(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (1024, 1024), "float32")
    B = T.alloc_buffer((1024, 1024), "float32")
    D = T.alloc_buffer((1024, 1024), "float32")
    C = T.match_buffer(c, (1024, 1024), "float32")
    for i, j in T.grid(1024, 1024):
        with T.block("A"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0

    for i, j in T.grid(1024, 1024):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = B[vi, vj] + 3.0

    for i, j in T.grid(1024, 1024):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = D[vi, vj] + 3.0


rules = ms.ScheduleRule.create("llvm")
with tempfile.TemporaryDirectory() as work_dir:
    target = Target("llvm --num-cores=16")
    database = ms.tir_integration.tune_tir(
        mod=two_step,
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
    sch = ms.tir_integration.compile_tir(database, two_step, target)
    assert sch is not None
    sch.mod.show()
    sch.trace.show()

    dev = tvm.cpu(0)
    a_np = np.random.uniform(size=(1024, 1024)).astype("float32")
    c_np = a_np * 2 + 6
    buff_a = tvm.nd.array(a_np, dev)
    buff_c = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)
    myfunc = tvm.build(sch.mod, target="llvm", name="add")
    myfunc(buff_a, buff_c)
    tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)
