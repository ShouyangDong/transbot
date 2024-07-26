from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm.script import tir as T
from tvm.target import Target


@T.prim_func
def element_wise(var_A: T.handle, var_B: T.handle) -> None:
    A = T.match_buffer(var_A, [512, 512], dtype="float32")
    B = T.match_buffer(var_B, [512, 512], dtype="float32")
    for i, j in T.grid(512, 512):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] + 1.0


decision_0 = [
    ("SampleCategorical", 5),
]
mod = element_wise
print("[INFO]*********************mod: ", mod)
actual = generate_design_space(
    kind="cuda",
    mod=mod,
    target=Target("nvidia/geforce-rtx-3080", host="llvm"),
    types=ms.schedule_rule.AutoBind,
)
print("==============================")
print("[INFO]**********************mod: ", actual[0].mod)


@T.prim_func
def add(q: T.handle, k: T.handle, o: T.handle) -> None:
    Q = T.match_buffer(q, [64, 4096, 12, 256])
    K = T.match_buffer(k, [64, 4096, 12, 256])
    O = T.match_buffer(o, [64, 4096, 12, 256])

    for i, j, m, n in T.grid(64, 4096, 12, 256):
        with T.block("add"):
            vi, vj, vm, vn = T.axis.remap("SSSS", [i, j, m, n])
            O[vi, vj, vm, vn] = Q[vi, vj, vm, vn] + K[vi, vj, vm, vn]


mod = add
print("[INFO]*********************mod: ", mod)
actual = generate_design_space(
    kind="cuda",
    mod=mod,
    target=Target("nvidia/nvidia-a100", host="llvm"),
    types=ms.schedule_rule.AutoBind,
)
print("==============================")
print("[INFO]**********************mod: ", actual[0].mod)
