import tempfile

from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.space_generation import generate_design_space
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


def test_cuda_element_wise():
    @T.prim_func
    def elementwise_0(
        A: T.Buffer((512, 512), "float32"),
        B: T.Buffer((512, 512), "float32"),
    ) -> None:
        # body
        # with T.block("root")
        for i_j_fused_0 in T.thread_binding(256, thread="blockIdx.x"):
            for i_j_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                with T.block("C"):
                    vi = T.axis.spatial(512, (i_j_fused_0 * 1024 + i_j_fused_1) // 512)
                    vj = T.axis.spatial(512, (i_j_fused_0 * 1024 + i_j_fused_1) % 512)
                    T.reads(A[vi, vj])
                    T.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] + T.float32(1)

    decision_0 = [
        ("SampleCategorical", 5),
    ]
    mod = element_wise
    spaces = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/nvidia-a100", host="llvm"),
        types=ms.schedule_rule.AutoBind,
    )

    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/nvidia-a100", host="llvm")
        database = ms.tir_integration.tune_tir(
            mod=mod,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
        )
        sch = ms.tir_integration.compile_tir(database, mod, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()


if __name__ == "__main__":
    test_cuda_element_wise()
