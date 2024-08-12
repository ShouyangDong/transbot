import logging
import tempfile


from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.target import Target

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@T.prim_func
def layernorm(
    input: T.handle, gamma: T.handle, beta: T.handle, output: T.handle
) -> None:
    input_ = T.match_buffer(input, [64, 100, 4096], dtype="float32")
    gamma_ = T.match_buffer(gamma, [4096], dtype="float32")
    beta_ = T.match_buffer(beta, [4096], dtype="float32")
    output_ = T.match_buffer(output, [64, 100, 4096], dtype="float32")
    input_sum = T.alloc_buffer([64, 100], dtype="float32")
    input_mean = T.alloc_buffer([64, 100], dtype="float32")

    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("input_sum"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSR", [ib, ir, ip])
            with T.init():
                input_sum[ib_0, ir_0] = T.float32(0)

            input_sum[ib_0, ir_0] = input_sum[ib_0, ir_0] + input_[ib_0, ir_0, ip_0]

    for ib, ir in T.grid(64, 100):
        with T.block("input_norm"):
            ib_0, ir_0 = T.axis.remap("SS", [ib, ir])

            input_mean[ib_0, ir_0] = input_sum[ib_0, ir_0] / T.float32(4096)

    input_diff = T.alloc_buffer([64, 100, 4096])
    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("input_diff"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSS", [ib, ir, ip])
            input_diff[ib_0, ir_0, ip_0] = (
                input_[ib_0, ir_0, ip_0] - input_mean[ib_0, ir_0]
            )

    input_variance = T.alloc_buffer([64, 100], dtype="float32")

    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("input_variance"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSR", [ib, ir, ip])

            with T.init():
                input_variance[ib_0, ir_0] = T.float32(0)

            input_variance[ib_0, ir_0] = (
                input_variance[ib_0, ir_0] + input_diff[ib_0, ir_0, ip_0]
            )

    variance_norm = T.alloc_buffer([64, 100], dtype="float32")
    for ib, ir in T.grid(64, 100):
        with T.block("variance_norm"):
            ib_0, ir_0 = T.axis.remap("SS", [ib, ir])
            variance_norm[ib_0, ir_0] = input_variance[ib_0, ir_0] / 4096

    variance_sqrt = T.alloc_buffer([64, 100], dtype="float32")
    for ib, ir in T.grid(64, 100):
        with T.block("variance_sqrt"):
            ib_0, ir_0 = T.axis.remap("SS", [ib, ir])
            variance_sqrt[ib_0, ir_0] = T.sqrt(variance_norm[ib_0, ir_0])

    diff_input = T.alloc_buffer([64, 100, 4096], dtype="float32")
    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("diff_input"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSS", [ib, ir, ip])
            diff_input[ib_0, ir_0, ip_0] = (
                input_[ib_0, ir_0, ip_0] - input_mean[ib_0, ir_0]
            )

    diff_gamma = T.alloc_buffer([64, 100, 4096], dtype="float32")
    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("diff_gamma"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSS", [ib, ir, ip])
            diff_gamma[ib_0, ir_0, ip_0] = input_diff[ib_0, ir_0, ip_0] * gamma_[ip_0]

    diff_div = T.alloc_buffer([64, 100, 4096], dtype="float32")
    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("diff_div"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSS", [ib, ir, ip])
            diff_div[ib_0, ir_0, ip_0] = diff_gamma[ib_0, ir_0, ip_0] / (
                variance_sqrt[ib_0, ir_0] + T.float32(1e-5)
            )

    for ib, ir, ip in T.grid(64, 100, 4096):
        with T.block("output"):
            ib_0, ir_0, ip_0 = T.axis.remap("SSS", [ib, ir, ip])
            output_[ib_0, ir_0, ip_0] = diff_div[ib_0, ir_0, ip_0] + beta_[ip_0]


def test_layernorm_cuda():
    rules = ms.ScheduleRule.create("cuda")
    context = ms.TuneContext(
        mod=layernorm,
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


def test_tune_layernorm_cuda():
    rules = ms.ScheduleRule.create("cuda")
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("nvidia/nvidia-a100")
        database = ms.tir_integration.tune_tir(
            mod=layernorm,
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
        sch = ms.tir_integration.compile_tir(database, layernorm, target)
        assert sch is not None
        sch.mod.show()
        sch.trace.show()


def test_layernorm_llvm():
    # rules = ms.ScheduleRule.create("llvm")
    rules = get_rules(kind="llvm", types=ms.schedule_rule.AutoInline) + [
        ms.schedule_rule.RandomComputeLocation(),
        ms.schedule_rule.InlineConstantScalars(),
    ]
    context = ms.TuneContext(
        mod=layernorm,
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
    test_layernorm_cuda()
    test_layernorm_llvm()
    test_tune_layernorm_cuda()
