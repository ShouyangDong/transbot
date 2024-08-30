import tempfile

import numpy as np

import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.target import Target


@T.prim_func
def sddmm(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [64, 4096, 12, 256], "float32")
    B = T.match_buffer(b, [64, 4096, 12, 256], "float32")
    C = T.match_buffer(c, [64, 4096, 12, 12], "float32")

    for i, j, k, m, n in T.grid(64, 4096, 12, 12, 256):
        with T.block("gemm"):
            vi, vj, vk, vm, vn = T.axis.remap("SSSSR", [i, j, k, m, n])
            with T.init():
                C[vi, vj, vk, vm] = 0.0
            C[vi, vj, vk, vm] = (
                C[vi, vj, vk, vm] + A[vi, vj, vk, vn] * B[vi, vj, vm, vn]
            )


def test_tune_sddmm_cuda():
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
        a_np = np.random.uniform(size=(64, 4096, 12, 256)).astype("float32")
        b_np = np.random.uniform(size=(64, 4096, 12, 256)).astype("float32")

        # c_np = a_np.dot(np.transpose(b_np), axes=[0, 1, 3, 2])
        buff_a = tvm.nd.array(a_np, dev)
        buff_b = tvm.nd.array(b_np, dev)
        buff_c = tvm.nd.array(np.zeros((64, 4096, 12, 12), dtype="float32"), dev)
        myfunc = tvm.build(sch.mod, target="cuda", name="sddmm")
        myfunc(buff_a, buff_b, buff_c)
        # tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)
        # Evaluate execution time.
        # evaluator = myfunc.time_evaluator(myfunc.entry_name, dev, min_repeat_ms=500)
        # mean_time = np.median(evaluator(buff_a, buff_b, buff_c).results)
        # print("Execution time of this operator: %.3f ms" % (mean_time * 1000))

        # tensor_a = torch.randn(512, 512).to(torch.float32).cuda()
        # tensor_b = torch.randn(512, 512).to(torch.float32).cuda()

        # def f():
        #     result = torch.matmul(tensor_a, tensor_b)
        #     # necessary because kernel launches are async
        #     torch.cuda.synchronize()

        # time_ms = timeit.timeit(f, number=100) / 100 * 1000
        # print(f"Handwritten CUDA: {time_ms:.3f}ms")
        # print(f"Speedup: {time_ms/(mean_time*1000):.2f}x")


def test_gemm_cuda():
    # rules = ms.ScheduleRule.create("cuda")
    # print("[INFO]*******rules: ", rules)
    rules = [ms.schedule_rule.AutoBind()]
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
    # print("[INFO]**************space: ", context.generate_design_space()[0].mod)
    # print("[INFO]**************num: ", context.generate_design_space())
    for space in context.generate_design_space():
        dev = tvm.device("cuda", 0)
        a_np = np.random.uniform(size=(64, 4096, 12, 256)).astype("float32")
        b_np = np.random.uniform(size=(64, 4096, 12, 256)).astype("float32")
        buff_a = tvm.nd.array(a_np, dev)
        buff_b = tvm.nd.array(b_np, dev)
        buff_c = tvm.nd.array(np.zeros((64, 4096, 12, 12), dtype="float32"), dev)

        try:
            print("[INFO] Attempting to build the CUDA kernel")
            myfunc = tvm.build(space.mod, target="cuda", name="sddmm")
        except (tvm.TVMError, RuntimeError) as e:
            print(f"[ERROR] Compilation failed for this module: {e}")
            continue

        try:
            print("[INFO] Kernel built successfully, executing...")
            myfunc(buff_a, buff_b, buff_c)
            print("Runtime success!")
            dev_module = myfunc.imported_modules[0]
            print("-----GPU code-----")
            print(dev_module.get_source())
        except (tvm.TVMError, RuntimeError) as e:
            print(f"[ERROR] Runtime failed for this module: {e}")
            continue


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
    # test_tune_sddmm_cuda()
    test_gemm_cuda()
    # test_sddmm_llvm()
