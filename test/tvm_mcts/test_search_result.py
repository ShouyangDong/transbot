import tvm
from tvm import meta_schedule as ms
from softmax import Softmax
from tvm.meta_schedule.testing.space_generation import generate_design_space
from tvm.target import Target
import numpy as np

ActionSpace = [
    ms.schedule_rule.AutoBind(),
    ms.schedule_rule.AutoInline(
        into_producer=True,
        into_consumer=True,
        inline_const_tensor=True,
        disallow_if_then_else=False,
        require_injective=False,
        require_ordered=False,
    ),
    ms.schedule_rule.CrossThreadReduction(
        thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]
    ),
    ms.schedule_rule.MultiLevelTiling(
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
    ),
    ms.schedule_rule.ParallelizeVectorizeUnroll(
        max_jobs_per_core=-1,  # disable parallelize
        max_vectorize_extent=-1,  # disable vectorize
        unroll_max_steps=[0, 16, 64, 512, 1024],
        unroll_explicit=True,
    ),
    ms.schedule_rule.RandomComputeLocation(),
    ms.schedule_rule.InlineConstantScalars(),
]

mod = Softmax
for act in [2, 2, 3, 3, 0, 3, 5, 4]:
    actions = [ActionSpace[act]]
    spaces = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/nvidia-a100", host="llvm"),
        types=None,
        sch_rules=actions,
    )

    mod = spaces[0].mod


def ref_program(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


print("[INFO]**************space: ", mod)
dev = tvm.device("cuda", 0)
a_np = np.random.uniform(size=(64, 1280)).astype("float32")
c_np = ref_program(a_np)
buff_a = tvm.nd.array(a_np, dev)
buff_c = tvm.nd.array(np.zeros((64, 1280), dtype="float32"), dev)
myfunc = tvm.build(mod, target="cuda", name="softmax")
myfunc(buff_a, buff_c)
dev_module = myfunc.imported_modules[0]
print("-----GPU code-----")
print(dev_module.get_source())
tvm.testing.assert_allclose(buff_c.numpy(), c_np, rtol=1e-3)
