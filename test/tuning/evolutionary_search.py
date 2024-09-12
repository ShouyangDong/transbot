from numpy import argsort
from numpy.random import seed

import random
import numpy as np
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.target import Target
from tvm.meta_schedule.testing.space_generation import generate_design_space

Actions = [
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


# objective function
def objective(mod, target, name, inputs, gflop):
    """We design an objective function. If compile and runtime error happens,
    then the score is quiet large.
    """

    try:
        myfunc = tvm.build(mod, target=target, name=name)
    except:
        return 0.0

    try:
        myfunc(*inputs)
    except:
        return 0.0

    evaluator = myfunc.time_evaluator(myfunc.entry_name, target, number=10)
    # gflops = (n_size * m_size * k_size) * 2 / 1e9
    time_ms = evaluator(*inputs).mean * 1e3
    print("%f ms, %f GOPS" % (time_ms, gflops / (time_ms / 1e3)))
    return gflops / (time_ms / 1e3)


def perform_action(mod, target, action):
    """Generates a design space for a given `action`. It calls `generate_design_space()`
    with specific parameters to apply the given scheduling rule (`action`) to the module.
    The function returns a new `ProgramState` object, which represents the new program
    state after applying the action."""
    # TODO(dongshouyang):change the spaces
    spaces = generate_design_space(
        kind="cuda",
        mod=mod,
        target=target,
        types=None,
        sch_rules=[action],
    )
    return spaces[0].mod


def es_comma(mod, n_iter, step_size, mu, lam):
    best, best_eval = None, 1e10
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = list()
    for _ in range(lam):
        candidate = None
        while candidate is None:
            candidate = random.choice(Actions)
        population.append(candidate)

    # perform the search
    for epoch in range(n_iter):
        states = [perform_action(mod, target, action) for action in population]
        for mod in states:
            print(mod)
        # evaluate fitness for the population
        scores = [objective(mod, target, name, inputs) for mod in states]
        # rank scores in ascending order
        ranks = argsort(argsort(scores))
        # select the indexes for the top mu ranked solutions
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]
        # create children from parents
        children = list()
        for i in selected:
            # check if this parent is the best solution ever seen
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                print("%d, Best: f(%s) = %.5f" % (epoch, best, best_eval))
            # create children for parent
            for _ in range(n_children):
                child = None
                while child is None:
                    child = random.choice(Actions)
                children.append(child)
        # replace population with children
        population = children
    return [best, best_eval]


@tvm.script.ir_module
class Softmax:
    @T.prim_func
    def main(
        A: T.Buffer((64, 1280), "float32"),
        T_softmax_norm: T.Buffer((64, 1280), "float32"),
    ) -> None:
        T_softmax_maxelem = T.alloc_buffer([64], dtype="float32", scope="local")
        T_softmax_exp = T.alloc_buffer([64, 1280], dtype="float32", scope="local")
        T_softmax_expsum = T.alloc_buffer([64], dtype="float32", scope="local")
        for i0, i1 in T.grid(64, 1280):
            with T.block("T_softmax_maxelem"):
                i0_1, k = T.axis.remap("SR", [i0, i1])
                with T.init():
                    T_softmax_maxelem[i0_1] = T.min_value("float32")
                T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], A[i0_1, k])
        for i0, i1 in T.grid(64, 1280):
            with T.block("T_softmax_exp"):
                i0_2, i1_1 = T.axis.remap("SS", [i0, i1])
                T_softmax_exp[i0_2, i1_1] = T.exp(
                    A[i0_2, i1_1] - T_softmax_maxelem[i0_2], dtype="float32"
                )
        for i0_3, i1 in T.grid(64, 1280):
            with T.block("T_softmax_expsum"):
                i0_4, k = T.axis.remap("SR", [i0_3, i1])
                with T.init():
                    T_softmax_expsum[i0_4] = T.float32(0)
                T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_exp[i0_4, k]
        for i0_5, i1 in T.grid(64, 1280):
            with T.block("T_softmax_norm"):
                i0_6, i1_2 = T.axis.remap("SS", [i0_5, i1])
                T_softmax_norm[i0_6, i1_2] = (
                    T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]
                )


def ref_program(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


if __name__ == "__main__":
    # seed the pseudorandom number generator
    seed(1)
    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(64, 1280)).astype("float32")
    c_np = ref_program(a_np)
    buff_a = tvm.nd.array(a_np, dev)
    buff_c = tvm.nd.array(np.zeros((64, 1280), dtype="float32"), dev)
    inputs = [buff_a, buff_c]
    target = Target("nvidia/nvidia-a100", host="llvm")
    name = "softmax"
    # define the total iterations
    n_iter = 100
    # define the maximum step size
    step_size = 0.15
    # number of parents selected
    mu = 4
    # the number of children generated by parents
    lam = 32
    # perform the evolution strategy (mu, lambda) search
    best, score = es_comma(Softmax, n_iter, step_size, mu, lam)
    print("Done!")
    print("f(%s) = %f" % (best, score))
