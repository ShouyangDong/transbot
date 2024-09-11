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


class ProgramState(object):
    def __init__(self, mod, inputs, target, name):
        self.mod = mod
        self.inputs = inputs
        self.target = target
        self.name = name

    def is_terminal(self):
        """Check if the current state is a terminal state."""
        if not self.evaluate():
            return True
        return False

    def get_legal_actions(self):
        """Obtain all possible actions."""
        return Actions

    def perform_action(self, action):
        """Generates a design space for a given `action`. It calls `generate_design_space()`
        with specific parameters to apply the given scheduling rule (`action`) to the module.
        The function returns a new `ProgramState` object, which represents the new program
        state after applying the action."""
        # TODO(dongshouyang):change the spaces
        spaces = generate_design_space(
            kind="cuda",
            mod=self.mod,
            target=self.target,
            types=None,
            sch_rules=[action],
        )
        return ProgramState(spaces[0].mod, self.inputs, self.target, self.name)

    def get_random_action(self):
        """Randomly select and return an action from the available list of Actions."""
        return random.choice(Actions)

    def evaluate(self):
        """This `evaluate()` function attempts to build and execute a function using TVM.
        - The function first tries to build the function using `tvm.build()`.
            If this step fails, it returns `False`.
        - If the build is successful, it tries to run the function with the provided inputs.
            If this execution step fails, it also returns `False`.
        - If both steps (build and execution) are successful, the function returns `True`,
            indicating that the evaluation was successful."""
        try:
            myfunc = tvm.build(self.mod, target=self.target, name=self.name)
        except:
            return False

        try:
            myfunc(*self.inputs)
        except:
            return False

        return True


class Node(object):
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0

    def __str__(self):
        return f"The creature state is {self.state} and the parenet is {self.parent}"

    def __repr__(self):
        return f"Node('{self.state}', {self.parent})"


def mcts(node, iterations):
    """Run MCTS for a given number of iterations."""
    for _ in range(iterations):
        leaf = select(node)
        simulation_result = simulate(leaf)
        backpropagate(leaf, simulation_result)


def select(node):
    """Select the leaf node to expand or simulate from."""
    while not node.children and not node.state.is_terminal():
        node = expand(node)
    return node


def expand(node):
    """Expand the node by adding a child for each legal action."""
    actions = node.state.get_legal_actions()
    for action in actions:
        child_state = node.state.perform_action(action)
        child_node = Node(child_state, parent=node)
        node.children.append(child_node)
    return node.children[0]


def simulate(node):
    """Simulate from the given node to a terminal state."""
    state = node.state
    while not state.is_terminal():
        action = state.get_random_action()
        state = state.perform_action(action)
    return state.evaluate()


def backpropagate(node, result):
    """Update the node's scores as the simulation result propagates back up the tree."""
    while node:
        node.visits += 1
        node.score += result
        print("[INFO]************parent node: ", node)
        node = node.parent.evaluate()


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
    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(64, 1280)).astype("float32")
    c_np = ref_program(a_np)
    buff_a = tvm.nd.array(a_np, dev)
    buff_c = tvm.nd.array(np.zeros((64, 1280), dtype="float32"), dev)
    inputs = [buff_a, buff_c]
    target = Target("nvidia/nvidia-a100", host="llvm")
    name = "softmax"
    state = ProgramState(Softmax, inputs, target, name)
    node = Node(state)
    iterations = 100
    mcts(node, iterations)
