import random
import tvm
from tvm import meta_schedule as ms

Actions = [
    ms.schedule_rule.add_rfactor(),
    ms.schedule_rule.auto_bind(),
    ms.schedule_rule.auto_inline(),
    ms.schedule_rule.cross_thread_reduction(),
    ms.schedule_rule.multi_level_tiling(),
    ms.schedule_rule.parallel_vector_unroll(),
    ms.schedule_rule.random_compute_location(),
    ms.schedule_rule.inline_constant_scalars(),
]


class ProgramState(object):
    def __init__(self, mod, inputs, target, name):
        self.mod = mod
        self.inputs = inputs
        self.target = target
        self.name = name

    def is_terminal(self):
        return False

    def get_legal_actions(self):
        return Actions

    def perform_action(self, action):
        (space,) = generate_design_space(
            kind="cuda", mod=self.mod, target=self.target, types=action
        )
        return ProgramState(space.mod, self.inputs, self.target, self.name)

    def get_random_action(self):
        return random.choice(Actions)

    def evaluate(self):
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


def mcts(node, iterations):
    for _ in range(iterations):
        leaf = select(node)
        simulation_result = simulate(leaf)
        backpropagate(leaf, simulation_result)


def select(node):
    while not node.children and not node.state.is_terminal():
        node = expand(node)
    return node


def expand(node):
    actions = node.state.get_legal_actions()
    for action in actions:
        child_state = node.state.perform_action(action)
        child_node = Node(child_state, parent=node)
        node.children.append(child_node)
    return node.children[0]


def simulate(node):
    state = node.state
    while not state.is_terminal():
        action = state.get_random_action()
        state = state.perform_action(action)
    return evaluate(state)


def backpropagate(node, result):
    while node:
        node.visits += 1
        node.score += result
        node = node.parent


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
