from treelib import Tree
import numpy as np
import tvm
from tvm import meta_schedule as ms
import random

def verify_runtime(node):
    """Tries to build and then execute a function defined in a TVM module.
    It uses exception handling to catch any errors that might occur during
    these processes. If everything goes smoothly, it returns True, indicating
    that the node is ready for runtime execution. If any issues arise during 
    building or execution, it catches the exceptions and returns False."""
    try:
        myfunc = tvm.build(node.mod, target=nod.target, name=node.name)
    except:
        return False

    try:
        myfunc(*node.inputs)
    except:
        return False

    return True


class Data(object):
    def __init__(
        self, domain=0, visits=0, best_score=(-np.inf), coef=2, is_terminal=False
    ):
        self.visits = visits
        self.best_score = best_score
        self.coef = coef
        self.domain = domain

    def ucb(self, parent_visits):
        if self.visits == 0:
            return np.inf
        return self.best_score + self.coef * np.sqrt(
            np.log(parent_visits) / self.visits
        )


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

class MCTS(object):
    def __init__(self, func, max_depth=10, rollout_times=10):
        self.max_depth = max_depth
        self.tree = Tree()
        self.func = func
        self.domain = Actions
        self.rollout_times = rollout_times
        self.root = self.tree.create_node("root", data=Data(domain=self.domain))

    def train(self, steps=100):
        for n in range(steps):
            node = self.root
            while not self.is_terminal(node):
                node = self.traverse(node)
                score = self.rollout(node)
                self.back_propagate(node, score)

    def get_optimal(self):
        node = self.traverse(self.root, greedy=True)
        return np.mean(node.data.domain), node.data.best_score

    def expand(self, node):
        domain_left = [
            node.data.domain[0],
            (node.data.domain[0] + node.data.domain[1]) / 2,
        ]
        domain_right = [
            (node.data.domain[0] + node.data.domain[1]) / 2,
            node.data.domain[1],
        ]
        left_node = self.tree.create_node(
            "left", parent=node, data=Data(domain=domain_left)
        )
        right_node = self.tree.create_node(
            "right", parent=node, data=Data(domain=domain_right)
        )
        return left_node

    def traverse(self, node, greedy=False):
        while True:
            if self.is_terminal(node):
                return node
            if not self.is_fully_expanded(node):
                return self.expand(node)
            node = self.get_best_child(node, greedy=greedy)

    def is_fully_expanded(self, node):
        return bool(self.tree.children(node.identifier))

    def is_terminal(self, node):
        return self.tree.level(node.identifier) == self.max_depth or verify_runtime(
            node
        )

    def back_propagate(self, node, score):
        while True:
            node.data.best_score = max(node.data.best_score, score)
            node.data.visits += 1
            if node.is_root():
                break
            node = self.tree.parent(node.identifier)

    def get_best_child(self, node, greedy):
        best_child = None
        children = self.tree.children(node.identifier)
        if children:
            parent_visits = node.data.visits
            if greedy:
                scores = [child.data.best_score for child in children]
            else:
                scores = [child.data.ucb(parent_visits) for child in children]
            best_child = children[np.argmax(scores)]
        return best_child

    def rollout(self, node):
        domain = node.data.domain
        scores = []
        for n in range(self.rollout_times):
            x = random.choices(domain)
            score = self.func(x, node.input)
            scores.append(score)
        return np.max(scores)


def func(x, input_sketch):
    target = Target("cuda", host="llvm")
    (space, ) = generate_design_space(
        kind = "cuda",
        mod = input_sketch,
        target=target,
        types=x
    )
    return space.mod

mcts = MCTS(func, max_depth=16)
mcts.train()
x_best, y = mcts.get_optimal()

print(
    "The optimal solution is ~ {:.5f}, which is located at x ~ {:.5f}.".format(
        y, x_best
    )
)
