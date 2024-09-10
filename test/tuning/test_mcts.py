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


class Node(object):
    def __init__(self, state):
        self.state = state
        self.win_value = 0
        self.policy_value = None
        self.visits = 0
        self.parent = None
        self.children = []
        self.expanded = False
        self.player_number = None
        self.discovery_factor = 0.35

    def update_win_value(self, value):
        self.win_value += value
        self.visits += 1

        if self.parent:
            self.parent.update_win_value(value)

    def update_policy_value(self, value):
        self.policy_value = value

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_preferred_child(self, root_node):
        best_children = []
        best_score = float("-inf")
        for child in self.children:
            score = child.get_score(root_node)

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def get_score(self, root_node):
        discovery_operand = (
            self.discovery_factor
            * (self.policy_value or 1)
            * sqrt(log(self.parent.visits) / (self.visits or 1))
        )
        win_multiplier = (
            1 if self.parent.player_number == root_node.player_number else -1
        )
        win_operand = win_multiplier * self.win_value / (self.visits or 1)
        self.score = win_operand + discovery_operand
        return self.score

    def is_scorable(self):
        return self.visits or self.policy_value != None


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

    def make_choice(self):
        best_children = []
        most_visits = float("-inf")

        for child in self.root_node.children:
            if child.visits > most_visits:
                most_visits = child.visits
                best_children = [child]
            elif child.visits == most_visits:
                best_children.append(child)

        return random.choice(best_children)

    def get_optimal(self):
        node = self.traverse(self.root, greedy=True)
        return np.mean(node.data.domain), node.data.best_score

    def expand(self, node):
        self.child_finder(node, self)

        for child in node.children:
            child_win_value = self.node_evaluator(child, self)

            if child_win_value != None:
                child.update_win_value(child_win_value)

            if not child.is_scorable():
                self.random_rollout(child)
                child.children = []

        if len(node.children):
            node.expanded = True

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

    def random_rollout(self, node):
        self.child_finder(node, self)
        child = random.choice(node.children)
        node.children = []
        node.add_child(child)
        child_win_value = self.node_evaluator(child, self)

        if child_win_value != None:
            node.update_win_value(child_win_value)
        else:
            self.random_rollout(child)


def func(x, input_sketch):
    target = Target("cuda", host="llvm")
    (space,) = generate_design_space(
        kind="cuda", mod=input_sketch, target=target, types=x
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
