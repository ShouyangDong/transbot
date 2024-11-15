import math
from enum import Enum

import numpy as np
import tvm
import tvm.testing
from pydantic import BaseModel
from tvm.meta_schedule.testing.space_generation import generate_design_space

ROOT_UCT_SCORE = 10000

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
    ms.schedule_rule.MultiLevelTilingWithIntrin(
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

ActionNameMap = {
    ms.schedule_rule.AutoBind(): "Auto Bind",
    ms.schedule_rule.AutoInline(
        into_producer=True,
        into_consumer=True,
        inline_const_tensor=True,
        disallow_if_then_else=True,
        require_injective=False,
        require_ordered=False,
    ): "AutoInline",
    ms.schedule_rule.CrossThreadReduction(
        thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]
    ): "CrossThreadReduction",
    ms.schedule_rule.MultiLevelTilingWithIntrin(
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
    ): "MultiLevelTilingWithIntrin",
    ms.schedule_rule.ParallelizeVectorizeUnroll(
        max_jobs_per_core=-1,
        max_vectorize_extent=-1,
        unroll_max_steps=[0, 16, 64, 512, 1024],
        unroll_explicit=True,
    ): "ParallelizeVectorizeUnroll",
    ms.schedule_rule.RandomComputeLocation(): "RandomComputeLocation",
    ms.schedule_rule.InlineConstantScalars(): "InlineConstantScalars",
}


class MCTSNode(BaseModel):
    action: str
    parent: MCTSNode | None = None
    children: list[MCTSNode] = []
    visits: int = 0
    Q: float = 0
    reward_samples: list[int] = []

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(action={ActionNameMap[self.action]}, Q={self.Q:.2f}, visits={self.visits})"

    def add_reward(self, reward: int):
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)

        # Average worse-case and average outcomes
        self.Q = (min_reward + arg_reward) / 2


class SelectionPolicy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2
    PAIRWISE_IMPORTANCE_SAMPLING = 3


class InitializeStrategy(Enum):
    ZERO_SHOT = 1
    DUMMY_action = 2


class MCTSr(BaseModel):
    problem: str
    max_rollouts: int
    exploration_constant: float = 1.0
    max_children: int = 2
    epsilon: float = 1e-10
    reward_limit: int = 95
    excess_reward_penalty: int = 5
    selection_policy: SelectionPolicy = SelectionPolicy.IMPORTANCE_SAMPLING
    initialize_strategy: InitializeStrategy = InitializeStrategy.ZERO_SHOT

    root: MCTSNode = MCTSNode(action="I don't know.")

    critiques: list[str] = []
    refinements: list[str] = []
    rewards: list[float] = []
    selected_nodes: list[MCTSNode] = []

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        raise NotImplementedError()

    def _evaluate_action(self, node: MCTSNode) -> int:
        raise NotImplementedError()

    def self_evaluate(self, node: MCTSNode):
        """Evaluate the quality of the action. Sample `num_samples` times and average the results."""
        reward = self._evaluate_action(node)

        if reward > self.reward_limit:
            reward -= self.excess_reward_penalty

        node.add_reward(reward)

    def backpropagate(self, node: MCTSNode):
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (parent.Q + best_child_Q) / 2
            parent.visits += 1
            parent = parent.parent

    def uct(self, node: MCTSNode):
        if not node.parent:
            # Using an arbitrarily high UCT score for the root node.
            # helps to prioritize breadth.
            return ROOT_UCT_SCORE

        return node.Q + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        )

    def is_fully_expanded(self, node: MCTSNode):
        return len(node.children) >= self.max_children or any(
            child.Q > node.Q for child in node.children
        )

    def select_node(self):
        """Select a non-fully expanded node with the highest UCT value.

        A node is fully expanded if either:
        1. It has reached the max number of children
        2. Any of its children have a Q value greater than its own
        """
        candidates: list[MCTSNode] = []
        to_consider = deque([self.root])

        while to_consider:
            current_node = to_consider.popleft()
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_consider.extend(current_node.children)

        if not candidates:
            return self.root

        if self.selection_policy == SelectionPolicy.GREEDY:
            return max(calculates, key=self.uct)
        elif self.selection_policy == SelectionPolicy.IMPORTANCE_SAMPLING:
            # Sample, weighted by UCT score
            uct_scores = [self.uct(node) for node in candidates]
            selected_pair_idx = random.choices(
                range(len(candidates)), weights=uct_scores, k=1
            )[0]
            return candidates[selected_pair_idx]

        elif self.selection_policy == SelectionPolicy.PAIRWISE_IMPORTANCE_SAMPLING:
            # Sample, weighted by the difference in UCT scores between pairs
            uct_scores = [self.uct(node) for node in candidates]
            pairs = [
                (i, j) for i in range(len(candidates)) for j in range(len(candidates))
            ]
            pair_weights = [
                max(uct_scores[i], uct_scores[j]) - min(uct_scores[i], uct_scores[j])
                for i, j in pairs
            ]
            selected_pair_idx = random.choices(
                range(len(pairs)), weightes=pair_weights, k=1
            )[0]
            selected_candidate_idx = max(
                pairs[selected_pair_idx], key=lambda x: uct_scores[x]
            )
            return candidates[selected_candidate_idx]
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")

    def zero_shot(self) -> str:
        """Generate a zero-shot action."""
        raise NotImplementedError()

    def initialize(self):
        """Generate a zero-shot action."""
        if self.initialize_strategy == InitializeStrategy.ZERO_SHOT:
            self.root = MCTSNode(action=self.zero_shot())
        elif self.initialize_strategy == InitializeStrategy.DUMMY_action:
            self.root = MCTSNode(action="I don't know.")
        else:
            raise ValueError(f"Invalid initialize strategy: {self.initialize_strategy}")

    def run(self):
        self.initialize()
        for _ in tqdm.tqdm(range(self.max_rollouts)):
            node = self.select_node()
            self.self_evaluate(node)
            child = self.self_refine(node)
            node.add_child(child)
            self.self_evaluate(child)
            self.backpropagate(child)

        return self.get_best_action()

    def get_best_action(self):
        from collections import deque

        to_visit = deque([self.root])
        best_node = self.root

        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
            to_visit.extend(current_node.children)

        return best_node.action

    def print(self):
        print_tree(self.root)


class MCTSrGPT4o(MCTSr):
    def zero_shot(self, node: MCTSNode) -> str:
        """Generates a design space for a given `action`. It calls `generate_design_space()`
        with specific parameters to apply the given scheduling rule (`action`) to the module.
        The function returns a new `ProgramState` object, which represents the new program
        state after applying the action."""
        # TODO(dongshouyang):change the spaces
        spaces = generate_design_space(
            kind="cuda",
            mod=node.mod,
            target=node.target,
            types=None,
            sch_rules=[node.action],
        )
        return spaces[0].mod

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        critique_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": gpt_4o_prompt_config.critic_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_action>\n{node.action}\n</current_action>",
                        ]
                    ),
                },
            ],
            model=gpt_4o_prompt_config.model,
            max_tokens=4000,
        )
        critique = critique_response.choices[0].message.content
        assert critique is not None
        self.critiques.append(critique)

        refined_action_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": gpt_4o_prompt_config.refine_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_action>\n{node.action}\n</current_action>",
                            f"<critique>\n{critique}\n</critique>",
                        ]
                    ),
                },
            ],
            model=gpt_4o_prompt_config.model,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        refined_action = RefineResponse.model_validate_json(
            refined_action_response.choices[0].message.content
        )
        self.refinements.append(refined_action)

        return MCTSNode(
            action=f"# Thought {refined_action.thought}\n\n# action\n{refined_action.action}",
            parent=node,
        )

    def _evaluate_action(self, node: MCTSNode) -> int:
        """Evaluate the final script. If the result is correct, then returns 1, otherwise, returns 0."""
        try:
            myfunc = tvm.build(node.mod, target=node.target, name=node.name)
        except:
            return 0

        try:
            myfunc(*inputs)
        except:
            return 0
        return 1


def print_tree(node: MCTSNode | None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = repr(node)
    for line in node_str.split("\n"):
        print(indent + line)
    for child in node.children:
        print_tree(child, level + 1)
