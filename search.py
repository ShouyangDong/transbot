import heapq
import random
from ast_transformation import loop_bind, loop_split, loop_fuse
from ast_visitor import get_ajcent_loop


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = 0  # 从初始状态到当前状态的代价
        self.heuristic = heuristic(state)  # 当前状态的启发式估计

    @property
    def total_cost(self):
        return self.cost + self.heuristic

    def __eq__(self, other):
        return self.total_cost == other.total_cost

    def __lt__(self, other):
        return self.total_cost < other.total_cost


def a_star_search(start_state, goal_state, actions, heuristic):
    def node_from_tuple(node_tuple):
        # 从元组中提取 Node 对象
        return node_tuple[1]

    open_set = []
    start_node = Node(start_state)
    heapq.heappush(open_set, (start_node.total_cost, start_node))

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)

        if current_node.state == goal_state:
            return reconstruct_path(current_node)

        for action in actions:
            next_state, action_cost = apply_action(current_node.state, action)
            next_cost = current_node.cost + action_cost  # action.cost
            next_node = Node(next_state, current_node, action)

            if all(node_from_tuple(t) != next_node for t in open_set):
                heapq.heappush(open_set, (next_cost + heuristic(next_state), next_node))


def reconstruct_path(node):
    # 从目标节点回溯到起始节点，以构建完整的路径
    path = []
    while node:
        path.append(node.action)
        node = node.parent
    return path[::-1]


def heuristic(state):
    h_cost = 40
    if "__global__" in state:
        h_cost -= 10
    if "threadIdx.x" in state:
        h_cost -= 10
    return h_cost


def apply_action(start_state, action):
    if action == "func_prefix":
        if "__global__" in start_state:
            return start_state, 100
        state = "__global__ " + start_state
        return state, 10

    elif action == "loop_fuse":
        # get the ajcent aixs
        axis = get_ajcent_loop(start_state)
        if len(axis) < 2:
            return start_state, 100
        state = loop_fuse(start_state, axis[0], axis[1])
        return state, 10

    elif action == "loop_bind":
        axises = get_ajcent_loop(start_state)
        if "threadIdx.x" in start_state:
            return start_state, 100
        axis = random.choice(axises)
        state = loop_bind(start_state, loop_index=axis, thread_name="threadIdx.x")
        return state, 10

    elif action == "loop_split":
        axises = get_ajcent_loop(start_state)
        if len(axises) < 1:
            return start_state, 100
        axis = random.choice(axises)
        state = loop_split(start_state, loop_index=axis, factor=2)
        return state, 10

    else:
        raise RuntimeError("Cannot handle!")


if __name__ == "__main__":
    # 定义初始状态和目标状态
    start_state = """
    void add_kernel(float* output, float* input1, float* input2) {
        for (int i = 0; i < 18; i++) {
            for (int j = 0; j < 128; j++) {
                int index = i * 128 + j;
                output[index] = input1[index] + input2[index];
            }
        }
    }
    """

    goal_state = """
    __global__ void add_kernel(float* output, float* input1, float* input2) {
        if (blockIdx.x < 3) {
            if (threadIdx.x < 1024) {
                if (blockIdx.x * 1024 + threadIdx.x < 2304) {
                    output[blockIdx.x * 1024 + threadIdx.x] = input1[blockIdx.x * 1024 + threadIdx.x] + input2[blockIdx.x * 1024 + threadIdx.x];
                }
            }
        }
    }
    """

    # 执行 A* 搜索
    actions = ["loop_fuse", "loop_split", "loop_bind", "func_prefix"]

    # 定义可能的动作列表
    transformation_sequence = a_star_search(start_state, goal_state, actions, heuristic)
