import heapq
import random
import subprocess
from ast_transformation import loop_bind, loop_split, loop_fuse
from ast_visitor import get_ajcent_loop


def run_test(file_name, test_file):
    try:
        output = subprocess.run(
            ["python", test_file, "--file", file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=400,
        )
        return True, output
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        return False, e.output


def run_compilation(so_name, file_name):
    try:
        output = subprocess.run(
            ["nvcc", "-Xcompiler", "-fPIC", "-shared", "-o", so_name, file_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            check=True,
            text=True,
            timeout=15,
        )
        return True, output
    except subprocess.CalledProcessError as e:
        return False, e.output


class Node(object):
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


def compile_check(code):
    template_code = """extern "C" void add_kernel(float *C, float *A, float *B, int size) {
        float *d_A, *d_B, *d_C;

        cudaMalloc(&d_A, size * sizeof(float));
        cudaMalloc(&d_B, size * sizeof(float));
        cudaMalloc(&d_C, size * sizeof(float));

        cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(1024);
        dim3 numBlocks((size + 1024 - 1) / 1024);

        add<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

        cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        }
    """
    with open("./macro/cuda_macro.txt", "r") as f:
        macro = f.read()
        f.close()

    code = macro + code + template_code

    with open("./add_18_128.cu", mode="w") as f:
        f.write(code)
        f.close()

    success, output = run_compilation("./add_18_128.so", "./add_18_128.cu")
    return success


def a_star_search(start_state, actions, heuristic):
    def node_from_tuple(node_tuple):
        # 从元组中提取 Node 对象
        return node_tuple[1]

    def check_file(code):
        template_code = """extern "C" void add_kernel(float *C, float *A, float *B, int size) {
            float *d_A, *d_B, *d_C;

            cudaMalloc(&d_A, size * sizeof(float));
            cudaMalloc(&d_B, size * sizeof(float));
            cudaMalloc(&d_C, size * sizeof(float));

            cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);

            dim3 blockSize(1024);
            dim3 numBlocks((size + 1024 - 1) / 1024);

            add<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

            cudaMemcpy(C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            }
        """
        with open("./macro/cuda_macro.txt", "r") as f:
            macro = f.read()
            f.close()

        code = macro + code + template_code

        with open("./add_18_128.cu", mode="w") as f:
            f.write(code)
            f.close()

        success, output = run_test("./add_18_128.cu", "./unittest/add_test.py")
        print("[INFO]******************output2: ", output)
        return success

    open_set = []
    start_node = Node(start_state)
    heapq.heappush(open_set, (start_node.total_cost, start_node))

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)
        print("[INFO]***********current state: ", current_node.state)
        if check_file(current_node.state):
            return reconstruct_path(current_node)

        for action in actions:
            next_state, action_cost = apply_action(current_node.state, action)
            next_cost = current_node.cost + action_cost
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
    if compile_check(state):
        h_cost -= 20
    if "threadIdx.x" in state:
        h_cost -= 10
    if "blockIdx.x" in state:
        h_cost -= 10
    return h_cost


def apply_action(start_state, action):
    """
    Apply a specific refactoring action to the given C code state.

    Parameters:
    - start_state: A string representing the initial C code.
    - action: A string specifying the refactoring action to apply.

    Returns:
    - A tuple containing the new state of the code and the cost of the action.
    """
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
        if len(axises) < 1:
            return start_state, 100
        axis = random.choice(axises)
        name = random.choice(["threadIdx.x", "blockIdx.x"])
        if name in start_state:
            return start_state, 100
        state = loop_bind(start_state, loop_index=axis, thread_name=name)
        return state, 10

    elif action == "loop_split":
        axises = get_ajcent_loop(start_state)
        if len(axises) < 1:
            return start_state, 100
        axis = random.choice(axises)
        factors = [1024, 256]
        factor = random.choice(factors)
        state = loop_split(start_state, loop_index=axis, factor=factor)
        return state, 10

    else:
        raise RuntimeError("Cannot handle!")


if __name__ == "__main__":
    # 定义初始状态和目标状态
    start_state = """
    void add(float* output, float* input1, float* input2) {
        for (int i = 0; i < 18; i++) {
            for (int j = 0; j < 128; j++) {
                int index = i * 128 + j;
                output[index] = input1[index] + input2[index];
            }
        }
    }
    """

    # 执行 A* 搜索
    actions = ["loop_fuse", "loop_split", "loop_bind", "func_prefix"]

    # 定义可能的动作列表
    transformation_sequence = a_star_search(start_state, actions, heuristic)
