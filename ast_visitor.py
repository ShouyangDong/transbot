from pycparser import c_parser, c_ast, c_generator


class LoopVariableVisitor(c_ast.NodeVisitor):
    """
    Custom visitor designed to extract loop variable names from C code.
    """

    def __init__(self):
        self.loop_index = []

    def visit_For(self, node):
        """
        Visit For loop nodes to extract loop variables.

        Parameters:
        node: A node of type c_ast.For representing a 'for' loop.
        """
        for decl in node.init:
            if isinstance(decl, c_ast.Decl):  # 检查是否是声明
                self.loop_index.append(decl.name)

        self.generic_visit(node)


def get_ajcent_loop(code):
    """
    Parses C code and extracts the variables of adjacent loops.

    Parameters:
    code: A string containing C code.

    Returns:
    A list of names of the loop variables.
    """
    # Check if the code contains a GPU kernel function declaration
    is_global_func = "__global__" in code
    if is_global_func:
        # Clean up the kernel function declaration if it exists
        code = code.replace("__global__ ", "")
    code = code.replace("threadIdx.x ", "threadIdx_x")
    parser = c_parser.CParser()
    ast = parser.parse(code)
    visitor = LoopVariableVisitor()
    visitor.visit(ast)
    return visitor.loop_index


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
    indices = get_ajcent_loop(start_state)
    print(indices)
