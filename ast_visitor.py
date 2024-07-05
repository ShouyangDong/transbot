from pycparser import c_parser, c_ast, c_generator


class LoopVariableVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.loop_index = []

    def visit_For(self, node):
        # 假设循环的初始化部分是变量声明
        for decl in node.init:
            if isinstance(decl, c_ast.Decl):  # 检查是否是声明
                self.loop_index.append(decl.name)

        self.generic_visit(node)


def get_ajcent_loop(code):
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
