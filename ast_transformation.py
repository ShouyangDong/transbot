from pycparser import c_parser, c_ast, c_generator


class NodeTransformer(c_ast.NodeVisitor):
    """
    A node transformer that visits each node in an AST and applies transformations.

    Attributes:
    - None explicitly defined here, but subclasses may add attributes.
    """

    def generic_visit(self, node):
        """
        A generic visit method that is called for nodes that don't have a specific visit_<nodetype> method.

        This method iterates over all fields in the current node. If a field contains a list of nodes,
        it applies the transformation to each item in the list. If a field contains a single node, it applies
        the transformation to that node.

        Parameters:
        - node: The AST node to visit and potentially transform.

        Returns:
        - The original node, potentially with some of its fields transformed or replaced.
        """
        for field, old_value in iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, c_ast.Node):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, c_ast.Node):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, c_ast.Node):
                new_node = self.visit(old_value)
                setattr(node, field, new_node)
        return node


def iter_fields(node):
    """
    Iterate over all fields of a pycparser AST node.

    Parameters:
    - node: The AST node whose fields are to be iterated over.

    Yields:
    - A tuple containing the name of the field and the value of the field.
    """
    index = 0
    children = node.children()
    while index < len(children):
        name, child = children[index]
        try:
            bracket_index = name.index("[")
        except ValueError:
            yield name, child
            index += 1
        else:
            name = name[:bracket_index]
            child = getattr(node, name)
            index += len(child)
            yield name, child


class LoopFuseVisitor(c_ast.NodeVisitor):
    def __init__(self, axis_name_1, axis_name_2):
        self.axis_name_1 = axis_name_1
        self.axis_name_2 = axis_name_2
        self.extend = {}

    def visit_For(self, node):
        if node.init.decls[0].name == self.axis_name_1:
            compound_nested_node = node.stmt
            generator = c_generator.CGenerator()
            if isinstance(compound_nested_node, c_ast.Compound) and isinstance(
                compound_nested_node.block_items[0], c_ast.For
            ):
                nested_node = compound_nested_node.block_items[0]
                if nested_node.init.decls[0].name == self.axis_name_2:
                    self.extend[self.axis_name_2] = nested_node.cond.right.value
                    extend = int(node.cond.right.value) * int(
                        nested_node.cond.right.value
                    )
                    node.init.decls[0].name = (
                        "fuse_" + self.axis_name_1 + "_" + self.axis_name_2
                    )
                    node.init.decls[0].type.declname = (
                        "fuse_" + self.axis_name_1 + "_" + self.axis_name_2
                    )
                    node.cond.left.name = (
                        "fuse_" + self.axis_name_1 + "_" + self.axis_name_2
                    )
                    node.cond.right.value = extend
                    node.next.expr.name = (
                        "fuse_" + self.axis_name_1 + "_" + self.axis_name_2
                    )
                    # replace the loop index with new loop index
                    node.stmt = nested_node.stmt
                    self.visit(node.stmt)

    def visit_BinaryOp(self, node):
        if isinstance(node.left, c_ast.BinaryOp) and node.op == "+":
            if (
                node.left.op == "*"
                and node.left.left.name == self.axis_name_1
                and node.left.right.value == str(self.extend[self.axis_name_2])
                and node.right.name == self.axis_name_2
            ):
                node.left = c_ast.Constant(
                    "int", "fuse_" + self.axis_name_1 + "_" + self.axis_name_2
                )
                node.right = c_ast.Constant("int", 0)


class LoopSplitVisitor(c_ast.NodeVisitor):
    def __init__(self, axis_name, factor):
        self.axis_name = axis_name
        self.factor = factor

    def visit_For(self, node):
        # check the loop index
        if node.init.decls[0].name == self.axis_name:
            org_extent = int(node.cond.right.value)
            node.cond.right.value = self.factor
            self.visit(node.stmt)
            init_node = c_ast.Decl(
                name=self.axis_name + "_in",
                quals=[],
                align=[],
                storage=[],
                funcspec=[],
                type=c_ast.TypeDecl(
                    declname=self.axis_name + "_in",
                    quals=[],
                    align=None,
                    type=c_ast.IdentifierType(["int"]),
                ),
                init=c_ast.Constant("int", "0"),
                bitsize=None,
            )
            cond_node = c_ast.BinaryOp(
                node.cond.op,
                c_ast.ID(self.axis_name + "_in"),
                c_ast.Constant("int", node.cond.right.value),
            )
            next_node = c_ast.UnaryOp(node.next.op, c_ast.ID(self.axis_name + "_in"))

            inner_loop = c_ast.For(
                init=init_node, cond=cond_node, next=next_node, stmt=node.stmt
            )
            inner_loop = c_ast.Compound(block_items=[inner_loop])
            node.init = c_ast.Decl(
                name=self.axis_name + "_out",
                quals=[],
                align=[],
                storage=[],
                funcspec=[],
                type=c_ast.TypeDecl(
                    declname=self.axis_name + "_out",
                    quals=[],
                    align=None,
                    type=c_ast.IdentifierType(["int"]),
                ),
                init=c_ast.Constant("int", "0"),
                bitsize=None,
            )
            node.cond = c_ast.BinaryOp(
                node.cond.op,
                c_ast.ID(self.axis_name + "_out"),
                c_ast.Constant("int", str(org_extent // self.factor)),
            )
            node.next = c_ast.UnaryOp(node.next.op, c_ast.ID(self.axis_name + "_out"))
            node.stmt = inner_loop

    def visit_ID(self, node):
        # modify the aixs name inside stmt
        if node.name == self.axis_name:
            node.name = (
                self.axis_name
                + "_out"
                + " * "
                + str(self.factor)
                + " + "
                + self.axis_name
                + "_in"
            )


class LoopBindVisitor(NodeTransformer):
    def __init__(self, axis_name, thread_name):
        self.axis_name = axis_name
        self.thread_name = thread_name

    def visit_For(self, node):
        if node.init.decls[0].name == self.axis_name:
            node.init.decls[0].name = self.thread_name
            stmt = self.visit(node.stmt)
            # replace the for loop as condition node
            if_node = c_ast.If(
                cond=c_ast.BinaryOp(
                    op=node.cond.op,
                    left=c_ast.ID(self.thread_name),
                    right=node.cond.right,
                ),
                iftrue=c_ast.Compound(block_items=[stmt]),
                iffalse=None,
            )

            return if_node
        self.generic_visit(node)
        return node

    def visit_ID(self, node):
        if node.name == self.axis_name:
            return c_ast.ID(self.thread_name)
        return node


def loop_split(code, loop_index, factor=2):
    is_global_func = True if "__global__" in code else False
    code = code.replace("__global__ ", "")
    parser = c_parser.CParser()
    ast = parser.parse(code)
    generator = c_generator.CGenerator()
    visitor = LoopSplitVisitor(loop_index, factor=factor)
    visitor.visit(ast)
    code = generator.visit(ast)

    code = "__global__ " + code if is_global_func else code
    return code


def loop_fuse(code, loop_index1, loop_index2):
    is_global_func = True if "__global__" in code else False
    code = code.replace("__global__ ", "")

    parser = c_parser.CParser()
    ast = parser.parse(code)
    generator = c_generator.CGenerator()
    visitor = LoopFuseVisitor(loop_index1, loop_index2)
    visitor.visit(ast)
    code = generator.visit(ast)

    code = "__global__ " + code if is_global_func else code
    return code


def loop_bind(code, loop_index, thread_name):
    is_global_func = True if "__global__" in code else False
    code = code.replace("__global__ ", "")
    parser = c_parser.CParser()
    ast = parser.parse(code)
    generator = c_generator.CGenerator()
    visitor = LoopBindVisitor(loop_index, thread_name)
    visitor.visit(ast)
    code = generator.visit(ast)
    code = "__global__ " + code if is_global_func else code
    return code


if __name__ == "__main__":
    original_code = """
    void add_kernel(float* output, float* input1, float* input2) {
        for (int i = 0; i < 18; i++){
            for (int j = 0; j < 128; j++){
                int index = i * 128 + j;
                output[index] = input1[index] + input2[index];
            }
        }
    }
    """
    parser = c_parser.CParser()
    ast = parser.parse(original_code)
    bind_visitor = LoopBindVisitor("j", "threadIdx_x")
    bind_visitor.visit(ast)
    generator = c_generator.CGenerator()
    print(generator.visit(ast))
