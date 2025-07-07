from pygraphviz import AGraph


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)


def register_hooks(var):
    fn_dict = {}

    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input

        fn.register_hook(register_grad)

    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return grad_output.isnan().any()  # or (grad_output.abs() >= 1e6).any()

    def size_to_str(size):
        return "(" + ", ".join(map(str, size)) + ")"

    def make_dot():
        # Use dictionary unpacking to define graph and node attributes
        dot = AGraph(
            strict=False,
            directed=True,
            node_attr={
                "style": "filled",
                "shape": "box",
                "align": "left",
                "fontsize": "12",
                "ranksep": "0.1",
                "height": "0.2",
            },
            graph_attr={"size": "12,12"},
        )

        def build_graph(fn):
            if hasattr(fn, "variable"):  # Leaf tensor (i.e., GradAccumulator)
                u = fn.variable
                node_name = f"Variable\n{size_to_str(u.size())}"
                dot.add_node(str(id(u)), label=node_name, fillcolor="lightblue")
            else:
                # Gracefully skip gradient check if no grad is available
                fillcolor = "white"
                if fn in fn_dict:
                    if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                        fillcolor = "red"
                dot.add_node(str(id(fn)), label=type(fn).__name__, fillcolor=fillcolor)

            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    src_id = str(id(getattr(next_fn, "variable", next_fn)))
                    dst_id = str(id(fn))
                    dot.add_edge(src_id, dst_id)


        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot
