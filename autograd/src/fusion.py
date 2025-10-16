import networkx as nx

FUSED_OP_PREFIX = ""


class ChainedFuser:
    PATTERNS = [
        [("view",), ("view",)],
        [("matmul",), ("add",)],
        [("add",), ("sigmoid", "leaky_relu")],
        [("matmul",), ("sigmoid", "leaky_relu")],
        [("matmul+add",), ("sigmoid", "leaky_relu")],
        # ...
    ]

    @classmethod
    def chained_fuse(
        cls,
        graph: nx.DiGraph,
        pattern: list[tuple[str, ...]],
        fused_op_prefix=FUSED_OP_PREFIX,
    ):
        """ Fuses consecutive operators in a computation graph that match a given pattern.
            Traverses the graph in topological order and detects linear chains of nodes
            matching the pattern (e.g., [("matmul",), ("add",), ("relu",)]).
            Each matched subchain is replaced with a single fused node while preserving
            all input/output edges and metadata.

            Params:
                graph : nx.DiGraph
                    Computation graph where nodes are ops (e.g., "matmul_0") and edges
                    carry tensor shapes.
                pattern : list[tuple[str, ...]]
                    List of operator prefixes describing the fuse chain.
                    Tuples allow alternatives, and can indecate previously fused operators
                    with "+".
                fused_op_prefix : str, optional
                    Prefix for fused node names. Default is "fused".
        """

        def match_pattern(pattern: str, op_name: str):
            flags = []
            sub_patterns = pattern.split("+")
            fused_ops = op_name.split("+")
            if len(sub_patterns) != len(fused_ops):
                return
            for sub_pattern, fused_op in zip(sub_patterns, fused_ops):
                flags.append(sub_pattern in fused_op)
            return all(flags)

        fused_count = 0
        for op_name in list(nx.topological_sort(graph)):
            if not any(match_pattern(p, op_name) for p in pattern[0]):
                continue

            chain = [op_name]
            cur = op_name
            matched = True

            # follow the successors to match the pattern
            for step in pattern[1:]:
                succs = list(graph.successors(cur))
                if len(succs) != 1:
                    matched = False
                    break
                nxt_op = succs[0]
                if not any(match_pattern(p, nxt_op) for p in step):
                    matched = False
                    break
                chain.append(nxt_op)
                cur = nxt_op

            if not matched:
                continue

            # Begin fusion
            fused_node = f"{fused_op_prefix}_{'+'.join(chain)}".lstrip("_")
            graph.add_node(fused_node)

            # process first chain predecessors
            for pred in list(graph.predecessors(chain[0])):
                name: str = graph.edges[(pred, chain[0])].get("name")
                graph.remove_edge(pred, chain[0])
                graph.add_edge(pred, fused_node, name=name)

            # process last chain successors
            for succ in list(graph.successors(chain[-1])):
                name = graph.edges[(chain[-1], succ)].get("name")
                graph.remove_edge(chain[-1], succ)
                graph.add_edge(fused_node, succ, name=name)

            # process internal chain predecessors & remove original edges
            for i in range(len(chain)-1):
                for pred in list(graph.predecessors(chain[i + 1])):
                    if pred != chain[i]:
                        name = graph.edges[(pred, chain[i + 1])].get("name")
                        graph.remove_edge(pred, chain[i + 1])
                        graph.add_edge(pred, fused_node, name=name)
                
                if graph.has_edge(chain[i], chain[i+1]):
                    graph.remove_edge(chain[i], chain[i+1])
            for n in chain:
                if n != fused_node and graph.has_node(n):
                    graph.remove_node(n)

            fused_count += 1

        if fused_count:
            print(f"Fused {fused_count} pattern(s) for {pattern}")
        return graph
    
    @classmethod
    def fuse(cls, graph: nx.DiGraph, fused_op_prefix: str) -> nx.DiGraph:
        for pattern in cls.PATTERNS:
            graph = cls.chained_fuse(
                graph,
                pattern=pattern,
                fused_op_prefix=fused_op_prefix,
            )
        return graph


class GraphFuser:
    fusers = [
        ChainedFuser(),
    ]
    
    @classmethod
    def fuse(cls, graph: nx.DiGraph, fused_op_prefix: str = FUSED_OP_PREFIX) -> nx.DiGraph:
        for fuser in cls.fusers:
            graph = fuser.__class__.fuse(graph, fused_op_prefix)
        return graph


if __name__ == "__main__":
    """ Example of subgraph fusion (run on Jupyter kernel to see visualization) """

    import networkx as nx
    from IPython.display import display

    from autograd.src.core import ComputationalGraph
    from autograd.src.fusion import GraphFuser

    G = nx.DiGraph()
    G.add_edge("Non-Param_0", "matmul_0", name=[16, 4])
    G.add_edge("Param_0", "matmul_0", name=[4, 64])
    G.add_edge("Param_1", "add_0", name=[64])
    G.add_edge("matmul_0", "add_0", name=[16, 64])
    G.add_edge("add_0", "sigmoid_0", name=[16, 64])
    G.add_edge("sigmoid_0", "matmul_1", name=[16, 64])
    G.add_edge("Param_2", "matmul_1", name=[16, 64])

    graph = ComputationalGraph(G)
    display(graph.draw())

    graph_opt = GraphFuser.fuse(G)

    display(graph.draw())
