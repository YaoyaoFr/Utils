import numpy as np



class Tree:

    def __init__(self,
                 index: int,
                 max_depth: int,
                 adj_matrix: np.ndarray,
                 path: list = None,
                 ):
        self.index = index
        self.child = {}

        if path is None:
            path = []
        self.path = path

        if (len(path) + 1) > max_depth:
            return

        adj_row = adj_matrix[self.index]
        for child_index in range(len(adj_row)):
            if adj_row[child_index] == 0:
                continue

            if (child_index, self.index) not in path and (self.index, child_index) not in path:
                new_path = list(path)
                new_path.append((self.index, child_index))
                self.child[child_index] = Tree(index=child_index,
                                               max_depth=max_depth,
                                               path=new_path,
                                               adj_matrix=adj_matrix,
                                               )

    def roots_by_depth(self, depth: int):
        roots = []

        if len(self.path) == depth:
            roots.append(self)
            return roots

        for child in self.child:
            roots.extend(self.child[child].roots_by_depth(depth=depth))

        return roots

    def transform_matrix(self, depth: int, v_num: int):
        adj_matrix = np.zeros(shape=[v_num, v_num], dtype=int)
        roots = self.roots_by_depth(depth=depth)
        for root in roots:
            for child in root.child:
                adj_matrix[root.index, child] = 1

        return adj_matrix

    def print_tree(self):
        depth = len(self.path)
        pre_fix = ''.join([' ' for _ in range(depth)])
        print('{:s} Node {:d}'.format(pre_fix, self.index))
        for child in self.child:
            self.child[child].print_tree()


class Vertex:

    def __init__(self,
                 adj_matrix: np.ndarray,
                 root_index: int,
                 ):
        self.adj_matrix = adj_matrix
        self.root_index = root_index
        self.diffusion_tree = None
        self.v_num = np.size(adj_matrix, 0)
        self.max_depth = 0

    def diffuse(self, max_depth: int,
                ):
        self.diffusion_tree = Tree(index=self.root_index,
                                   adj_matrix=self.adj_matrix,
                                   max_depth=max_depth)
        self.max_depth = max_depth

    def adj_by_depth(self, depth: int):
        if self.diffusion_tree is None or self.max_depth < depth + 1:
            self.diffuse(max_depth=depth + 1)

        adj = self.diffusion_tree.transform_matrix(depth=depth, v_num=self.v_num)
        return adj

    def nodes_by_depth(self, depth: int):
        if self.diffusion_tree is None or self.max_depth < depth:
            self.diffuse(max_depth=depth)

        nodes = self.diffusion_tree.roots_by_depth(depth)
        return nodes

    def paths_by_depth(self,
                       depth: int):
        nodes = self.nodes_by_depth(depth=depth)
        paths = [node.path for node in nodes]

        return paths

    def graph_by_depth(self,
                       depth: int):
        paths = self.paths_by_depth(depth=depth)
        graph = np.zeros(shape=[1, self.v_num])
        for path in paths:
            graph[0, path[-1][1]] = 1
        return graph


    def print_diffusion_tree(self):
        self.diffusion_tree.print_tree()


class Graph:
    adj_matrix = None
    v_num = 0
    vertexes = []

    def __init__(self,
                 adj_matrix: np.ndarray,
                 v_num=None):
        if v_num is None:
            v_num = np.size(adj_matrix, 0)

        assert (v_num, v_num) == np.shape(adj_matrix)

        self.adj_matrix = adj_matrix
        self.v_num = v_num
        self.vertexes = [Vertex(adj_matrix=adj_matrix, root_index=i)
                         for i in range(self.v_num)]

    def diffuse(self, max_depth: int = 5):
        for node in self.vertexes:
            node.diffuse(max_depth=max_depth)

    def adj_by_depth(self, depth: int):
        adj_matrix = np.concatenate([np.expand_dims(vertex.adj_by_depth(depth=depth), axis=-1)
                                     for vertex in self.vertexes], axis=-1)
        return adj_matrix

    def graph_by_depth(self, depth: int):
        graph = np.concatenate([vertex.graph_by_depth(depth=depth) for vertex in self.vertexes], axis=0)
        return graph

