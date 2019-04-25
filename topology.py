'''
Methods for calculating lower-dimensional persistent homology.
'''

import numpy as np


class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, n_vertices):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''

        self._parent = np.arange(n_vertices, dtype=int)

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistentHomologyCalculation:
    def __init__(self):
        pass

    def __call__(self, matrix):

        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        triu_indices = np.triu_indices_from(matrix)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights, kind='stable')

        # 1st dimension: 'source' vertex index of edge
        # 2nd dimension: 'target' vertex index of edge
        persistence_pairs = np.array((2, n_vertices - 1), dtype=int)

        for i, (edge_index, edge_weight) in \
                enumerate(zip(edge_indices, edge_weights[edge_indices])):

            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            if younger_component == older_component:
                continue

            elif younger_component > older_component:
                u, v = v, u

            uf.merge(u, v)
            persistence_pairs[0][i] = u
            persistence_pairs[1][i] = v

        return persistence_pairs


class PersistenceDiagramCalculator:
    '''
    Given a weighted graph, calculates a persistence diagram. The client
    can modify the filtration order and the vertex weight assignment.
    '''

    def __init__(self,
                 order='sublevel',
                 unpaired_value=None,
                 vertex_attribute=None):
        '''
        Initializes a new instance of the persistence diagram
        calculation class.

        :param order: Filtration order (ignored for now)
        :param unpaired_value: Value to use for unpaired vertices. If
        not specified the largest weight (sublevel set filtration) is
        used.
        :param vertex_attribute: Graph attribute to query for vertex
        values. If not specified, no vertex attributes will be used,
        and each vertex will be assigned a value of zero.
        '''

        self._order = order
        self._unpaired_value = unpaired_value
        self._vertex_attribute = vertex_attribute

        if self._order not in ['sublevel', 'superlevel']:
            raise RuntimeError(
                '''
                Unknown filtration order \"{}\"
                '''.format(self._order)
            )

    def fit_transform(self, graph):
        '''
        Applies a filtration to a graph and calculates its persistence
        diagram. The function will return the persistence diagram plus
        all edges that are involved in cycles.

        :param graph: Weighted graph whose persistence diagram will be
        calculated.

        :return: Tuple consisting of the persistence diagram, followed
        by a list of all edge indices that create a cycle.
        '''

        num_vertices = graph.vcount()
        uf = UnionFind(num_vertices)

        edge_weights = np.array(graph.es['weight'])   # All edge weights
        edge_indices = None                           # Ordering for filtration
        edge_indices_cycles = []                      # Edge indices of cycles

        if self._order == 'sublevel':
            edge_indices = np.argsort(edge_weights, kind='stable')
        elif self._order == 'superlevel':
            edge_indices = np.argsort(-edge_weights, kind='stable')

        assert edge_indices is not None

        # Will be filled during the iteration below. This will become
        # the return value of the function.
        pd = PersistenceDiagram()

        # Go over all edges and optionally create new points for the
        # persistence diagram.
        for edge_index, edge_weight in \
                zip(edge_indices, edge_weights[edge_indices]):
            u, v = graph.es[edge_index].tuple

            # Preliminary assignment of younger and older component. We
            # will check below whether this is actually correct, for it
            # is possible that u is actually the older one.
            younger = uf.find(u)
            older = uf.find(v)

            # Nothing to do here: the two components are already the
            # same
            if younger == older:
                edge_indices_cycles.append(edge_index)
                continue

            # Ensures that the older component precedes the younger one
            # in terms of its vertex index
            elif younger > older:
                u, v = v, u
                younger, older = older, younger

            vertex_weight = 0.0

            # Vertex attributes have been set, so we use them for the
            # persistence diagram creation below.
            if self._vertex_attribute:
                vertex_weight = graph.vs[self._vertex_attribute][younger]

            creation = vertex_weight    # x coordinate for persistence diagram
            destruction = edge_weight   # y coordinate for persistence diagram

            uf.merge(u, v)
            pd.append(creation, destruction, younger)

        # By default, use the largest (sublevel set) or lowest
        # (superlevel set) weight, unless the user specified a
        # different one.
        unpaired_value = edge_weights[edge_indices[-1]]
        if self._unpaired_value:
            unpaired_value = self._unpaired_value

        # Add tuples for every root component in the Union--Find data
        # structure. This ensures that multiple connected components
        # are handled correctly.
        for root in uf.roots():

            vertex_weight = 0.0

            # Vertex attributes have been set, so we use them for the
            # creation of the root tuple.
            if self._vertex_attribute:
                vertex_weight = graph.vs[self._vertex_attribute][root]

            creation = vertex_weight
            destruction = unpaired_value

            pd.append(creation, destruction, root)

            if pd.betti is not None:
                pd.betti = pd.betti + 1
            else:
                pd.betti = 1

        return pd, edge_indices_cycles


def assign_filtration_values(
        graph,
        attributes,
        order='sublevel',
        normalize=False):
    '''
    Given a vertex attribute of a graph, assigns filtration values as
    edge weights to the graph edges.

    :param graph: Graph to modify
    :param attribute: Attribute sequence to use for the filtration
    :param order: Order of filtration
    :param normalize: If set, normalizes according to filtration order

    :return: Graph with added edges
    '''

    selection_function = max if order == 'sublevel' else min

    if normalize:
        offset = np.max(attributes) if order == 'sublevel' \
                                    else np.min(attributes)
    else:
        offset = 1.0

    for edge in graph.es:
        source = edge.source
        target = edge.target

        source_weight = attributes[source] / offset
        target_weight = attributes[target] / offset
        edge_weight = selection_function(source_weight, target_weight)

        edge['weight'] = edge_weight

    return graph
