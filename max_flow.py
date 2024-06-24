"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pickle

from create_graph import EDGES_KEY, FILE, NODE_COUNT_KEY

import cvxpy as cp


# An object oriented max-flow problem.
class Edge:
    """ An undirected, capacity limited edge. """

    def __init__(self, capacity, from_node, to_node) -> None:
        self.capacity = capacity
        self.flow = cp.Variable(nonneg=True)
        self.from_node = from_node.name
        self.to_node = to_node.name
        self.connect(from_node,to_node)

    # Connects two nodes via the edge.
    def connect(self, in_node, out_node):
        in_node.edge_flows.append(-self.flow)
        out_node.edge_flows.append(self.flow)

    # Returns the edge's internal constraints.
    def constraints(self):
        return [self.flow <= self.capacity]
    
    def __str__(self) -> str:
        return f'Flow from {self.from_node} to {self.to_node}: {self.flow.value} <= capacity {self.capacity}'


class Node:
    """ A node with accumulation. """

    def __init__(self, name='', accumulation: float = 0.0) -> None:
        self.name = name
        self.accumulation = accumulation
        self.edge_flows = []

    # Returns the node's internal constraints.
    def constraints(self):
        return [cp.sum([f for f in self.edge_flows]) == self.accumulation]
