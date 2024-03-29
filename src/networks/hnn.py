import json

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class NN:
    def __init__(self, nodes: list):
        self.nodes = nodes
        self.activations = [[0 for i in range(node)] for node in nodes]
        self.nweights = sum([self.nodes[i] * self.nodes[i + 1] for i in
                             range(len(self.nodes) - 1)])  # nodes[0]*nodes[1]+nodes[1]*nodes[2]+nodes[2]*nodes[3]

        self.weights = [[] for _ in range(len(self.nodes) - 1)]

    def activate(self, inputs):
        self.activations[0] = [np.tanh(x) for x in inputs]
        for i in range(1, len(self.nodes)):
            self.activations[i] = [0. for _ in range(self.nodes[i])]
            for j in range(self.nodes[i]):
                sum = 0  # self.weights[i - 1][j][0]
                for k in range(self.nodes[i - 1]):
                    sum += self.activations[i - 1][k - 1] * self.weights[i - 1][j][k]
                self.activations[i][j] = np.tanh(sum)
        return np.array(self.activations[-1])

    def set_weights(self, weights):
        # self.weights = [[] for _ in range(len(self.nodes) - 1)]
        c = 0
        for i in range(1, len(self.nodes)):
            self.weights[i - 1] = [[0 for _ in range(self.nodes[i - 1])] for __ in range(self.nodes[i])]
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    self.weights[i - 1][j][k] = weights[c]
                    c += 1
        # print(c)

    def get_list_weights(self):
        wghts = []
        for i in range(1, len(self.nodes)):
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    wghts.append(np.abs(self.weights[i - 1][j][k]))
        return wghts

    def nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(self, G):
        pos = {}
        nodes_G = list(G)
        input_space = 1.75 / self.nodes[0]
        output_space = 1.75 / self.nodes[-1]

        for i in range(self.nodes[0]):
            pos[i] = np.array([-1., i * input_space])

        c = 0
        for i in range(self.nodes[0] + self.nodes[1], sum(self.nodes)):
            pos[i] = np.array([1, c * output_space])
            c += 1

        center_node = []
        for n in nodes_G:
            if not n in pos:
                center_node.append(n)

        center_space = 1.75 / len(center_node)
        for i in range(len(center_node)):
            pos[center_node[i]] = np.array([0, i * center_space])
        return pos

    def nxpbiwthtaamfalaiwftb(self, G):
        return self.nx_pos_because_it_was_too_hard_to_add_a_multipartite_from_a_list_as_it_works_for_the_bipartite(G)

    def nn_prune_weights(self, prune_ratio, fold=None):
        wghts_abs = self.get_list_weights()
        thr = np.percentile(wghts_abs, prune_ratio)
        for i in range(1, len(self.nodes)):
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    if np.abs(self.weights[i - 1][j][k]) <= thr:
                        self.weights[i - 1][j][k] = 0.
        if fold is not None:
            mat = self.from_list_to_matrix()
            graph = nx.from_numpy_matrix(mat, create_using=nx.DiGraph)
            plt.clf()
            pos = self.nxpbiwthtaamfalaiwftb(graph)
            nx.draw(graph, pos=pos, with_labels=True, font_weight='bold')
            # print("saving")
            plt.savefig(fold + "_init.png")

    def from_list_to_matrix(self):
        matrix = np.zeros((sum(self.nodes), sum(self.nodes)))
        # set inputs
        for i in range(1, len(self.nodes)):
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    ix = k + sum(self.nodes[:i - 1])
                    ox = j + (sum(self.nodes[:i]))
                    matrix[ix][ox] = self.weights[i - 1][j][k]
        return matrix


class HNN(NN):
    def __init__(self, nodes, eta=0.1):
        super().__init__(nodes)
        self.hrules = [[[0, 0, 0, 0] for i in range(node)] for node in nodes]
        self.eta = eta
        self.set_weights([0 for _ in range(self.nweights)])
        self.total_number_of_nodes = sum(self.nodes)
        self.pruned_synapses = set()

    def set_hrules(self, hrules):
        self.hrules = [[] for _ in range(len(self.nodes) - 1)]
        c = 0
        for i in range(1, len(self.nodes)):
            self.hrules[i - 1] = [[0 for a in range(self.nodes[i - 1])] for b in range(self.nodes[i])]
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    self.hrules[i - 1][j][k] = [hrules[c + i] for i in range(4)]
                    c += 4

    def update_weights(self):
        for l in range(1, len(self.nodes)):
            for o in range(self.nodes[l]):
                if not (l - 1, o, 0) in self.pruned_synapses:
                    # print(self.hrules[l - 1][o][0])
                    dw = self.eta * (self.hrules[l - 1][o][0][0] * self.weights[l - 1][o][0] * self.activations[l][o] +
                                     self.hrules[l - 1][o][0][1] * self.activations[l][o] +
                                     self.hrules[l - 1][o][0][2] * self.weights[l - 1][o][0] +
                                     self.hrules[l - 1][o][0][3])
                    self.weights[l - 1][o][0] += dw
                for i in range(1, self.nodes[l - 1]):
                    if not (l - 1, o, i) in self.pruned_synapses:
                    # print(self.hrules[l - 1][o][i])
                        dw = self.eta * (
                                self.hrules[l - 1][o][i][0] * self.activations[l - 1][i - 1] * self.activations[l][o] +
                                self.hrules[l - 1][o][i][1] * self.activations[l][o] +
                                self.hrules[l - 1][o][i][2] * self.activations[l - 1][i - 1] +
                                self.hrules[l - 1][o][i][3])
                        self.weights[l - 1][o][i] += dw

    def get_prunable_weights(self):
        wghts = []
        for i in range(1, len(self.nodes)):
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    if not (i - 1, j, k) in self.pruned_synapses:
                        wghts.append(np.abs(self.weights[i - 1][j][k]))
        return wghts

    def prune_weights(self, prune_ratio: float, folder: str = None, prune_time: int = None, voxel_id: int = None):
        weights_prunable = self.get_prunable_weights()
        threshold = np.percentile(weights_prunable, prune_ratio)
        for i in range(1, len(self.nodes)):
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    if np.abs(self.weights[i - 1][j][k]) <= threshold:
                        self.weights[i - 1][j][k] = 0.
                        self.pruned_synapses.add((i - 1, j, k))
        self.dag(fold=folder, voxel_id=voxel_id, prune_time=prune_time, status="final")

    def from_list_to_matrix(self):
        matrix = np.zeros((sum(self.nodes), sum(self.nodes)))
        # set inputs
        for i in range(1, len(self.nodes)):
            for j in range(self.nodes[i]):
                for k in range(self.nodes[i - 1]):
                    ix = k + sum(self.nodes[:i - 1])
                    ox = j + (sum(self.nodes[:i]))
                    matrix[ix][ox] = self.weights[i - 1][j][k]
        return matrix

    def dag(self, fold=None, voxel_id: int = None, prune_time: int = None, status: str = "init"):
        mat = self.from_list_to_matrix()
        graph = nx.from_numpy_matrix(mat, create_using=nx.DiGraph)
        if fold is not None and voxel_id is not None:
            plt.clf()
            pos = self.nxpbiwthtaamfalaiwftb(graph)
            nx.draw(graph, pos=pos, with_labels=True, font_weight='bold')
            plt.savefig(fold + f"/prune_{prune_time}_voxel_{voxel_id}_{status}.png")

            inputs_mapping = self.get_list_of_connections(graph._adj)
            with open(fold + f"/prune_{prune_time}_voxel_{voxel_id}_{status}.json", "w") as f:
                json.dump(inputs_mapping, f)

    def get_list_of_connections(self, adj: dict) -> dict:
        outputs = self.nodes[-1]
        inputs = self.nodes[0]
        hidden = self.total_number_of_nodes - outputs - inputs
        output_connections = {}
        output_input = {}
        input_adj = {}
        for i in range(self.total_number_of_nodes - outputs, self.total_number_of_nodes):
            output_connections[i] = []
            output_input[i] = []
        for i in range(inputs + hidden):
            input_adj[i] = adj[i].keys()

        for i in range(inputs, inputs + hidden):
            for output in range(inputs + hidden, self.total_number_of_nodes):
                if output in input_adj[i]:
                    output_connections[output].append(i)
        if hidden != 0:
            for i in range(inputs):
                for output in range(inputs + hidden, self.total_number_of_nodes):
                    for hidden_node in output_connections[output]:
                        if hidden_node in input_adj[i]:
                            output_input[output].append(i)
        else:
            for i in range(inputs):
                for output in range(self.total_number_of_nodes - outputs, self.total_number_of_nodes):
                    if output in input_adj[i]:
                        output_input[output].append(i)
        for key, value in output_input.items():
            value = list(set(value))
            output_input[key] = value
        return output_input
