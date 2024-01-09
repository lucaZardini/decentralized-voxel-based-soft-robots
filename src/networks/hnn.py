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
                # print(self.hrules[l - 1][o][0])
                dw = self.eta * (self.hrules[l - 1][o][0][0] * self.weights[l - 1][o][0] * self.activations[l][o] +
                                 self.hrules[l - 1][o][0][1] * self.activations[l][o] +
                                 self.hrules[l - 1][o][0][2] * self.weights[l - 1][o][0] +
                                 self.hrules[l - 1][o][0][3])
                self.weights[l - 1][o][0] += dw
                for i in range(1, self.nodes[l - 1]):
                    # print(self.hrules[l - 1][o][i])
                    dw = self.eta * (
                            self.hrules[l - 1][o][i][0] * self.activations[l - 1][i - 1] * self.activations[l][o] +
                            self.hrules[l - 1][o][i][1] * self.activations[l][o] +
                            self.hrules[l - 1][o][i][2] * self.activations[l - 1][i - 1] +
                            self.hrules[l - 1][o][i][3])
                    self.weights[l - 1][o][i] += dw

    def get_prunable_weights(self):
        ws = []
        for i in range(self.nodes[0]):
            for o in range(self.nodes[0], self.total_number_of_nodes):
                if not (i == o or (i, o) in self.pruned_synapses):
                    ws.append(np.abs(self.weights[i, o]))

        for i in range(self.nodes[0], self.nodes[0] + self.nodes[1]):
            for o in range(self.nodes[0], self.nodes[0] + self.nodes[1]):
                if not (i == o or (i, o) in self.pruned_synapses):
                    ws.append(np.abs(self.weights[i, o]))

        for i in range(self.nodes[0], self.nodes[0] + self.nodes[1]):
            for o in range(self.nodes[0] + self.nodes[1], self.total_number_of_nodes):
                if not (i == o or (i, o) in self.pruned_synapses):
                    ws.append(np.abs(self.weights[i, o]))
        return ws

    def sanitize_weights(self):
        # clean impossible connections
        #   outputs to other nodes
        for o in range(self.nodes[0] + self.nodes[1], self.total_number_of_nodes):
            for n in range(self.total_number_of_nodes):
                self.weights[o, n] = 0
        #   incoming edges to input
        for n in range(self.total_number_of_nodes):
            for i in range(self.nodes[0]):
                self.weights[n, i] = 0

    def prune_weights(self, prune_ratio: float, folder: str = None):
        weights_prunable = self.get_prunable_weights()
        threshold = np.percentile(weights_prunable, prune_ratio)

        for i in range(self.total_number_of_nodes):
            for j in range(i, self.total_number_of_nodes):
                if np.abs(self.weights[i, j]) <= threshold:
                    self.weights[i, j] = 0.
                    self.pruned_synapses.add((i, j))

        self.sanitize_weights()
        dag, hc = self.dag(fold=folder)
        top_sort = nx.topological_sort(dag)
        self.top_sort = list(top_sort)
        self.cycle_history = hc

    def dag(self, fold=None):
        graph = nx.from_numpy_matrix(np.array(self.weights), create_using=nx.DiGraph)
        adj_matrix = nx.to_dict_of_lists(graph)
        if not fold is None:
            plt.clf()
            pos = self.nxpbiwthtaamfalaiwftb(graph)
            nx.draw(graph, pos=pos, with_labels=True, font_weight='bold')
            # print("saving")
            plt.savefig(fold + "_init.png")
        cycles = self._merge_equivalent_cycle(list(nx.simple_cycles(graph)))
        history_of_cycles = list()
        dag = None
        offset = 0
        cc = 1
        # print(adj_matrix)
        # print("-------")
        while len(cycles) != 0:
            history_of_cycles.append(dict())
            dag = nx.DiGraph()

            dag, cycles, history_of_cycles, cycling_nodes, max_fn_id = self._add_nodes(dag, graph, cycles,
                                                                                       history_of_cycles,
                                                                                       offset)
            offset += max_fn_id
            not_cycling_nodes = [n for n in list(graph) if n not in cycling_nodes]
            dag = self._add_edges(dag, adj_matrix, history_of_cycles, not_cycling_nodes)
            graph = dag.copy()
            # print("dddddddd")
            if not fold is None:
                plt.clf()
                pos = self.nxpbiwthtaamfalaiwftb(dag)
                nx.draw(dag, pos=pos, with_labels=True, font_weight='bold')
                # print("saving")

                plt.savefig(fold + "_" + str(cc) + ".png")
            cc += 1
            cycles = self._merge_equivalent_cycle(list(nx.simple_cycles(graph)))
            adj_matrix = nx.to_dict_of_lists(graph)  # nx.to_numpy_matrix(graph)
            # print(adj_matrix)
            # print("-------")
        if dag is None:
            dag = graph
        if not fold is None:
            with open(fold + "_history.txt", "w") as f:
                for i in range(len(history_of_cycles)):
                    for k in history_of_cycles[i].keys():
                        f.write(str(i) + ";" + str(k) + ";" + str(history_of_cycles[i][k]) + "\n")
        if not fold is None:
            plt.clf()
            pos = self.nxpbiwthtaamfalaiwftb(dag)
            nx.draw(dag, pos=pos, with_labels=True, font_weight='bold')
            # print("saving")
            plt.savefig(fold + "_final.png")
        return dag, history_of_cycles

    def _get_in_edges(self, adj_m, node):
        return [i for i in adj_m.keys() if node in adj_m[i]]

    def _get_out_edges(self, adj_m, node):
        return adj_m[node]

    def _add_edges(self, dag, adj_m, history_of_cycles, not_cycling_nodes):
        for n in not_cycling_nodes:
            inc = self._get_in_edges(adj_m, n)
            outc = self._get_out_edges(adj_m, n)
            for i in inc:
                if i in not_cycling_nodes:
                    if not i == n:
                        dag.add_edge(i, n)
                else:
                    fnid = None
                    for id in history_of_cycles[-1].keys():
                        fnid = id if i in history_of_cycles[-1][id] else fnid

                    if not fnid == n:
                        dag.add_edge(fnid, n)

            for o in outc:
                if o in not_cycling_nodes:
                    if o in not_cycling_nodes:
                        if not n == o:
                            dag.add_edge(n, o)
                    else:
                        fnid = None
                        for id in history_of_cycles[-1].keys():
                            fnid = id if o in history_of_cycles[-1][id] else fnid
                        if not n == fnid:
                            dag.add_edge(n, fnid)

        for fake_node in history_of_cycles[-1].keys():
            for n in history_of_cycles[-1][fake_node]:
                inc = self._get_in_edges(adj_m, n)
                outc = self._get_out_edges(adj_m, n)
                for i in inc:
                    if i in not_cycling_nodes:
                        if not i == fake_node:
                            dag.add_edge(i, fake_node)
                    else:
                        fnid = None
                        for id in history_of_cycles[-1].keys():
                            fnid = id if i in history_of_cycles[-1][id] else fnid
                        if not fnid == fake_node:
                            dag.add_edge(fnid, fake_node)
                for o in outc:
                    if o in not_cycling_nodes:
                        if not o == fake_node:
                            dag.add_edge(fake_node, o)
                    else:
                        fnid = None
                        for id in history_of_cycles[-1].keys():
                            fnid = id if o in history_of_cycles[-1][id] else fnid
                        if not fnid == fake_node:
                            dag.add_edge(fake_node, fnid)
        return dag

    def _merge_equivalent_cycle(self, cycles):
        merged_cycles = list()
        if len(cycles) > 0:
            merged_cycles.append(cycles[0])
            for i in range(1, len(cycles)):
                switching = []
                for j in range(len(merged_cycles)):
                    tmp = set(*[cycles[i] + merged_cycles[j]])
                    if len(tmp) != len(merged_cycles[j]):
                        if len(tmp) == len(cycles[i]):
                            switching.append(j)
                        else:
                            switching.append(-1)
                    else:
                        switching.append(-2)
                if not -2 in switching:
                    if not -1 in switching:
                        merged_cycles[max(switching)] = cycles[i]
                    else:
                        merged_cycles.append(cycles[i])

        return merged_cycles

    def _add_nodes(self, dag, old_dag, cycles, history_of_cycles, offset):
        max_fn_id = 0
        cycling_nodes = set()
        old_nodes = list(old_dag)
        for cycle in cycles:
            for node in cycle:
                cycling_nodes.add(node)
        # inputs have no cycles
        for i in range(self.nodes[0]):
            dag.add_node(i)
        # outputs neither
        for o in range(self.nodes[0] + self.nodes[1], self.total_number_of_nodes):
            dag.add_node(o)

        # inner nodes can have cycle
        for n in range(self.nodes[0], self.nodes[0] + self.nodes[1]):
            if n not in cycling_nodes:
                if n in old_nodes:
                    dag.add_node(n)
            else:
                fns = [i for i in range(len(cycles)) if n in cycles[i]]
                for fn in fns:
                    if not dag.has_node(self.total_number_of_nodes + fn + offset):
                        dag.add_node(self.total_number_of_nodes + fn + offset)
                        history_of_cycles[-1][self.total_number_of_nodes + fn + offset] = cycles[fn][:]
                        max_fn_id += 1

        # and also fake nodes that hide cycle can have cycle
        for n in old_nodes:
            if not n in cycling_nodes:
                dag.add_node(n)
            else:
                fns = [i for i in range(len(cycles)) if n in cycles[i]]
                for fn in fns:
                    if not dag.has_node(self.total_number_of_nodes + fn + offset):
                        dag.add_node(self.total_number_of_nodes + fn + offset)
                        history_of_cycles[-1][self.total_number_of_nodes + fn + offset] = cycles[fn][:]
                        max_fn_id += 1

        return dag, cycles, history_of_cycles, cycling_nodes, max_fn_id
