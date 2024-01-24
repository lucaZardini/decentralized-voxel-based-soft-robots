import copy
import json

import matplotlib.pyplot as plt

generation_pruned = [15, 21, 24, 25, 27]
env = "soft_bridge"
path = f"../data/hebbian_parameters/{env}/worm/.robot_fitness_"


def box_plot_x_axis_pruned(name: str, all_when_pruned: bool = True, when_pruned: int = None):
    global path
    x_values = ["0%", "20%", "40%", "60%", "80%"]
    x_values_to_iterate = [0, 20, 40, 60, 80]
    y_values = []
    if all_when_pruned:
        for i in range(len(x_values)):
            y_values.append([])
        for generation in generation_pruned:
            for index, i in enumerate(x_values_to_iterate):
                if i == 0:
                    my_path = copy.deepcopy(path)
                    my_path += f"{generation}_pruned_no_pruning.json"
                    with open(my_path, "r") as f:
                        data = json.load(f)
                        y_values[index].append(data["fitness"])
                else:
                    for update_pruned in [3, 5, 8]:
                        my_path = copy.deepcopy(path)
                        my_path += f"{generation}_pruned_{update_pruned}_ratio_{i}.json"
                        with open(my_path, "r") as f:
                            data = json.load(f)
                            y_values[index].append(data["fitness"])
    else:
        for i in range(len(x_values)):
            y_values.append([])
        for generation in generation_pruned:
            for index, i in enumerate(x_values_to_iterate):
                if i == 0:
                    my_path = copy.deepcopy(path)
                    my_path += f"{generation}_pruned_no_pruning.json"
                    with open(my_path, "r") as f:
                        data = json.load(f)
                        y_values[index].append(data["fitness"])
                else:
                    my_path = copy.deepcopy(path)
                    my_path += f"{generation}_pruned_{when_pruned}_ratio_{i}.json"
                    with open(my_path, "r") as f:
                        data = json.load(f)
                        y_values[index].append(data["fitness"])
    plt.boxplot(y_values, labels=x_values)

    # Set labels and title
    plt.xlabel('Prune ratio')
    plt.ylabel('Fitness')

    plt.savefig(name)
    # Show the plot
    plt.show()


def box_plot_x_axis_when_pruned(name: str, all_pruning_ratios: bool = True, when_pruned: int = None):
    x_values = [3, 5, 8, "Never"]
    global path
    y_values = []
    for i in range(len(x_values)):
        y_values.append([])
    if all_pruning_ratios:
        for generation in generation_pruned:
            for index, i in enumerate(x_values):
                if i == "Never":
                    my_path = copy.deepcopy(path)
                    my_path += f"{generation}_pruned_no_pruning.json"
                    with open(my_path, "r") as f:
                        data = json.load(f)
                        y_values[index].append(data["fitness"])
                else:
                    for prune_ratio in [20, 40, 60, 80]:
                        my_path = copy.deepcopy(path)
                        my_path += f"{generation}_pruned_{i}_ratio_{prune_ratio}.json"
                        with open(my_path, "r") as f:
                            data = json.load(f)
                            y_values[index].append(data["fitness"])
    else:
        for generation in generation_pruned:
            for index, i in enumerate(x_values):
                if i == "Never":
                    my_path = copy.deepcopy(path)
                    my_path += f"{generation}_pruned_no_pruning.json"
                    with open(my_path, "r") as f:
                        data = json.load(f)
                        y_values[index].append(data["fitness"])
                else:
                    my_path = copy.deepcopy(path)
                    my_path += f"{generation}_pruned_{i}_ratio_{when_pruned}.json"
                    with open(my_path, "r") as f:
                        data = json.load(f)
                        y_values[index].append(data["fitness"])

    plt.boxplot(y_values, labels=x_values)

    # Set labels and title
    plt.xlabel('Weights update time')
    plt.ylabel('Fitness')

    plt.savefig(name)
    # Show the plot
    plt.show()


box_plot_x_axis_pruned(name="pruned_all.png")
box_plot_x_axis_pruned(name="pruned_3.png", all_when_pruned=False, when_pruned=3)
box_plot_x_axis_when_pruned(name="prune_ratio_all.png")
box_plot_x_axis_when_pruned(name="prune_ratio_80.png", all_pruning_ratios=False, when_pruned=80)
