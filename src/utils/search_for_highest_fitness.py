import copy
import json

generation_pruned = [15, 21, 24, 25, 27]
update_pruned = [3, 5, 8]
prune_ratio = [20, 40, 60, 80]
env = "soft_bridge"
path = f"../data/hebbian_parameters/{env}/worm/.robot_fitness_"


class PruneInfo:

    def __init__(self, generation, prune_ratio, update, fitness):
        self.generation = generation
        self.prune_ratio = prune_ratio
        self.update = update
        self.fitness = fitness

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness


prune_infos = []
for generation in generation_pruned:
    for update in update_pruned:
        for ratio in prune_ratio:
            my_path = copy.deepcopy(path)
            my_path += f"{generation}_pruned_{update}_ratio_{ratio}.json"
            with open(my_path, "r") as f:
                data = json.load(f)
                prune_infos.append(PruneInfo(generation, ratio, update, data["fitness"]))

for generation in generation_pruned:
    my_path = copy.deepcopy(path)
    my_path += f"{generation}_pruned_no_pruning.json"
    with open(my_path, "r") as f:
        data = json.load(f)
        prune_infos.append(PruneInfo(generation, 0, 0, data["fitness"]))


def best_individual_per_pruned_ratio(pruned_ratio: int):
    best_prune_infos = sorted([prune_info for prune_info in prune_infos if prune_info.prune_ratio == pruned_ratio], reverse=True)
    # for best_prune_info in best_prune_infos:
    #     print(f"Generation: {best_prune_info.generation}, prune ratio: {best_prune_info.prune_ratio}, update: {best_prune_info.update}, fitness: {best_prune_info.fitness}")
    print(f"Generation: {best_prune_infos[0].generation}, prune ratio: {best_prune_infos[0].prune_ratio}, update: {best_prune_infos[0].update}, fitness: {best_prune_infos[0].fitness}")
    return best_prune_infos[0]


def best_individual_per_timestep_to_be_pruned(timestep_to_be_pruned: int):
    best_prune_infos = sorted([prune_info for prune_info in prune_infos if prune_info.update == timestep_to_be_pruned], reverse=True)
    # for best_prune_info in best_prune_infos:
    #     print(f"Generation: {best_prune_info.generation}, prune ratio: {best_prune_info.prune_ratio}, update: {best_prune_info.update}, fitness: {best_prune_info.fitness}")
    print(f"Generation: {best_prune_infos[0].generation}, prune ratio: {best_prune_infos[0].prune_ratio}, update: {best_prune_infos[0].update}, fitness: {best_prune_infos[0].fitness}")
    return best_prune_infos[0]


best_individual_per_pruned_ratio(20)
best_individual_per_pruned_ratio(40)
best_individual_per_pruned_ratio(60)
best_individual_per_pruned_ratio(80)


best_individual_per_timestep_to_be_pruned(3) 
best_individual_per_timestep_to_be_pruned(5)
best_individual_per_timestep_to_be_pruned(8)