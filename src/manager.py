import json
from multiprocessing import Pool
from typing import List, Any, Optional

import numpy as np
from evogym import EvoViewer

from environments.environment import EnvironmentType, EnvironmentManager
from evolutionary_algorithm.evo_alg import EvoAlgoManager, EvoAlgoType
from networks.type import NetworkType
from robots.robot import RobotType, RobotManager, Robot
from utils import NpEncoder


def initialize_robot(robot_type: RobotType, structure_path: str, random_structure: bool,
                     raise_error_in_case_of_loading_structure_path: bool,
                     network_type: NetworkType, nodes: List[int], eta: float = 0.1):
    return RobotManager.get_robot(robot_type, structure_path, random_structure,
                                  raise_error_in_case_of_loading_structure_path, nodes, network_type, eta)


def initialize_environment(environment_type: EnvironmentType, robot: Robot):
    return EnvironmentManager.get_environment(environment_type, robot.structure, robot.connections)


class Manager:

    def __init__(self,
                 robot_type: RobotType, structure_path: str, random_structure: bool,
                 raise_error_in_case_of_loading_structure_path: bool,
                 environment_type: EnvironmentType,
                 network_type: NetworkType, nodes: List[int], weight_path: str,
                 evo_algo_type: EvoAlgoType, offsprings: int, population_size: int, sigma: float,
                 eta: float = 0.1):
        self.structure_path = structure_path
        self.random_structure = random_structure
        self.offsprings = offsprings
        self.environment_type = environment_type
        self.robot_type = robot_type
        self.network_type = network_type
        self.nodes = nodes
        self.eta = eta
        self.weight_path = weight_path
        self.raise_error_in_case_of_loading_structure_path = raise_error_in_case_of_loading_structure_path
        self.robot = initialize_robot(self.robot_type, self.structure_path, self.random_structure, self.raise_error_in_case_of_loading_structure_path,
                                      self.network_type, self.nodes, self.eta)
        self.evolutionary_algorithm = EvoAlgoManager.get_evolutionary_algorithm(evo_algo_type, self.robot.parameters_number, offsprings, population_size, sigma)

    def train(self, generations: int, max_steps: int, weight_update_steps: int,
              multi_processing: bool = False, display: bool = False):
        best_individuals = {}
        for i in range(generations):
            candidates = self.evolutionary_algorithm.get_candidates()
            if multi_processing:
                fitnesses = self.run_multi_processing_simulations(candidates, max_steps)
            else:
                fitnesses = self.run_single_processing_simulations(candidates, max_steps, weight_update_steps, display)
            self.evolutionary_algorithm.update(candidates, fitnesses)
            best_fitness_index = np.argmin(fitnesses)
            candidate = candidates[best_fitness_index]
            best_individuals[candidate] = fitnesses[best_fitness_index]
            self.save_individual(candidate, i, fitnesses[best_fitness_index])
        # Get best 5 individuals
        best_individuals = sorted(best_individuals.items(), key=lambda x: x[1])[:5]
        self.save_individual(best_individuals, None)

    def save_individual(self, candidate: Any, generation: Optional[int], fitness: float = None):
        if generation is not None:
            file_path = self.weight_path + f'_{generation}.json'
            with open(file_path, 'w') as f:
                json.dump({"candidate": candidate, "fitness": fitness}, f, indent=4, cls=NpEncoder)
        else:
            file_path = self.weight_path + "_best.json"
            with open(file_path, 'w') as f:
                json.dump({"candidates": json.dumps(candidate)}, f, indent=4, cls=NpEncoder)

    def run_multi_processing_simulations(self, candidates: List[Any], max_steps: int):
        with Pool(8) as p:
            arguments = []
            for argument in range(len(candidates)):
                arguments.append((candidates[argument],
                                  self.robot_type,
                                  self.structure_path,
                                  self.random_structure,
                                  self.raise_error_in_case_of_loading_structure_path,
                                  self.network_type,
                                  self.nodes,
                                  self.eta,
                                  self.environment_type,
                                  max_steps,
                                  False))
            return p.map(run_candidate_simulation, arguments)

    def run_single_processing_simulations(self, candidates: List[Any], max_steps: int,
                                          weight_update_steps: int, display: bool = False):
        fitnesses = []
        for candidate in candidates:
            fitness = run_candidate_simulation(
                [
                    candidate,
                    self.robot_type,
                    self.structure_path,
                    self.random_structure,
                    self.raise_error_in_case_of_loading_structure_path,
                    self.network_type,
                    self.nodes,
                    self.eta,
                    self.environment_type,
                    max_steps,
                    weight_update_steps,
                    display
                ])
            fitnesses.append(fitness)
        return fitnesses

    def prune(self, max_steps: int, weight_update_steps: int, prune_ratio: Optional[int], weight_pruning_time: Optional[int],
              results_folder: Optional[str] = None, display: bool = False):
        with open(self.weight_path, 'r') as f:
            data = json.load(f)

        candidate = data["candidate"]
        robot = initialize_robot(self.robot_type, self.structure_path, self.random_structure,
                                 self.raise_error_in_case_of_loading_structure_path,
                                 self.network_type, self.nodes, self.eta)
        is_ratio_computed = True
        environment = initialize_environment(self.environment_type, robot)
        environment.seed(1)
        robot.set_hrules(candidate)
        simulator = environment.sim
        simulator.reset()
        viewer = None
        if display:
            viewer = EvoViewer(simulator)
            viewer.track_objects('robot')
        number_of_updates = 0
        done = False
        are_weights_to_be_updated = True
        step = 0
        reward = 0
        obs = environment.get_first_obs(robot.voxel_number, is_ratio_computed)
        robot.update_weights()
        is_nn_pruned = False
        while not done:
            step += 1
            output = robot.get_action(obs, is_ratio_computed)
            output = np.array(output)
            obs, rew, done, _ = environment.step(robot.voxel_number, output, is_ratio_computed)
            reward += rew
            if step % weight_update_steps == 0 and are_weights_to_be_updated:
                robot.update_weights()
                number_of_updates += 1
                if step >= 3 * max_steps / 4:
                    are_weights_to_be_updated = False
            if weight_pruning_time is not None and number_of_updates == weight_pruning_time and not is_nn_pruned:
                robot.prune(folder=results_folder, prune_time=number_of_updates, prune_ratio=prune_ratio)
                are_weights_to_be_updated = False
                is_nn_pruned = True
            if step >= max_steps:
                done = True
            if display:
                viewer.render('screen')

    def test(self, max_steps: int, weight_update_steps: int):
        with open(self.weight_path, 'r') as f:
            data = json.load(f)
        candidate = data["candidate"]
        self.test_simulation(candidate, max_steps, weight_update_steps)

    def test_simulation(self, candidate: Any, max_steps: int, weight_update_steps: int):
        robot = initialize_robot(
            self.robot_type, self.structure_path, self.random_structure,
            self.raise_error_in_case_of_loading_structure_path,
            self.network_type, self.nodes, self.eta
        )
        is_ratio_computed = True
        environment = initialize_environment(self.environment_type, robot)
        environment.seed(1)
        robot.set_hrules(candidate)
        simulator = environment.sim
        simulator.reset()
        viewer = EvoViewer(simulator)
        viewer.track_objects('robot')
        done = False
        are_weights_to_be_updated = True
        step = 0
        reward = 0
        obs = environment.get_first_obs(robot.voxel_number, is_ratio_computed)
        robot.update_weights()
        while not done:
            step += 1
            output = robot.get_action(obs, is_ratio_computed)
            output = np.array(output)
            obs, rew, done, _ = environment.step(robot.voxel_number, output, is_ratio_computed)
            reward += rew
            if step % weight_update_steps == 0 and are_weights_to_be_updated:
                robot.update_weights()
                if step >= 3*max_steps/4:
                    are_weights_to_be_updated = False
            viewer.render('screen')


def run_candidate_simulation(kwargs: list) -> float:
    candidate = kwargs[0]
    robot_type = kwargs[1]
    structure_path = kwargs[2]
    random_structure = kwargs[3]
    raise_error_in_case_of_loading_structure_path = kwargs[4]
    network_type = kwargs[5]
    nodes = kwargs[6]
    eta = kwargs[7]
    environment_type = kwargs[8]
    max_steps = kwargs[9]
    weight_update_steps = kwargs[10]
    display = kwargs[11]
    is_ratio_computed = True
    robot = initialize_robot(robot_type, structure_path, random_structure,
                             raise_error_in_case_of_loading_structure_path,
                             network_type, nodes, eta)
    environment = initialize_environment(environment_type, robot)
    environment.seed(1)
    robot.set_hrules(candidate)
    viewer = None
    simulator = environment.sim
    if display:
        viewer = EvoViewer(simulator)
    rewards = 0
    step = 0
    done = False
    simulator.reset()
    are_weights_to_be_updated = True
    obs = environment.get_first_obs(len(robot.voxels), is_ratio_computed)
    robot.update_weights()
    while not done:
        step += 1
        output = robot.get_action(obs, is_ratio_computed)
        output = np.array(output)
        obs, rew, done, _ = environment.step(robot.voxel_number, output, is_ratio_computed)
        rewards += rew
        if step % weight_update_steps == 0 and are_weights_to_be_updated:
            robot.update_weights()
            if step >= 3 * max_steps / 4:
                are_weights_to_be_updated = False
        if display:
            viewer.render('screen')
        if step >= max_steps:
            done = True
    if display:
        viewer.render('screen')
    return -rewards
