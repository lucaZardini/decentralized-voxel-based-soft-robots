import json
from multiprocessing import Pool
from typing import List, Any, Optional

import numpy as np
from evogym import EvoViewer

from environments.type import EnvironmentType, EnvironmentManager
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

    def train(self, generations: int, number_of_attempts: int, max_steps: int, multi_processing: bool = False, display: bool = False):
        for i in range(generations):
            candidates = self.evolutionary_algorithm.get_candidates()
            if multi_processing:
                fitnesses = self.run_multi_processing_simulations(candidates, number_of_attempts, max_steps)
            else:
                fitnesses = self.run_single_processing_simulations(candidates, number_of_attempts, max_steps, display)
            self.evolutionary_algorithm.update(candidates, fitnesses)
            best_fitness_index = np.argmax(fitnesses)
            candidate = candidates[best_fitness_index]
            self.save_individual(candidate, i, fitnesses[best_fitness_index])
        best_individual = self.evolutionary_algorithm.get_best()
        best_fitness = self.evolutionary_algorithm.get_best_fitness()
        self.save_individual(best_individual, None, best_fitness)

    def save_individual(self, candidate: Any, generation: Optional[int], fitness: float):
        if generation is not None:
            file_path = self.weight_path + f'_{generation}.json'
        else:
            file_path = self.weight_path + "_best.json"
        with open(file_path, 'w') as f:
            json.dump({"candidate": candidate, "fitness": fitness}, f, indent=4, cls=NpEncoder)

    def run_multi_processing_simulations(self, candidates: List[Any], number_of_attempts: int, max_steps: int):
        with Pool(self.offsprings) as p:
            return p.map(run_candidate_simulation, (
                candidates,
                self.robot_type,
                self.structure_path,
                self.random_structure,
                self.raise_error_in_case_of_loading_structure_path,
                self.network_type,
                self.nodes,
                self.eta,
                self.environment_type,
                number_of_attempts,
                max_steps)
                         )

    def run_single_processing_simulations(self, candidates: List[Any], number_of_attempts: int, max_steps: int, display: bool = False):
        fitnesses = []
        for candidate in candidates:
            fitness = run_candidate_simulation(
                candidate,
                self.robot_type,
                self.structure_path,
                self.random_structure,
                self.raise_error_in_case_of_loading_structure_path,
                self.network_type,
                self.nodes,
                self.eta,
                self.environment_type,
                number_of_attempts,
                max_steps,
                display)
            fitnesses.append(fitness)
        return fitnesses

    def prune(self):
        pass

    def test(self):
        with open(self.weight_path, 'r') as f:
            data = json.load(f)
        candidate = data["candidate"]
        fitness = data["fitness"]
        self.test_simulation(candidate)

    def test_simulation(self, candidate: Any):
        robot = initialize_robot(self.robot_type, self.structure_path, self.random_structure, self.raise_error_in_case_of_loading_structure_path,
                                    self.network_type, self.nodes, self.eta)
        environment = initialize_environment(self.environment_type, robot)
        robot.set_hrules(candidate)
        simulator = environment.sim
        viewer = EvoViewer(simulator)
        done = False
        while not done:
            velocity = environment.get_vel_com_obs('robot')
            output = robot.get_action(velocity_x=velocity[0], velocity_y=velocity[1])
            output = np.array(output)
            obs, rew, done, _ = environment.step(output)
            robot.update_weights()
            viewer.render('screen')


def run_candidate_simulation(candidate: Any, robot_type: RobotType, structure_path: str, random_structure: bool,
                             raise_error_in_case_of_loading_structure_path: bool, network_type: NetworkType,
                             nodes: List[int], eta: float, environment_type: EnvironmentType,
                             number_of_attempts: int, max_steps: int,
                             display: bool = False) -> float:
    robot = initialize_robot(robot_type, structure_path, random_structure, raise_error_in_case_of_loading_structure_path,
                             network_type, nodes, eta)
    environment = initialize_environment(environment_type, robot)
    robot.set_hrules(candidate)
    simulator = None
    viewer = None
    if display:
        simulator = environment.sim
        viewer = EvoViewer(simulator)
    rewards = []
    for j in range(number_of_attempts):
        step = 0
        done = False
        if display:
            simulator.reset()
        rewards.append(0)
        while not done:  # TODO: maybe set also max number of iterations
            # TODO: get the inputs from the env
            step += 1
            velocity = environment.get_vel_com_obs('robot')
            output = robot.get_action(velocity_x=velocity[0], velocity_y=velocity[1])  # TODO: pass the inputs
            output = np.array(output)
            obs, rew, done, _ = environment.step(output)
            rewards[j] += rew
            robot.update_weights()
            if display:
                viewer.render('screen')
            if step > max_steps:
                done = True
    if display:
        viewer.render('screen')
    return sum(rewards) / max_steps
