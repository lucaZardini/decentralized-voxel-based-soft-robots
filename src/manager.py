from multiprocessing import Pool
from typing import List, Any

from evogym import EvoViewer

from environments.type import EnvironmentType, EnvironmentManager
from evolutionary_algorithm.evo_alg import EvoAlgoManager, EvoAlgoType
from networks.type import NetworkType
from robots.robot import RobotType, RobotManager, Robot


class Manager:

    def __init__(self,
                 robot_type: RobotType, structure_path: str, random_structure: bool,
                 raise_error_in_case_of_loading_structure_path: bool,
                 environment_type: EnvironmentType,
                 network_type: NetworkType, nodes: List[int],
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
        self.raise_error_in_case_of_loading_structure_path = raise_error_in_case_of_loading_structure_path
        self.robot = self.initialize_robot()
        self.evolutionary_algorithm = EvoAlgoManager.get_evolutionary_algorithm(evo_algo_type, self.robot.parameters_number, offsprings, population_size, sigma)

    def initialize_robot(self):
        return RobotManager.get_robot(self.robot_type, self.structure_path, self.random_structure, self.raise_error_in_case_of_loading_structure_path, self.nodes, self.network_type, self.eta)

    def initialize_environment(self, robot: Robot):
        return EnvironmentManager.get_environment(self.environment_type, robot.structure, robot.connections)

    def train(self, generations: int, max_steps: int, multi_processing: bool = False, display: bool = False):
        # self.simulator.reset()
        for i in range(generations):
            candidates = self.evolutionary_algorithm.get_candidates()
            if multi_processing:
                fitnesses = self.run_multi_processing_simulations(candidates, max_steps)
            else:
                fitnesses = self.run_single_processing_simulations(candidates, max_steps, display)
            self.evolutionary_algorithm.update(candidates, fitnesses)
            #  TODO: when to save the best individual of that generation?

    def run_multi_processing_simulations(self, candidates: List[Any], max_steps: int):
        with Pool(self.offsprings) as p:
            return p.map(self.run_candidate_simulation, candidates, max_steps)

    def run_single_processing_simulations(self, candidates: List[Any], max_steps: int, display: bool = False):
        fitnesses = []
        for candidate in candidates:
            fitness = self.run_candidate_simulation(candidate, max_steps, display)
            fitnesses.append(fitness)
        return fitnesses

    def run_candidate_simulation(self, candidate: Any, max_steps: int, display: bool = False) -> float:
        robot = self.initialize_robot()
        environment = self.initialize_environment(robot)
        robot.set_hrules(candidate)
        simulator = None
        viewer = None
        if display:
            simulator = environment.sim
            viewer = EvoViewer(simulator)
        rewards = []
        for j in range(max_steps):
            done = False
            if display:
                simulator.reset()
            rewards[j] = 0
            while not done:  # TODO: maybe set also max number of iterations
                # TODO: implement action:
                #  get the inputs from the env and pass them to the network together with the previous activation function
                obs, rew, done, _ = environment.step()  # TODO: pass the action to the env
                rewards[j] += rew
                robot.update_weights()
                if display:
                    viewer.render('screen')
        if display:
            viewer.render('screen')
        return -sum(rewards) / max_steps

    def prune(self):
        pass
