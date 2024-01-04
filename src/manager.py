from typing import List, Any

from evogym import EvoViewer

from environments.type import EnvironmentType, EnvironmentManager
from networks.type import NetworkManager, NetworkType
from robots.robot import RobotType, RobotManager


class Manager:

    def __init__(self,
                 robot_type: RobotType, structure_path: str, random_structure: bool,
                 raise_error_in_case_of_loading_structure_path: bool,
                 environment_type: EnvironmentType,
                 network_type: NetworkType, nodes: List[int],
                 offsprings: int, population_size: int, sigma: float,  # TODO: define evolutionary algorithm
                 eta: float = 0.1):
        self.robot = RobotManager.get_robot(robot_type, structure_path, random_structure, raise_error_in_case_of_loading_structure_path, nodes, network_type, eta)
        self.environment = EnvironmentManager.get_environment(environment_type, self.robot.structure, self.robot.connections)
        self.simulator = self.environment.sim
        self.viewer = EvoViewer(self.simulator)

    def train(self, generations: int, max_steps: int, display: bool = False):
        self.simulator.reset()
        best_individual = None
        best_fitness = 0
        fitnesses = []
        for i in range(generations):
            candidates = self.evolutionary_algorithm.get_candidates()
            for candidate in candidates:
                fitness = self.run_candidate_simulation(candidate, max_steps, display)
                fitnesses.append(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = candidate
            self.evolutionary_algorithm.update(candidates, fitnesses)

    def run_candidate_simulation(self, candidate: Any, max_steps: int, display: bool = False) -> float:
        self.robot.set_hrules(candidate)
        rewards = []
        for j in range(max_steps):
            done = False
            self.simulator.reset()
            rewards[j] = 0
            while not done: # TODO: maybe set also max number of iterations
                # TODO: implement action:
                #  get the inputs from the env and pass them to the network together with the previous activation function
                obs, rew, done, _ = self.environment.step()  # TODO: pass the action to the env
                rewards[j] += rew
                self.robot.update_weights()
                if display:
                    self.viewer.render('screen')
        if display:
            self.viewer.render('screen')
        return -sum(rewards) / max_steps

    def prune(self):
        self.simulator.reset()
        pass
