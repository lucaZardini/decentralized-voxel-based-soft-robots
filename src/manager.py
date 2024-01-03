from typing import List

from environments.type import EnvironmentType, EnvironmentManager
from networks.type import NetworkManager, NetworkType
from robots.robot import RobotType, RobotManager


class Manager:

    def __init__(self,
                 robot_type: RobotType, structure_path: str, random_structure: bool,
                 raise_error_in_case_of_loading_structure_path: bool,
                 environment_type: EnvironmentType,
                 network_type: NetworkType, nodes: List[int],
                 eta: float = 0.1):
        self.robot = RobotManager.get_robot(robot_type, structure_path, random_structure, raise_error_in_case_of_loading_structure_path, nodes, network_type, eta)
        self.environment = EnvironmentManager.get_environment(environment_type, self.robot.structure, self.robot.connections)

    def train(self):
        pass

    def prune(self):
        pass
