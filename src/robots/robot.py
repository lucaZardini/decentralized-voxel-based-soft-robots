import json
import os
from abc import ABC
from enum import Enum

from typing import Tuple, List, Optional, Any

import numpy as np
from evogym import sample_robot

from networks.type import NetworkType, NetworkManager
from robots.voxel import Voxel
from utils import NpEncoder


class RobotType(Enum):
    """Enumeration of robot types."""
    WORM = "worm"
    BIPED = "biped"


class RobotManager:

    @staticmethod
    def get_robot(robot_type: RobotType, structure_path: str, random_structure: bool, raise_error_in_case_of_failure: bool,
                  nodes: List[int], network_type: NetworkType, eta: float):
        """Returns a robot of the specified type."""
        if robot_type == RobotType.WORM:
            return Worm(structure_path, nodes, network_type, eta, random_structure, raise_error_in_case_of_failure)
        elif robot_type == RobotType.BIPED:
            pass
        else:
            raise ValueError(f'robot type {robot_type} not supported')


class Robot(ABC):
    def __init__(self, structure_path: str, nodes: List[int], network_type: NetworkType, eta: float):
        self.structure = None
        self.connections = None
        self._inner_connections = None
        self._node_assignment = None
        self.voxels: Optional[List[Voxel]] = None
        self.structure_path = structure_path
        self.nodes = nodes
        self.network_type = network_type
        self.eta = eta
        self._previous_activation_value = None
        self._current_activation_value = None

    @property
    def voxel_number(self) -> int:
        """
        Returns the number of voxels of the robot
        """
        return len(self.voxels)

    @property
    def parameters_number(self) -> int:
        """
        Returns the number of parameters of the robot
        """
        params_number = 0
        for voxel in self.voxels:
            params_number += voxel.parameters_number * 4
        return params_number

    @staticmethod
    def _check_structure_path(structure_path: str, to_read: bool):
        """
        Checks if the structure path is valid
        """
        # Check if a file exists at the specified path
        if to_read:
            if not os.path.isfile(structure_path):
                raise ValueError(f'file {structure_path} does not exist')
            # Check if the file is a json file
            if not structure_path.endswith('.json'):
                raise ValueError(f'file {structure_path} is not a json file')

    def _generate_inner_connections(self) -> dict:
        """
        Generates the inner connections of the robot.
        First, generates a dictionary with the same shape of the robot structure.
        The first layer stands for the rows while the second layer stands for the columns.
        Then, for each element of the dictionary, row and column, the list of the neighbors is generated.
        """
        inner_connection = {}
        number_of_voxels = self.structure.shape[0] * self.structure.shape[1]
        for i in range(number_of_voxels):
            inner_connection[i] = []

        number_of_rows_in_connections = len(self.connections[0])
        for i in range(number_of_rows_in_connections):
            first_voxel = self.connections[0][i]
            second_voxel = self.connections[1][i]
            inner_connection[first_voxel].append(second_voxel)
            inner_connection[second_voxel].append(first_voxel)

        return inner_connection

    def get_connection_per_voxel_index(self, voxel) -> list:
        """
        Returns the inner connections of the robot
        """
        return self._inner_connections[voxel]

    def _generate_predefined_structure(self) -> Tuple[np.ndarray, np.ndarray, List[Voxel]]:
        """
        Generates a predefined robot structure by reading a json file
        """
        try:
            # read file as json
            with open(self.structure_path, 'r') as json_file:
                json_data = json.load(json_file)
            # check if the json file contains the required keys
            if 'structure' not in json_data.keys() or 'connections' not in json_data.keys():
                raise ValueError(f'file {self.structure_path} does not contain the key "structure" and "connections"')
            # check if the structure and connections are valid
            robot_structure = np.array(json_data['structure'])
            robot_connections = np.array(json_data['connections'])
            voxels: List[Voxel] = self._generate_voxels(robot_structure)
            return robot_structure, robot_connections, voxels
        except json.JSONDecodeError:
            raise ValueError(f'file {self.structure_path} is not a valid json file')
        except Exception as e:
            raise e

    def _generate_random_structure(self, rows: int, columns: int) -> Tuple[np.ndarray, np.ndarray, List[Voxel]]:
        """
        Generates a random robot structure taking advantage of sample_robot function
        """
        structure, connections = sample_robot((rows, columns))
        voxels: List[Voxel] = self._generate_voxels(structure)
        return structure, connections, voxels

    @staticmethod
    def _generate_voxels(structure: np.ndarray) -> List[Voxel]:
        voxels: List[Voxel] = []
        for row_index, row in enumerate(list(structure)):
            for value_index, value in enumerate(row):
                if value in [4, 3]:
                    voxel_id = row_index * len(row) + value_index
                    voxels.append(Voxel(value, voxel_id))
        return voxels

    def _assign_nn_to_each_voxel(self) -> dict:
        """
        Assigns a neural network to each voxel
        """
        node_assignment = {}
        for voxel in self.voxels:
            if voxel.type in [4, 3]:
                voxel.assign_nn(NetworkManager.get_network(self.network_type, self.nodes, self.eta))
                node_assignment[voxel.id] = voxel
        return node_assignment

    def set_hrules(self, hrules: Any):
        """
        Sets the hrules of each voxel
        """
        number_of_parameters = self.parameters_number
        if len(hrules) != number_of_parameters:
            raise ValueError(f'hrules must be of length {number_of_parameters}')
        parameter_per_nn = int(number_of_parameters/self.voxel_number)
        start = 0
        for voxel in self.voxels:
            voxel.set_nn_hrules(hrules[start:start+parameter_per_nn])
            start += parameter_per_nn

    def update_weights(self):
        """
        Updates the weights of each voxel
        """
        for voxel in self.voxels:
            voxel.nn.update_weights()

    def get_action(self, obs: np.ndarray, is_ratio_computed: bool = True) -> Any:
        action = []
        self._current_activation_value = {}
        velocity_x = obs[0]
        velocity_y = obs[1]
        for voxel_iter, voxel in enumerate(self.voxels):
            adjacent_activations = []
            adjacent_voxels = self._inner_connections[voxel.id]
            for adjacent_voxel in adjacent_voxels:
                if self._previous_activation_value is None or adjacent_voxel not in self._previous_activation_value:
                    adjacent_activations.append(0.0)
                else:
                    adjacent_activations.append(self._previous_activation_value[adjacent_voxel])
            if len(adjacent_activations) < 4:
                adjacent_activations += [0.0] * (4 - len(adjacent_activations))
            elif len(adjacent_activations) > 4:
                raise ValueError(f'adjacent activations must be of length 4, found {len(adjacent_activations)}')
            input = np.array(
                [velocity_x, velocity_y, obs[2+8*voxel_iter], obs[3+8*voxel_iter], obs[4+8*voxel_iter], obs[5+8*voxel_iter],
                 obs[6+8*voxel_iter], obs[7+8*voxel_iter], obs[8+8*voxel_iter], obs[9+8*voxel_iter]] + adjacent_activations
            )
            if is_ratio_computed:
                ratios = obs[-self.voxel_number:]
                input = np.concatenate((input, [ratios[voxel_iter]]))
            output = voxel.nn.activate(input)
            self._current_activation_value[voxel.id] = output[1]
            action.append(output[0])
        self._previous_activation_value = self._current_activation_value
        self._current_activation_value = None
        return action

    def save_structure(self):
        """
        Saves the structure of the robot in a json file
        """
        try:
            structure_dict = {
                'structure': self.structure.tolist(),
                'connections': self.connections.tolist()
            }
            with open(self.structure_path, 'w') as json_file:
                json.dump(structure_dict, json_file, indent=4, cls=NpEncoder)
        except Exception as e:
            print("Cannot save structure, error: ", e)

    def save_hrules(self):
        """
        Saves the hrules of the robot in a json file
        """
        try:
            hrules = []
            for voxel in self.voxels:
                hrules += voxel.nn.hrules
            with open(self.structure_path, 'w') as json_file:
                json.dump(hrules, json_file, indent=4, cls=NpEncoder)
        except Exception as e:
            print("Cannot save hrules, error: ", e)

    def _load_hrules(self):
        """
        Loads the hrules of the robot from a json file
        """
        try:
            with open(self.structure_path, 'r') as json_file:
                hrules = json.load(json_file)
            return hrules
        except Exception as e:
            print("Cannot load hrules, error: ", e)

    def prune(self, folder: str, prune_time: int, prune_ratio: int = 40):
        for voxel in self.voxels:
            voxel.nn.prune_weights(prune_ratio, folder, prune_time, voxel.id)


class Worm(Robot):

    DEFAULT_STRUCTURE = np.array([3, 3, 3, 3, 3])
    DEFAULT_CONNECTIONS = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])

    def __init__(self, structure_path: str, nodes: List[int], network_type: NetworkType, eta: float,
                 random_structure: bool = False, raise_error_in_case_of_failure: bool = False,
                 pretrained: bool = False):
        super().__init__(structure_path, nodes, network_type, eta)
        is_structure_to_read = not random_structure
        self._check_structure_path(structure_path, is_structure_to_read)
        if random_structure:
            self.structure, self.connections, self.voxels = self._generate_random_structure(1, 5)
            self.save_structure()
        else:
            try:
                self.structure, self.connections, self.voxels = self._generate_predefined_structure()
            except Exception as e:
                if raise_error_in_case_of_failure:
                    raise e
                else:
                    self.structure = self.DEFAULT_STRUCTURE
                    self.connections = self.DEFAULT_CONNECTIONS
                    self.voxels = self._generate_voxels(self.structure)

        self._inner_connections = self._generate_inner_connections()
        if pretrained:
            self._node_assignment = self._assign_nn_to_each_voxel()
            hrules = self._load_hrules()
            self.set_hrules(hrules)
        else:
            self._node_assignment = self._assign_nn_to_each_voxel()
