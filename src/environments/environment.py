from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from evogym.envs import SoftBridge as SoftBridgeEvo, WalkingFlat as WalkingFlatEvo

from evogym.envs.base import BenchmarkBase
from typing import Union


class EnvironmentType(Enum):
    """
    The type of environment to use.
    """
    WALKING_FLAT = "walking_flat"
    SOFT_BRIDGE = "soft_bridge"


class EnvironmentManager:

    @staticmethod
    def get_environment(environment_type: EnvironmentType, robot_structure: np.ndarray, robot_connections: np.ndarray):
        """
        Returns an environment of the specified type.
        """
        if environment_type == EnvironmentType.SOFT_BRIDGE:
            return SoftBridge(body=robot_structure, connections=robot_connections)
        elif environment_type == EnvironmentType.WALKING_FLAT:
            return WalkingFlat(body=robot_structure, connections=robot_connections)
        else:
            raise ValueError(f'environment type {environment_type} not supported')


class Environment(ABC):

    def __init__(self, body: np.ndarray, connections: np.ndarray):
        self._body = body
        self._connections = connections

    @abstractmethod
    def evo_env(self) -> BenchmarkBase:
        pass

    @abstractmethod
    def get_first_obs(self, n_voxels: int) -> np.ndarray:
        pass

    def step(self, n_voxels: int, action: np.ndarray) -> Union[np.ndarray, float, bool, dict]:
        """
        Step the environment with the given action.
        """
        return self.evo_env().step(action)

    def reset(self) -> None:
        """
        Reset the environment.
        """
        return self.evo_env().reset()

    @property
    def sim(self):
        return self.evo_env().sim

    def seed(self, seed: int):
        self.evo_env().seed(seed)

    def calculate_polygon_area(self, x_coords, y_coords):
        n = len(x_coords)
        if n != 4:
            raise ValueError("The function requires exactly four points.")

        area = 0.5 * abs(
            sum(x_coords[i] * y_coords[(i + 1) % n] - x_coords[(i + 1) % n] * y_coords[i] for i in range(n)))
        return area


class SoftBridge(Environment):

    def __init__(self, body: np.ndarray, connections: np.ndarray):
        super().__init__(body, connections)
        self._evo_sim = SoftBridgeEvo(body=body, connections=connections)

    def evo_env(self) -> BenchmarkBase:
        return self._evo_sim

    def get_first_obs(self, n_voxels: int) -> np.ndarray:
        return np.concatenate((
            self._evo_sim.get_vel_com_obs("robot"),
            self._evo_sim.get_ort_obs("robot"),
            self._evo_sim.get_relative_pos_obs("robot"),
            ))


class WalkingFlat(Environment):

    def __init__(self, body: np.ndarray, connections: np.ndarray):
        super().__init__(body, connections)
        self._evo_sim = WalkingFlatEvo(body=body, connections=connections)

    def evo_env(self) -> BenchmarkBase:
        return self._evo_sim

    def get_obs(self, n_voxels: int, compute_ratio: bool = True) -> np.ndarray:
        self.evo_env().get_relative_pos_obs('robot')
        relative_pos = self._evo_sim.get_relative_pos_obs("robot")
        relative_pos_x = relative_pos[:int(len(relative_pos) / 2)]
        relative_pos_y = relative_pos[int(len(relative_pos) / 2):]
        voxels_pos = self.get_voxel_pos(n_voxels, relative_pos_x, relative_pos_y)
        if compute_ratio:
            robot_pos_com = self.evo_env().object_pos_at_time(self.evo_env().get_time(), 'robot')
            ratio_expansion_contraction = self.compute_ratio(robot_pos_com, n_voxels)
            obs: np.ndarray = np.concatenate((
                self._evo_sim.get_vel_com_obs("robot"),
                voxels_pos,
                ratio_expansion_contraction
            ))
        else:
            obs: np.ndarray = np.concatenate((
                self._evo_sim.get_vel_com_obs("robot"),
                voxels_pos
            ))
        return obs

    def compute_ratio(self, robot_pos_com: np.ndarray, n_voxels: int, relax_volume: float = 1) -> list:
        """
        Compute the ratio of contraction/expansion for each voxel.
        """
        pos_x = robot_pos_com[0]
        pos_y = robot_pos_com[1]
        voxels_pos = np.array(
            [pos_x[0], pos_x[2], pos_x[3], pos_x[1], pos_y[0], pos_y[2], pos_y[3], pos_y[1]])
        for i in range(n_voxels):
            if i != 0:
                if i == 1:
                    voxel_pos = np.array(
                        [pos_x[1], pos_x[3], pos_x[5], pos_x[4], pos_y[1], pos_y[3], pos_y[5], pos_y[4]])
                else:
                    voxel_pos = np.array([pos_x[2 * i], pos_x[2 * i + 1], pos_x[2 * i + 3], pos_x[2 * i + 2],
                                          pos_y[2 * i], pos_y[2 * i + 1], pos_y[2 * i + 3], pos_y[2 * i + 2]])
                voxels_pos = np.concatenate((voxels_pos, voxel_pos))
        voxels_ratio = []
        for voxel_coords in range(0, len(voxels_pos), 8):
            x_coords = voxels_pos[voxel_coords:voxel_coords + 4]
            y_coords = voxels_pos[voxel_coords + 4:voxel_coords + 8]
            area = self.calculate_polygon_area(x_coords, y_coords)
            voxels_ratio.append(area / relax_volume)
        return voxels_ratio

    def step(self, n_voxels: int, action: np.ndarray, compute_ratio: bool = True) -> Union[np.ndarray, float, bool, dict]:
        """
        Step the environment with the given action.
        """
        obs, rew, done, _ = self._evo_sim.step(action)
        obs = self.get_obs(n_voxels, compute_ratio)
        return obs, rew, done, {}

    def get_voxel_pos(self, n_voxels: int, relative_pos_x: np.ndarray, relative_pos_y: np.ndarray) -> np.ndarray:
        voxels_pos = np.array(
            [relative_pos_x[0], relative_pos_y[0], relative_pos_x[2], relative_pos_y[2], relative_pos_x[1],
             relative_pos_y[1], relative_pos_x[3], relative_pos_y[3]])
        for i in range(n_voxels):
            if i != 0:
                if i == 1:
                    voxel_pos = np.array(
                        [relative_pos_x[1], relative_pos_y[1], relative_pos_x[3], relative_pos_y[3], relative_pos_x[4],
                         relative_pos_y[4], relative_pos_x[5], relative_pos_y[5]])
                else:
                    voxel_pos = np.array([relative_pos_x[2 * i], relative_pos_y[2 * i], relative_pos_x[2 * i + 1],
                                          relative_pos_y[2 * i + 1], relative_pos_x[2 * i + 2],
                                          relative_pos_y[2 * i + 2], relative_pos_x[2 * i + 3],
                                          relative_pos_y[2 * i + 3]])
                voxels_pos = np.concatenate((voxels_pos, voxel_pos))
        return voxels_pos

    def get_first_obs(self, n_voxels: int, compute_ratio: bool = True) -> np.ndarray:
        return self.get_obs(n_voxels, compute_ratio)
