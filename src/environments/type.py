from enum import Enum

import numpy as np
from evogym.envs import SoftBridge, WalkingFlat


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
