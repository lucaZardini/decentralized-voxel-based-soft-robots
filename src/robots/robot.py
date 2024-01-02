from enum import Enum


class RobotType(Enum):
    """Enumeration of robot types."""
    WORM = 1
    BIPED = 2


class RobotManager:

    @staticmethod
    def get_robot(robot_type: RobotType):
        """Returns a robot of the specified type."""
        if robot_type == RobotType.WORM:
            pass
        elif robot_type == RobotType.BIPED:
            pass
        else:
            raise ValueError(f'robot type {robot_type} not supported')
