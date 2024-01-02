from enum import Enum


class EnvironmentType(Enum):
    """
    The type of environment to use.
    """
    TODO = 1


class EnvironmentManager:

    @staticmethod
    def get_environment(environment_type: EnvironmentType):
        """
        Returns an environment of the specified type.
        """
        if environment_type == EnvironmentType.TODO:
            pass
        else:
            raise ValueError(f'environment type {environment_type} not supported')
