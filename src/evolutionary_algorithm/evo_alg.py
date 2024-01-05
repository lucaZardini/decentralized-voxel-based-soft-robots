import random
from abc import abstractmethod, ABC
from enum import Enum

import numpy as np
from cma import CMAEvolutionStrategy as cmaes
from typing import Optional, List


class EvoAlgoType(Enum):
    CMAES = "cmaes"


class EvoAlgoManager:

    @staticmethod
    def get_evolutionary_algorithm(evo_algo_type: EvoAlgoType, number_of_parameters: int, offsprings: int,
                                   population_size: int, sigma: float, pop_init_range: Optional[List[int]] = None):
        if evo_algo_type == EvoAlgoType.CMAES:
            return CMAES(number_of_parameters, offsprings, population_size, sigma, pop_init_range)
        else:
            raise ValueError(f'evolutionary algorithm type {evo_algo_type} not supported')


class EvoAlgo(ABC):

    def __init__(self, number_of_parameters: int):
        self.number_of_parameters = number_of_parameters

    @abstractmethod
    def get_candidates(self):
        pass

    @abstractmethod
    def update(self, candidates, fitnesses):
        pass


class CMAES(EvoAlgo):

    def __init__(self, number_of_parameters: int, offsprings: int, population_size: int, sigma: float,
                 pop_init_range: Optional[List[int]]):
        super().__init__(number_of_parameters)
        self.offsprings = offsprings
        self.population_size = population_size
        self.sigma = sigma
        self.pop_init_range = pop_init_range if pop_init_range is not None else [0, 1]
        self.cmaes = cmaes(
            self.__generate_init_range(),
            self.sigma,
            {'popsize': self.offsprings, 'seed': 0, 'CMA_mu': self.population_size}
        )

    def get_candidates(self) -> np.ndarray:
        return self.cmaes.ask()

    def update(self, candidates, fitnesses):
        self.cmaes.tell(candidates, fitnesses)

    def __generate_init_range(self):
        return np.asarray([random.uniform(self.pop_init_range[0],
                                          self.pop_init_range[1])
                           for _ in range(self.number_of_parameters)])
