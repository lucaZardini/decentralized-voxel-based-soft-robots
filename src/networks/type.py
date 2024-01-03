from enum import Enum

from typing import List

from networks.hnn import HNN


class NetworkType(Enum):
    """Enumeration of network types."""
    HNN = "hnn"


class NetworkManager:

    @staticmethod
    def get_network(network_type: NetworkType, nodes: List[int], eta: float = 0.1):
        """Returns a network of the specified type."""
        if network_type == NetworkType.HNN:
            return HNN(nodes, eta)
        else:
            raise ValueError(f'network type {network_type} not supported')
