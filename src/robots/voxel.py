from typing import Any, Optional

from networks.hnn import HNN


class Voxel:

    def __init__(self, voxel_type: int, voxel_id: int, nn: Optional[HNN] = None):
        """
        Args:
            voxel_type (int): type of voxel, ranging from 0 to 5 (the values of the evogym VoxelType enum).
            voxel_id (int): id of the voxel.
        """
        self.type = voxel_type
        self.id = voxel_id
        self.nn = nn

    @property
    def parameters_number(self) -> int:
        params_number = 0
        for index, nodes in enumerate(self.nn.nodes):
            if not index == len(self.nn.nodes) - 1:
                params_number += nodes * self.nn.nodes[index + 1]
        return params_number

    def assign_nn(self, nn: HNN):
        self.nn = nn

    def set_nn_hrules(self, hrules: Any):
        self.nn.set_hrules(hrules)
