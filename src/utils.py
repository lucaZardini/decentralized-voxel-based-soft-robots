import json

import numpy as np


class NpEncoder(json.JSONEncoder):
    """
    JSON encoder for numpy types, used to serialize numpy types in json in the encoding metadata
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
