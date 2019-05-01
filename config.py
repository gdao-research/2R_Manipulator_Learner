from collections import namedtuple
import numpy as np

_CONFIG = {
    'PI': np.pi,
    'tau': 1e-3,
    'clip_norm': None,
    'critic_l2_reg': 0,
    'discount_factor': 0.9
}

Params = namedtuple(typename='Params', field_names=list(_CONFIG.keys()))
CONFIG = Params(**_CONFIG)
