import numpy as np

import deepxde as dde
from . import baseclass

import logging
logger = logging.getLogger(__name__)

class KAN_Test(baseclass.BasePDE):
    
    def __init__(self, bbox=[-1, 1, -1, 1]):
        super().__init__()

        # Output Dim
        self.output_config = [{'name': 'out'}]

        # Geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])

        # PDE
        def kan_pde(xy, u):
            x, y = xy[:, 0:1], xy[:, 1:2]
            u_xx = dde.grad.hessian(u, xy, i=0, j=0)
            u_yy = dde.grad.hessian(u, xy, i=1, j=1)

            source = -2 * np.pi**2 * dde.backend.sin(np.pi * x) * dde.backend.sin(np.pi * y)

            return [u_xx + u_yy - source]
        
        self.pde = kan_pde
        self.set_pdeloss(names=["pde_loss"])

        # Reference Solution
        self.ref_sol = lambda x: np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])

        # Boundary Condition
        def boundary_space(x, on_boundary):
            return on_boundary
        
        # Value on Boundary
        def boundary_func(x):
            return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
        
        # BCs
        self.add_bcs([{
            'name': 'bc',
            'component': 0,
            'function': (lambda x: boundary_func(x)),
            'bc': boundary_space,
            'type': 'dirichlet'
        }])

        # Training Config
        self.training_points()
