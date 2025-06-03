import numpy as np

import deepxde as dde
from . import baseclass
from deepxde import config

class Magnetism_2D(baseclass.BasePDE):
    
    def __init__(self, bbox=[-1, 1, -1, 1], space=[0, 0, 1, 0.6], circ=[-0.2, -0.2, 0.02], mu0=4*np.pi*1e-7, I=100.0, sigma=0.02):
        super().__init__()
        # output dim
        self.output_config = [{'name': s} for s in ['Bx', 'By']]

        # geom
        self.bbox = bbox
        self.space = space
        self.circ = circ
        #geom = dde.geometry.Disk(space[0:2], space[2])
        geom = dde.geometry.Ellipse(space[0:2], space[2], space[3])
        #geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        #ngeom = dde.geometry.Rectangle(xmin=[space[0], space[2]], xmax=[space[1], space[3]])
        #geom = dde.geometry.csg.CSGDifference(geom, ngeom)
        circ = dde.geometry.Disk(circ[0:2], circ[2])
        self.geom = dde.geometry.csg.CSGDifference(geom, circ)

        self.mu0 = mu0
        self.I = I
        self.sigma = sigma

        # PDE
        def mag_pde(x, u):
            Bx_x = dde.grad.jacobian(u, x, i=0, j=0)
            Bx_y = dde.grad.jacobian(u, x, i=0, j=1)
            By_x = dde.grad.jacobian(u, x, i=1, j=0)
            By_y = dde.grad.jacobian(u, x, i=1, j=1)

            def f(xy):
                x, y = xy[:, 0:1], xy[:, 1:2]
                x0, y0 = self.circ[0], self.circ[1]
                return ((self.mu0 * self.I) / (2 * np.pi * self.sigma**2)) * dde.backend.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))
            
            gauss = Bx_x + By_y
            amp_max = By_x - Bx_y - f(x)

            return [gauss, amp_max]
        
        self.pde = mag_pde
        self.set_pdeloss(names=["gauss", "amp_max"])

        # Reference Solution
        self.ref_sol = lambda xy: np.concatenate((Bx_ref(xy), By_ref(xy)), axis=1)

        def Bx_ref(xy):
            x, y = xy[:, 0:1], xy[:, 1:2]
            x0, y0 = self.circ[0], self.circ[1]
            return -(self.mu0 * self.I * (y - y0)) / (2 * np.pi * ((x - x0)**2 + (y - y0)**2))

        def By_ref(xy):
            x, y = xy[:, 0:1], xy[:, 1:2]
            x0, y0 = self.circ[0], self.circ[1]
            return (self.mu0 * self.I * (x - x0)) / (2 * np.pi * ((x - x0)**2 + (y - y0)**2))

        # Boundary Condition
        def boundary_space(x, on_boundary):
            center_circ = np.array(self.circ[0:2], dtype=config.real(np))
            return on_boundary and not np.isclose(np.linalg.norm(x - center_circ, axis=-1), self.circ[2]) #and not (x[0] > self.space[0] and x[1] > self.space[2])
        
        # Value on Boundary
        def boundary_func(x, component):
            if component == 0:
                return -(self.mu0 * self.I * x[:, 1:2]) / (2 * np.pi * (x[:, 0:1]**2 + x[:, 1:2]**2))
            else:
                return (self.mu0 * self.I * x[:, 0:1]) / (2 * np.pi * (x[:, 0:1]**2 + x[:, 1:2]**2))
        
        # BCs
        self.add_bcs([{
            'name': 'spaceX',
            'component': 0,
            'function': (lambda x: boundary_func(x, component=0)),
            #'function': (lambda x: 0),
            'bc': boundary_space,
            'type': 'dirichlet'
        }, {
            'name': 'spaceY',
            'component': 1,
            'function': (lambda x: boundary_func(x, component=1)),
            #'function': (lambda x: 0),
            'bc': boundary_space,
            'type': 'dirichlet'
        }])

        # Training Config
        self.training_points()


class ElectroMag_2DTime(baseclass.BasePDE):
    
    def __init__(self, bbox=[-1, 1, -1, 1, 0, 5], space=[0, 0, 1, 0.6], circ=[0, 0, 0.02], mu0=4*np.pi*1e-7, I=100.0, sigma=0.01):
        super().__init__()
        # output dim
        self.output_config = [{'name': s} for s in ['Bx', 'By']]

        # geom
        self.bbox = bbox
        self.space = space
        self.circ = circ
        geom = dde.geometry.Ellipse(space[0:2], space[2], space[3])
        circ = dde.geometry.Disk(circ[0:2], circ[2])
        self.geom = dde.geometry.csg.CSGDifference(geom, circ)
        timedomain = dde.geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        self.mu0 = mu0
        self.I = I
        self.sigma = sigma

        # PDE
        def mag_pde(x, u):
            Bx_x = dde.grad.jacobian(u, x, i=0, j=0)
            Bx_y = dde.grad.jacobian(u, x, i=0, j=1)
            By_x = dde.grad.jacobian(u, x, i=1, j=0)
            By_y = dde.grad.jacobian(u, x, i=1, j=1)

            def f(xy):
                x, y = xy[:, 0:1], xy[:, 1:2]
                return ((self.mu0 * self.I) / (2 * np.pi * self.sigma**2)) * dde.backend.exp(-(x**2 + y**2) / (2 * self.sigma**2))
            
            gauss = Bx_x + By_y
            amp_max = By_x - Bx_y - f(x)

            return [gauss, amp_max]
        
        self.pde = mag_pde
        self.set_pdeloss(names=["gauss", "amp_max"])

        # Reference Solution
        self.ref_sol = lambda xy: np.concatenate((Bx_ref(xy), By_ref(xy)), axis=1)

        def Bx_ref(xy):
            x, y = xy[:, 0:1], xy[:, 1:2]
            return -(self.mu0 * self.I * y) / (2 * np.pi * (x**2 + y**2))

        def By_ref(xy):
            x, y = xy[:, 0:1], xy[:, 1:2]
            return (self.mu0 * self.I * x) / (2 * np.pi * (x**2 + y**2))

        # Boundary Condition
        def boundary_space(x, on_boundary):
            center = np.array(self.space[0:2], dtype=config.real(np))
            return on_boundary and np.isclose(np.linalg.norm(x - center, axis=-1), self.space[2])
        
        # Value on Boundary
        def boundary_func(x, component):
            if component == 0:
                return -(self.mu0 * self.I * x[:, 1:2]) / (2 * np.pi * (x[:, 0:1]**2 + x[:, 1:2]**2))
            else:
                return (self.mu0 * self.I * x[:, 0:1]) / (2 * np.pi * (x[:, 0:1]**2 + x[:, 1:2]**2))
        
        # BCs
        self.add_bcs([{
            'name': 'spaceX',
            'component': 0,
            'function': (lambda x: boundary_func(x, component=0)),
            #'function': (lambda x: 0),
            'bc': boundary_space,
            'type': 'dirichlet'
        }, {
            'name': 'spaceY',
            'component': 1,
            'function': (lambda x: boundary_func(x, component=1)),
            #'function': (lambda x: 0),
            'bc': boundary_space,
            'type': 'dirichlet'
        }])

        # Training Config
        self.training_points()
