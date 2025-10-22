import numpy as np
from skfem import *
from skfem.helpers import dot, grad
from adaptmesh import triangulate

import deepxde as dde
from . import baseclass
from deepxde import config

import logging
logger = logging.getLogger(__name__)

class Magnetism_2D(baseclass.BasePDE):
    
    def __init__(self, bbox=[-1, 1, -1, 1], mu0=4*np.pi*1e-7, I=100000, sigma=0.02, form="disk"):
        super().__init__()

        # Output Dim
        self.output_config = [{'name': s} for s in ['Bx', 'By']]

        # Geom
        self.bbox = bbox
        self.form = form
        if form == "disk":
            self.space = [0, 0, 1]
            self.circ = [0, 0, 0.02]
            geom = dde.geometry.Disk(self.space[0:2], self.space[2])
        elif form == "ellipse":
            self.space = [0, 0, 1, 0.6]
            self.circ = [-0.3, 0, 0.02]
            geom = dde.geometry.Ellipse(self.space[0:2], self.space[2], self.space[3])
        elif form == "polygon":
            self.space = [0, 1, 0, 1]
            self.circ = [-0.3, -0.3, 0.02]
            geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
            ngeom = dde.geometry.Rectangle(xmin=[self.space[0], self.space[2]], xmax=[self.space[1], self.space[3]])
            geom = dde.geometry.csg.CSGDifference(geom, ngeom)
        wire = dde.geometry.Disk(self.circ[0:2], self.circ[2])
        self.geom = dde.geometry.csg.CSGDifference(geom, wire)

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
        self.set_pdeloss(names=["pde_gauss", "pde_amp_max"])

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
            if self.form == "polygon":
                return on_boundary and not np.isclose(np.linalg.norm(x - center_circ, axis=-1), self.circ[2]) \
                       and not (x[0] > self.space[0] or x[1] > self.space[2])
            else:
                return on_boundary and not np.isclose(np.linalg.norm(x - center_circ, axis=-1), self.circ[2])
        
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
            'bc': boundary_space,
            'type': 'dirichlet'
        }, {
            'name': 'spaceY',
            'component': 1,
            'function': (lambda x: boundary_func(x, component=1)),
            'bc': boundary_space,
            'type': 'dirichlet'
        }])

        # Training Config
        self.training_points()


class Magnetism_Ritz(Magnetism_2D):
    def __init__(self, bbox=[-1, 1, -1, 1], mu0=4*np.pi*1e-7, I=100000, sigma=0.02, form="disk"):
        super().__init__(bbox=[-1, 1, -1, 1], mu0=4*np.pi*1e-7, I=100000, sigma=0.02, form="disk")

        #def pde(x, u):
            


class Electric_2D(baseclass.BasePDE):
    
    def __init__(self, bbox=[-1, 1, -1, 1], gamma=100.0, frequency = 1e12, Q=1e-9, eps0=8.854e-12, sigma_x=0.3, sigma_y=0.3, form="disk"):
        super().__init__()

        # Output Dim
        self.output_config = [{'name': 'Ez'}]

        # Geom
        self.bbox = bbox
        self.form = form
        if form == "disk":
            self.space = [0, 0, 1]
            self.beam = [0, 0]
            self.geom = dde.geometry.Disk(self.space[0:2], self.space[2])
        elif form == "ellipse":
            self.space = [0, 0, 0.6, 1]
            self.beam = [0, 0.3]
            self.geom = dde.geometry.Ellipse(self.space[0:2], self.space[2], self.space[3])
        elif form == "polygon":
            self.space = [0, 1, 0, 1]
            self.beam = [-0.3, -0.3]
            geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
            ngeom = dde.geometry.Rectangle(xmin=[self.space[0], self.space[2]], xmax=[self.space[1], self.space[3]])
            self.geom = dde.geometry.csg.CSGDifference(geom, ngeom)

        self.gamma = gamma
        self.frequency = frequency
        self.Q = Q
        self.eps0 = eps0
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        c = 3e8
        beta = np.sqrt(1 - 1 / self.gamma**2)
        self.k = 2 * np.pi * self.frequency / (beta * c)

        def rho_transverse(x):
            x, y = x[:, 0:1], x[:, 1:2]
            return (self.Q / (2 * np.pi * self.sigma_x * self.sigma_y)) * dde.backend.exp(
                -((x - self.beam[0])**2 / (2 * self.sigma_x**2) + (y - self.beam[1])**2 / (2 * self.sigma_y**2))
            )

        # PDE
        def electric_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            
            rhs = (self.k / (self.eps0 * self.gamma**2)) * rho_transverse(x)

            return [u_xx + u_yy - (self.k**2 / self.gamma**2) * u + rhs]
        
        self.pde = electric_pde
        self.set_pdeloss(names=["pde"])

        # Solution FEM
        if form == "disk":
            theta = np.linspace(0, 2 * np.pi, 250, endpoint=False)
            x_disk = self.space[2] * np.cos(theta)
            y_disk = self.space[2] * np.sin(theta)
            points = np.vstack([x_disk, y_disk]).T
            self.mesh = triangulate(points)
        elif form == "ellipse":
            theta = np.linspace(0, 2 * np.pi, 250, endpoint=False)
            x_ellipse = self.space[2] * np.cos(theta)
            y_ellipse = self.space[3] * np.sin(theta)
            points = np.vstack([x_ellipse, y_ellipse]).T
            self.mesh = triangulate(points)
        elif form == "polygon":
            self.mesh = MeshTri.init_lshaped().refined(5)
        self.basis = Basis(self.mesh, ElementTriP1())

        @BilinearForm
        def helmholtz(u, v, w):
            return dot(grad(u), grad(v)) + ((self.k**2) / (self.gamma**2)) * u * v
        
        @LinearForm
        def rhs(v, w):
            x, y = w.x[0], w.x[1]
            rho = (self.Q / (2 * np.pi * self.sigma_x * self.sigma_y)) * np.exp(
                -((x - self.beam[0])**2 / (2 * self.sigma_x**2) + (y - self.beam[1])**2 / (2 * self.sigma_y**2))
            )
            return (self.k / (self.eps0 * self.gamma**2)) * rho * v
        
        A = asm(helmholtz, self.basis)
        b = asm(rhs, self.basis)

        D = self.basis.get_dofs().all()
        A, b = enforce(A, b, D=D)
        self.u_fem = solve(A, b)

        # Reference Solution
        def reference_solution(xy):
            values = np.zeros((xy.shape[0], 1))
            for i, pt in enumerate(xy):
                try:
                    probe = self.basis.probes(pt.reshape(2, 1))
                    values[i] = probe @ self.u_fem
                except ValueError:
                    continue
            return values
        
        self.ref_sol = lambda xy: reference_solution(xy)
        
        # Boundary Condition
        def boundary_space(x, on_boundary):
            if self.form == "polygon":
                return on_boundary and not (x[0] > self.space[0] or x[1] > self.space[2])
            else:
                return on_boundary

        # BCs
        self.add_bcs([{
            'name': 'space',
            'component': 0,
            'function': (lambda x: 0),
            'bc': boundary_space,
            'type': 'dirichlet'
        }])

        # Training Config
        self.training_points()


class Electric_Ritz(Electric_2D):
    def __init__(self, bbox=[-1, 1, -1, 1], gamma=100, frequency=1000000000000, Q=1e-9, eps0=8.854e-12, sigma_x=0.3, sigma_y=0.3, form="disk"):
        super().__init__(bbox, gamma, frequency, Q, eps0, sigma_x, sigma_y, form)

        def rho_transverse(x):
            x, y = x[:, 0:1], x[:, 1:2]
            return (self.Q / (2 * np.pi * self.sigma_x * self.sigma_y)) * dde.backend.exp(
                -((x - self.beam[0])**2 / (2 * self.sigma_x**2) + (y - self.beam[1])**2 / (2 * self.sigma_y**2))
            )

        def pde(x, u):
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_y = dde.grad.jacobian(u, x, i=0, j=1)
            
            rhs = (self.k / (self.eps0 * self.gamma**2)) * rho_transverse(x)

            return [0.5 * (u_x**2 + u_y**2) + 0.5 * (self.k**2 / self.gamma**2) * u**2 - rhs * u]
        
        self.pde = pde
class Electric_3D(baseclass.BasePDE):

    def __init__(self, bbox=[-1, 1, -1, 1, -1, 1], gamma=100.0, frequency = 1e12, Q=1e-9, eps0=8.854e-12, sigma_x=0.3, sigma_y=0.3, sigma_z=0.3, form="sphere"):
        super().__init__()

        # Output Dim
        self.output_config = [{'name': 'Ez'}]

        # Geometry
        self.bbox = bbox
        self.form = form
        if form == "sphere":
            self.space = [0, 0, 0, 1]  # center (x, y, z) and radius
            self.beam = [0, 0, 0]
            self.geom = dde.geometry.Sphere(self.space[0:3], self.space[3])
        elif form == "ellipsoid":
            self.space = [0, 0, 0, 0.6, 1, 0.8]  # center and semi-axes a, b, c
            self.beam = [0, 0.3, 0]
            # Note: DeepXDE might not have direct ellipsoid, use sphere or cuboid
            self.geom = dde.geometry.Sphere(self.space[0:3], self.space[3])
        elif form == "cuboid":
            self.space = [0, 1, 0, 1, 0, 1]
            self.beam = [-0.3, -0.3, -0.3]
            geom = dde.geometry.Cuboid(xmin=[bbox[0], bbox[2], bbox[4]], xmax=[bbox[1], bbox[3], bbox[5]])
            ngeom = dde.geometry.Cuboid(xmin=[self.space[0], self.space[2], self.space[4]], 
                                        xmax=[self.space[1], self.space[3], self.space[5]])
            self.geom = dde.geometry.csg.CSGDifference(geom, ngeom)

        self.gamma = gamma
        self.frequency = frequency
        self.Q = Q
        self.eps0 = eps0
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        c = 3e8
        beta = np.sqrt(1 - 1 / self.gamma**2)
        self.k = 2 * np.pi * self.frequency / (beta * c)

        def rho_transverse(x):
            x_coord, y_coord, z_coord = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            return (self.Q / ((2 * np.pi)**(3/2) * self.sigma_x * self.sigma_y * self.sigma_z)) * dde.backend.exp(
                -((x_coord - self.beam[0])**2 / (2 * self.sigma_x**2) + 
                  (y_coord - self.beam[1])**2 / (2 * self.sigma_y**2) + 
                  (z_coord - self.beam[2])**2 / (2 * self.sigma_z**2))
            )

        # PDE
        def electric_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            u_zz = dde.grad.hessian(u, x, i=2, j=2)
            
            rhs = (self.k / (self.eps0 * self.gamma**2)) * rho_transverse(x)

            return [u_xx + u_yy + u_zz - (self.k**2 / self.gamma**2) * u + rhs]
        
        self.pde = electric_pde
        self.set_pdeloss(names=["pde"])

        # No FEM reference for 3D in this class
        # If a 3D FEM reference is needed, consider using a tetrahedral mesh (e.g., MeshTet)
        # and appropriate basis (e.g., ElementTetP1) in a separate utility.
        self.ref_sol = None
        
        # Boundary Condition
        def boundary_space(x, on_boundary):
            if not on_boundary:
                return False
            if self.form == "cuboid":
                tol = 1e-6
                return (
                    abs(x[0] - self.bbox[0]) < tol or abs(x[0] - self.bbox[1]) < tol or
                    abs(x[1] - self.bbox[2]) < tol or abs(x[1] - self.bbox[3]) < tol or
                    abs(x[2] - self.bbox[4]) < tol or abs(x[2] - self.bbox[5]) < tol
                )
            else:
                return True

        # BCs
        self.add_bcs([{
            'name': 'space',
            'component': 0,
            'function': (lambda x: 0),
            'bc': boundary_space,
            'type': 'dirichlet'
        }])

        # Training Config
        self.training_points()

class Electric_Ritz_3D(Electric_3D):
    def __init__(self, bbox=[-1, 1, -1, 1, -1, 1], gamma=100, frequency=1000000000000, Q=1e-9, eps0=8.854e-12, sigma_x=0.3, sigma_y=0.3, sigma_z=0.3, form="sphere"):
        super().__init__(bbox, gamma, frequency, Q, eps0, sigma_x, sigma_y, sigma_z, form)

        def rho_transverse(x):
            x_coord, y_coord, z_coord = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            return (self.Q / ((2 * np.pi)**(3/2) * self.sigma_x * self.sigma_y * self.sigma_z)) * dde.backend.exp(
                -((x_coord - self.beam[0])**2 / (2 * self.sigma_x**2) + 
                  (y_coord - self.beam[1])**2 / (2 * self.sigma_y**2) + 
                  (z_coord - self.beam[2])**2 / (2 * self.sigma_z**2))
            )

        def pde(x, u):
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_z = dde.grad.jacobian(u, x, i=0, j=2)
            
            rhs = (self.k / (self.eps0 * self.gamma**2)) * rho_transverse(x)

            return [0.5 * (u_x**2 + u_y**2 + u_z**2) + 0.5 * (self.k**2 / self.gamma**2) * u**2 - rhs * u]

        self.pde = pde