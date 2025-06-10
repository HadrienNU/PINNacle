import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from scipy.special import kv
from scipy.signal import fftconvolve
from scipy.interpolate import RegularGridInterpolator


def mag_ref_solution(x, y):
    mu0=4*np.pi*1e-7
    I=100.0
    x0, y0 = 0, 0

    u = -(mu0 * I * (y - y0)) / (2 * np.pi * ((x - x0)**2 + (y - y0)**2))
    v = (mu0 * I * (x - x0)) / (2 * np.pi * ((x - x0)**2 + (y - y0)**2))
    return u, v

def electric_ref_solution(xy):
    bbox=[-1, 1, -1, 1]
    Q=1e-9
    sigma_x=0.3
    sigma_y=0.3
    beam = [0, 0]
    gamma=100.0
    c = 3e8
    beta = np.sqrt(1 - 1 / gamma**2)
    frequency = 1e12
    eps0=8.854e-12
    k = 2 * np.pi * frequency / (beta * c)

    grid_res = 256
    x = np.linspace(bbox[0], bbox[1], grid_res)
    y = np.linspace(bbox[2], bbox[3], grid_res)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)

    rho = (Q / (2 * np.pi * sigma_x * sigma_y)) * np.exp(
        -((X - beam[0])**2 / (2 * sigma_x**2) + (Y - beam[1])**2 / (2 * sigma_y**2))
    )

    R = np.sqrt((X - beam[0])**2 + (Y - beam[1])**2) + 1e-10
    G = (1 / (2 * np.pi)) * kv(0, (k / gamma) * R)

    Ez = (k / (eps0 * gamma**2)) * fftconvolve(rho, G, mode='same') * dx * dy

    interp = RegularGridInterpolator((x, y), Ez.T, bounds_error=False, fill_value=0)

    Ez = interp(xy)

    return Ez

if __name__ == "__main__":
    '''#data = np.loadtxt("runs/06.02-14.37.44-adam+epochs/0-0/model_output.txt", comments="#", delimiter=" ")
    #data = np.loadtxt("runs/06.02-15.00.50-adam/0-0/model_output.txt", comments="#", delimiter=" ")
    #data = np.loadtxt("runs/06.02-15.16.29-adam_sigma01/0-0/model_output.txt", comments="#", delimiter=" ")
    #data = np.loadtxt("runs/06.03-13.39.19-adam_dislocated/0-0/model_output.txt", comments="#", delimiter=" ")
    data = np.loadtxt("runs/06.03-13.52.05-adam_ellipse/0-0/model_output.txt", comments="#", delimiter=" ")
    #data = np.loadtxt("runs/06.03-14.39.17-adam_polygon/0-0/model_output.txt", comments="#", delimiter=" ")

    x, y, u, v = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    u_ref, v_ref = mag_ref_solution(x, y)

    color = np.sqrt((u)**2 + (v)**2)
    color_ref = np.sqrt((u_ref)**2 + (v_ref)**2)

    plt.subplot(1, 2, 1)
    plt.quiver(x, y, u_ref, v_ref, color_ref)
    plt.gca().set_aspect("equal")
    plt.title("Reference Solution Vectors")

    plt.subplot(1, 2, 2)
    plt.quiver(x, y, u, v, color)
    plt.gca().set_aspect("equal")
    plt.title("Model Output Vectors")

    plt.show()
    plt.close()'''

    data = np.loadtxt("runs/06.05-15.12.49-electric-adam-disk/0-0/model_output.txt", comments="#", delimiter=" ")
    #data = np.loadtxt("runs/06.05-15.30.50-electric-adam-ellipse/0-0/model_output.txt", comments="#", delimiter=" ")
    #data = np.loadtxt("runs/06.05-15.52.43-electric-adam-polygon/0-0/model_output.txt", comments="#", delimiter=" ")

    x, y, o = data[:, 0], data[:, 1], data[:, 2]
    xy = data[:, 0:2]
    o_ref = electric_ref_solution(xy)

    xx = np.linspace(np.min(x), np.max(x), 100)
    yy = np.linspace(np.min(y), np.max(y), 100)
    X, Y = np.meshgrid(xx, yy)

    Z = griddata((x, y), o, (X, Y), method='cubic')
    Z_ref = griddata((x, y), o_ref, (X, Y), method='cubic')

    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, Z_ref, shading='auto', cmap='viridis')
    plt.colorbar(label='Charge')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title("Reference Solution Heatmap")

    plt.subplot(1, 2, 2)
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    plt.colorbar(label='Charge')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title("Model Output Heatmap")

    plt.tight_layout()
    plt.show()
    plt.close()
