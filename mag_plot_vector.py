import matplotlib.pyplot as plt
import numpy as np

def ref_solution(x, y):
        mu0=4*np.pi*1e-7
        I=100.0
        x0, y0 = 0, 0

        u = -(mu0 * I * (y - y0)) / (2 * np.pi * ((x - x0)**2 + (y - y0)**2))
        v = (mu0 * I * (x - x0)) / (2 * np.pi * ((x - x0)**2 + (y - y0)**2))
        return u, v

if __name__ == "__main__":
    #data = np.loadtxt("runs/06.02-14.37.44-adam+epochs/0-0/model_output.txt", comments="#", delimiter=" ")
    #data = np.loadtxt("runs/06.02-15.00.50-adam/0-0/model_output.txt", comments="#", delimiter=" ")
    #data = np.loadtxt("runs/06.02-15.16.29-adam_sigma01/0-0/model_output.txt", comments="#", delimiter=" ")

    #data = np.loadtxt("runs/06.03-13.39.19-adam_dislocated/0-0/model_output.txt", comments="#", delimiter=" ")
    data = np.loadtxt("runs/06.03-13.52.05-adam_ellipse/0-0/model_output.txt", comments="#", delimiter=" ")
    #data = np.loadtxt("runs/06.03-14.39.17-adam_polygon/0-0/model_output.txt", comments="#", delimiter=" ")

    x, y, u, v = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    u_ref, v_ref = ref_solution(x, y)

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
    plt.close()

