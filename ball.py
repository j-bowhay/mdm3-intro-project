from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


@dataclass
class BallSimParameters:
    radius: float
    mass: float
    stokes_drag_coeff: float
    water_density: float
    g: float
    t_end: float
    n_points: int
    drag_coeff: float

    @property
    def volume(self):
        return (4 / 3) * np.pi * self.radius**3

    @property
    def area(self):
        return np.pi * self.radius**2


def stokes_analytical_sol_n(params: BallSimParameters):
    t = np.linspace(0, params.t_end, params.n_points)
    const = (1 - (params.water_density * params.volume) / (params.mass)) * (
        (params.g * params.mass) / params.stokes_drag_coeff
    )
    v = const * (1 - np.exp(-(params.stokes_drag_coeff / params.mass) * t))
    return v, t


def cda_num_sol_n(params: BallSimParameters):
    b = (params.water_density * params.drag_coeff * params.area)/(2*params.mass)
    c = params.g - (params.water_density * params.g * params.volume)/params.mass
    def rhs(t, y):
        return -b*y[0]**2 + c
    sol = solve_ivp(rhs, (0, params.t_end), y0=[0], method='LSODA', rtol=1e-9)
    return sol.y[0], sol.t
    
    
def main():
    params = BallSimParameters(
        radius=0.0015,
        mass=0.11e-3,
        stokes_drag_coeff=0.0251e-3,
        water_density=997,
        g=9.81,
        t_end=55,
        n_points=50,
        drag_coeff=0.47
    )
    v_stokes_anal, t_stokes = stokes_analytical_sol_n(params)
    v_drag_anal, t_drag = cda_num_sol_n(params)

    data = np.genfromtxt("velData.csv", delimiter=",")

    plt.style.use(['science', 'ieee'])
    fig, ax = plt.subplots()
    ax.plot(t_stokes, v_stokes_anal, label="Stokes' Law Model")
    ax.plot(t_drag, v_drag_anal, 'r-', label="Drag Coefficient Model", zorder=10)
    ax.plot(data[:, 0], data[:, 1], "bx", label="Experimental Data", markersize=2)
    ax.set_xlim((0, params.t_end))
    ax.set_ylim((0, 40))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    plt.legend()
    plt.tight_layout
    plt.savefig("all_comparison.pdf")

    fig, ax = plt.subplots()
    ax.plot(t_drag, v_drag_anal, 'r-', label="Drag Coefficient Model", zorder=10)
    ax.plot(data[:, 0], data[:, 1], "bx", label="Experimental Data", markersize=2)
    ax.set_xlim((0, params.t_end))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_ylim((0, 0.9))
    plt.legend()
    plt.tight_layout
    plt.savefig("better_comparison.pdf")


if __name__ == "__main__":
    main()
