"""
Numerical solution methods for Initial Value Problems of the form y'(t)=f(t,y(t)), y(t_0)=y_0.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def solve_ivp(ode, solver, y0, t0, t_end, h):
    num_steps = int((t_end - t0) / h) + 1
    y = np.zeros(num_steps)
    t = np.zeros(num_steps)

    y[0] = y0
    t[0] = t0

    y_current = y0
    t_current = t0

    for i in range(1, num_steps):
        y_current, t_current = solver(ode, y_current, t_current, h)
        y[i] = y_current
        t[i] = t_current

    return y, t


def ode(y, t):
    """Sample ODE function."""
    return y


def euler(ode, y_current, t_current, h):
    """Euler method for numerical integration."""
    y_next = y_current + h * ode(y_current, t_current)
    return (y_next, t_current + h)


def midpoints(ode, y_current, t_current, h):
    """Midpoint method for numerical integration."""
    y_next = y_current + h * ode(
        y_current + h / 2 * ode(y_current, t_current), t_current + h / 2
    )
    return (y_next, t_current + h)


def runge_kutta_classic(ode, y_current, t_current, h):
    """Runge-Kutta 4th order method for numerical integration."""
    # calculating slopes for weighted average
    k1 = ode(y_current, t_current)
    k2 = ode(y_current + (h * k1 / 2), t_current + h / 2)
    k3 = ode(y_current + (h * k2 / 2), t_current + h / 2)
    k4 = ode(y_current + h * k3, t_current + h)

    y_next = y_current + (h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return (y_next, t_current + h)


if __name__ == "__main__":
    y0 = 1
    t0 = 0
    t_end = 4
    h = 1

    methods = {
        "Euler": euler,
        "Midpoints": midpoints,
        "Runge-Kutta": runge_kutta_classic,
    }

    results = {
        name: solve_ivp(ode, method, y0, t0, t_end, h)
        for name, method in methods.items()
    }

    for name, (y_values, t_values) in results.items():
        print(f"y({round(t_values[-1],2)}) = {round(y_values[-1],2)}")

    sns.set_theme()
    for name, (y_values, t_values) in results.items():
        sns.lineplot(x=t_values, y=y_values, label=name)

    plt.xlim(0)
    plt.ylim(0)
    plt.legend()
    plt.show()
