import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Numerical solution methods for Initial Value Problems of the form y'(t)=f(t,y(t)), y(t_0)=y_0.
'''

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

    return y,t 

def ode(y, t):
    return y

def euler(ode, y_current, t_current, h):
    y_next = y_current + h * ode(y_current, t_current)
    return (y_next, t_current + h) 

def midpoints(ode, y_current, t_current, h):
    y_next = y_current + h * ode(y_current + h/2 * ode(y_current, t_current), t_current + h/2)
    return (y_next, t_current + h) 

def runge_kutta_classic(ode, y_current, t_current, h):
    #calculating slopes for weighted average
    k1 = ode(y_current, t_current)
    k2 = ode(y_current + (h * k1/2), t_current + h/2)
    k3 = ode(y_current + (h * k2/2), t_current + h/2)
    k4 = ode(y_current + h * k3, t_current + h)

    y_next = y_current + (h/6 * (k1 + 2*k2 + 2*k3 + k4))
    return (y_next, t_current + h)

if __name__ == "__main__":
    y0 = 1
    t0 = 0
    t_end = 4
    h = 1
    y_values, t_values = solve_ivp(ode, euler, y0, t0, t_end, h)
    #print("t values: ", t_values, "y values: ", y_values)
    print(f"y({round(t_values[-1],2)}) = {round(y_values[-1],2)}")

    y_midpoints, t_midpoints = solve_ivp(ode, midpoints, y0, t0, t_end, h)
    print(f"y({round(t_midpoints[-1],2)}) = {round(y_midpoints[-1],2)}")

    y_rk4, t_rk4 = solve_ivp(ode, runge_kutta_classic, y0, t0, t_end, h)
    print(f"y({round(t_rk4[-1],2)}) = {round(y_rk4[-1],2)}")

    sns.set_theme()
    sns.lineplot({"y values":y_values, "t values":t_values}, x = "t values", y = "y values", label="euler")
    sns.lineplot({"y values":y_midpoints, "t values":t_midpoints}, x = "t values", y = "y values", label="midpoints")
    sns.lineplot({"y values":y_rk4, "t values":t_rk4}, x = "t values", y = "y values", label="runge-kutta")
    plt.xlim(0)
    plt.ylim(0)
    plt.legend()
    plt.show()