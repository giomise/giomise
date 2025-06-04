"""Simple solver for second-order ODEs using RK4 method.

This script provides a function ``solve_second_order`` that integrates
an equation of the form ``y'' = f(t, y, y')`` with given initial
conditions. It uses the fourth-order Runge-Kutta method.

Example usage:

```
# Solve y'' = -y with y(0)=0, y'(0)=1
from ode_solver import solve_second_order
import math

def f(t, y, v):
    return -y

solution = solve_second_order(f, t0=0.0, y0=0.0, v0=1.0, t_end=10.0, dt=0.1)
for t, y, v in solution:
    print(f"{t:.2f} {y:.6f} {v:.6f}")
```
"""

from typing import Callable, List, Tuple


def solve_second_order(
    f: Callable[[float, float, float], float],
    t0: float,
    y0: float,
    v0: float,
    t_end: float,
    dt: float,
) -> List[Tuple[float, float, float]]:
    """Solve ``y'' = f(t, y, y')`` from ``t0`` to ``t_end``.

    Parameters
    ----------
    f : callable
        Function ``f(t, y, v)`` returning the second derivative.
    t0 : float
        Initial time.
    y0 : float
        Initial position ``y(t0)``.
    v0 : float
        Initial velocity ``y'(t0)``.
    t_end : float
        Final time to integrate to.
    dt : float
        Time step for integration.

    Returns
    -------
    list of tuples
        Sequence of ``(t, y, v)`` points including the initial state.
    """
    t = t0
    y = y0
    v = v0
    out = []

    while t <= t_end + 1e-12:
        out.append((t, y, v))

        # Helper to compute derivatives
        def deriv(tt: float, yy: float, vv: float) -> Tuple[float, float]:
            return vv, f(tt, yy, vv)

        k1_y, k1_v = deriv(t, y, v)
        k2_y, k2_v = deriv(t + dt / 2, y + k1_y * dt / 2, v + k1_v * dt / 2)
        k3_y, k3_v = deriv(t + dt / 2, y + k2_y * dt / 2, v + k2_v * dt / 2)
        k4_y, k4_v = deriv(t + dt, y + k3_y * dt, v + k3_v * dt)

        y += dt / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        v += dt / 6 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        t += dt

    return out


if __name__ == "__main__":
    import math

    def example_f(t: float, y: float, v: float) -> float:
        """Example equation ``y'' = -y`` (harmonic oscillator)."""
        return -y

    sol = solve_second_order(example_f, t0=0.0, y0=0.0, v0=1.0, t_end=2 * math.pi, dt=0.1)
    for t, y, v in sol:
        print(f"{t:.2f} {y:.6f} {v:.6f}")
