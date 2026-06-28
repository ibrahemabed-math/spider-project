import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
mu = 0.02
beta = 0.45
gamma = 0.08
alpha = 0.25
sigma = 0.3

# Initial conditions
P0 = 0.60
S0 = 0.25
Qt0 = 0.10
Qp0 = 0.05
y0 = [P0, S0, Qt0, Qp0]

# Model
def smoking_model(t, y):
    P, S, Qt, Qp = y

    dP = mu - mu * P - beta * P * S
    dS = -(mu + gamma) * S + beta * P * S + alpha * S * Qt
    dQt = -mu * Qt - alpha * S * Qt + gamma * (1 - sigma) * S
    dQp = -mu * Qp + gamma * sigma * S

    return [dP, dS, dQt, dQp]

# Time interval עד 200
t_span = (0, 200)
t_eval = np.linspace(0, 200, 1000)

# Solve
sol = solve_ivp(smoking_model, t_span, y0, t_eval=t_eval)

t = sol.t
P = sol.y[0]
S = sol.y[1]
Qt = sol.y[2]
Qp = sol.y[3]

# Final values
print(f"Final values at t = {t[-1]:.1f}")
print(f"P(t)  = {P[-1]:.4f}")
print(f"S(t)  = {S[-1]:.4f}")
print(f"Qt(t) = {Qt[-1]:.4f}")
print(f"Qp(t) = {Qp[-1]:.4f}")

# One graph for all
plt.figure(figsize=(10, 6))
plt.plot(t, P, label='P', linewidth=2)
plt.plot(t, S, label='S', linewidth=2)
plt.plot(t, Qt, label='Qt', linewidth=2)
plt.plot(t, Qp, label='Qp', linewidth=2)

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Smoking Model')
plt.xlim(0, 200)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
