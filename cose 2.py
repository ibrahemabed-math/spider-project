import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# Smoking Model with Age Groups
# Groups:
# 1 - Youth
# 2 - Young adults
# 3 - Adults
# ============================================================

# ============================================================
# Initial normalized data
# ============================================================

N = np.array([0.3323, 0.1045, 0.5633])

P0 = np.array([0.2268, 0.0460, 0.2267])
S0 = np.array([0.0201, 0.0251, 0.1378])
Q0 = np.array([0.0781, 0.0272, 0.1364])
QP0 = np.array([0.0073, 0.0062, 0.0624])

# Combine all variables into one initial vector
y0 = np.concatenate([P0, S0, Q0, QP0])

# Small normalization correction because the data sum is not exactly 1 due to rounding
y0 = y0 / np.sum(y0)

# After normalization, update initial values used inside the model
P0 = y0[0:3]
S0 = y0[3:6]
Q0 = y0[6:9]
QP0 = y0[9:12]

# ============================================================
# Empirical data from the chart
# ============================================================

G = np.array([0.045, 0.0636, 0.0426])
R = np.array([0.022, 0.023, 0.0294])

# Mortality rates
mu = np.array([0.0003, 0.0005, 0.008])

# Aging transition rates
delta1 = 1 / 18   # Group 1 -> Group 2
delta2 = 1 / 7    # Group 2 -> Group 3

# ============================================================
# Calculate gamma and alpha
# ============================================================

gamma = N * G / S0
alpha = N * R / (S0 * Q0)
print("gamma =", gamma)
print("alpha =", alpha)

# ============================================================
# Sigma
# ============================================================

# Sigma estimated from CDC report:
# CDC "Recent successful cessation" = quit smoking for at least 6 months.
sigma = np.array([0.10, 0.153, 0.079])

print("sigma =", sigma)

# ============================================================
# Social influence weight matrix C
# ============================================================

C = np.array([
    [0.6,  0.3,  0.1],
    [0.25, 0.5,  0.25],
    [0.1,  0.3,  0.6]
])

# ============================================================
# Delta values for smoking initiation
# ============================================================

Delta = np.array([0.1015, 0.038, 0.005])

# ============================================================
# Calculate beta_i and influence matrix B
#
# New formula:
# beta_ij = beta_i * C_ij * (1 / N_i)
#
# Therefore:
# Delta_i = (beta_i / N_i) * sum_j C_ij S_j
#
# Let:
# D_i = sum_j C_ij S_j
#
# Then:
# beta_i = Delta_i * N_i / D_i
# ============================================================

D = C @ S0

# New beta_i calculation
beta = Delta * N / D

# New influence matrix B
B = beta[:, None] * C / N[:, None]

print("D =", D)
print("beta =", beta)
print("B =")
print(B)

# ============================================================
# Helper function for numerical normalization
# ============================================================

def normalize_state(y):
    """
    Keeps all values non-negative and normalizes the total population to 1.
    """
    y = np.maximum(y, 0)
    total = np.sum(y)

    if total > 0:
        y = y / total

    return y

# ============================================================
# Differential equations
# ============================================================

def smoking_model(t, y):
    P = y[0:3]
    S = y[3:6]
    Q = y[6:9]
    QP = y[9:12]

    # Prevent very small numerical negative values
    P = np.maximum(P, 0)
    S = np.maximum(S, 0)
    Q = np.maximum(Q, 0)
    QP = np.maximum(QP, 0)

    # Smoking influence from all groups on each group
    influence = B @ S

    # Births enter the youngest group.
    # To preserve total normalization, births equal total deaths.
    total_deaths = np.sum(mu * (P + S + Q + QP))
    birth = total_deaths

    dP = np.zeros(3)
    dS = np.zeros(3)
    dQ = np.zeros(3)
    dQP = np.zeros(3)

    # ========================================================
    # Group 1 - Youth
    # ========================================================

    dP[0] = (
        birth
        - P[0] * influence[0]
        - mu[0] * P[0]
        - delta1 * P[0]
    )

    dS[0] = (
        P[0] * influence[0]
        - gamma[0] * S[0]
        + alpha[0] * S[0] * Q[0]
        - mu[0] * S[0]
        - delta1 * S[0]
    )

    dQ[0] = (
        gamma[0] * (1 - sigma[0]) * S[0]
        - alpha[0] * S[0] * Q[0]
        - mu[0] * Q[0]
        - delta1 * Q[0]
    )

    dQP[0] = (
        gamma[0] * sigma[0] * S[0]
        - mu[0] * QP[0]
        - delta1 * QP[0]
    )

    # ========================================================
    # Group 2 - Young adults
    # ========================================================

    dP[1] = (
        delta1 * P[0]
        - P[1] * influence[1]
        - mu[1] * P[1]
        - delta2 * P[1]
    )

    dS[1] = (
        delta1 * S[0]
        + P[1] * influence[1]
        - gamma[1] * S[1]
        + alpha[1] * S[1] * Q[1]
        - mu[1] * S[1]
        - delta2 * S[1]
    )

    dQ[1] = (
        delta1 * Q[0]
        + gamma[1] * (1 - sigma[1]) * S[1]
        - alpha[1] * S[1] * Q[1]
        - mu[1] * Q[1]
        - delta2 * Q[1]
    )

    dQP[1] = (
        delta1 * QP[0]
        + gamma[1] * sigma[1] * S[1]
        - mu[1] * QP[1]
        - delta2 * QP[1]
    )

    # ========================================================
    # Group 3 - Adults
    # ========================================================

    dP[2] = (
        delta2 * P[1]
        - P[2] * influence[2]
        - mu[2] * P[2]
    )

    dS[2] = (
        delta2 * S[1]
        + P[2] * influence[2]
        - gamma[2] * S[2]
        + alpha[2] * S[2] * Q[2]
        - mu[2] * S[2]
    )

    dQ[2] = (
        delta2 * Q[1]
        + gamma[2] * (1 - sigma[2]) * S[2]
        - alpha[2] * S[2] * Q[2]
        - mu[2] * Q[2]
    )

    dQP[2] = (
        delta2 * QP[1]
        + gamma[2] * sigma[2] * S[2]
        - mu[2] * QP[2]
    )

    dydt = np.concatenate([dP, dS, dQ, dQP])

    # Numerical correction:
    # makes sure the sum of derivatives is exactly zero
    dydt = dydt - np.sum(dydt) * y / np.sum(y)

    return dydt

# ============================================================
# Solve the system
# ============================================================

t_span = (0, 80)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(
    smoking_model,
    t_span,
    y0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-8,
    atol=1e-10
)

# ============================================================
# Normalize the solution after solving
# ============================================================

Y = sol.y.copy()

for k in range(Y.shape[1]):
    Y[:, k] = normalize_state(Y[:, k])

P = Y[0:3, :]
S = Y[3:6, :]
Q = Y[6:9, :]
QP = Y[9:12, :]

total = np.sum(Y, axis=0)

# ============================================================
# Calculate percentages inside each age group
#
# Group total:
# N_i(t) = P_i(t) + S_i(t) + Q_i(t) + QP_i(t)
#
# Percentage of each variable inside its group:
# X_i%(t) = X_i(t) / N_i(t) * 100
# ============================================================

group_total = P + S + Q + QP

P_percent = P / group_total * 100
S_percent = S / group_total * 100
Q_percent = Q / group_total * 100
QP_percent = QP / group_total * 100

# ============================================================
# Print normalization check and final values
# ============================================================

print("\nNormalization check:")
print("Initial total =", total[0])
print("Final total   =", total[-1])
print("Min total     =", np.min(total))
print("Max total     =", np.max(total))

print("\nFinal values as proportion of total population:")
print("Final P  =", P[:, -1])
print("Final S  =", S[:, -1])
print("Final Q  =", Q[:, -1])
print("Final QP =", QP[:, -1])

print("\nFinal percentages inside each age group:")

groups = ["Youth", "Young adults", "Adults"]

for i, group in enumerate(groups):
    print("\n" + group)
    print("P%  =", P_percent[i, -1])
    print("S%  =", S_percent[i, -1])
    print("Q%  =", Q_percent[i, -1])
    print("QP% =", QP_percent[i, -1])
    print("Total =", (
        P_percent[i, -1]
        + S_percent[i, -1]
        + Q_percent[i, -1]
        + QP_percent[i, -1]
    ))

# ============================================================
# Graph 1 - Model dynamics by age group
# Values are proportions out of the total population
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

# Group 1
axes[0].plot(sol.t, P[0], label="P1 - Potential smokers")
axes[0].plot(sol.t, S[0], label="S1 - Active smokers")
axes[0].plot(sol.t, Q[0], label="Q1 - Temporary quitters")
axes[0].plot(sol.t, QP[0], label="QP1 - Permanent quitters")
axes[0].set_title("Group 1 - Youth")
axes[0].set_ylabel("Proportion of total population")
axes[0].legend()
axes[0].grid(True)

# Group 2
axes[1].plot(sol.t, P[1], label="P2 - Potential smokers")
axes[1].plot(sol.t, S[1], label="S2 - Active smokers")
axes[1].plot(sol.t, Q[1], label="Q2 - Temporary quitters")
axes[1].plot(sol.t, QP[1], label="QP2 - Permanent quitters")
axes[1].set_title("Group 2 - Young adults")
axes[1].set_ylabel("Proportion of total population")
axes[1].legend()
axes[1].grid(True)

# Group 3
axes[2].plot(sol.t, P[2], label="P3 - Potential smokers")
axes[2].plot(sol.t, S[2], label="S3 - Active smokers")
axes[2].plot(sol.t, Q[2], label="Q3 - Temporary quitters")
axes[2].plot(sol.t, QP[2], label="QP3 - Permanent quitters")
axes[2].set_title("Group 3 - Adults")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Proportion of total population")
axes[2].legend()
axes[2].grid(True)

plt.suptitle(
    "Smoking Model Dynamics by Age Group - Proportion of Total Population",
    fontsize=16
)
plt.tight_layout()
plt.show()

# ============================================================
# Graph 2 - Percentage of each variable inside its own age group
# Values are percentages within the relevant group
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

# Group 1
axes[0].plot(sol.t, P_percent[0], label="P1% - Potential smokers")
axes[0].plot(sol.t, S_percent[0], label="S1% - Active smokers")
axes[0].plot(sol.t, Q_percent[0], label="Q1% - Temporary quitters")
axes[0].plot(sol.t, QP_percent[0], label="QP1% - Permanent quitters")
axes[0].set_title("Group 1 - Youth")
axes[0].set_ylabel("Percentage within group (%)")
axes[0].legend()
axes[0].grid(True)

# Group 2
axes[1].plot(sol.t, P_percent[1], label="P2% - Potential smokers")
axes[1].plot(sol.t, S_percent[1], label="S2% - Active smokers")
axes[1].plot(sol.t, Q_percent[1], label="Q2% - Temporary quitters")
axes[1].plot(sol.t, QP_percent[1], label="QP2% - Permanent quitters")
axes[1].set_title("Group 2 - Young adults")
axes[1].set_ylabel("Percentage within group (%)")
axes[1].legend()
axes[1].grid(True)

# Group 3
axes[2].plot(sol.t, P_percent[2], label="P3% - Potential smokers")
axes[2].plot(sol.t, S_percent[2], label="S3% - Active smokers")
axes[2].plot(sol.t, Q_percent[2], label="Q3% - Temporary quitters")
axes[2].plot(sol.t, QP_percent[2], label="QP3% - Permanent quitters")
axes[2].set_title("Group 3 - Adults")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Percentage within group (%)")
axes[2].legend()
axes[2].grid(True)

plt.suptitle(
    "Smoking Model Dynamics by Age Group - Percentage Within Each Group",
    fontsize=16
)
plt.tight_layout()
plt.show()