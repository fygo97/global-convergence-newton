import numpy as np
import matplotlib.pyplot as plt
# This script computes the L_semi that we need for the AICN
def f(w):
    numerator = -12 * w * (1 + w**2)**3 - 12 * w * (1 - 3 * w**2) * (1 + w**2)**2
    numerator = numerator * 0.001
    denominator = (1 + w**2)**6
    return numerator / denominator

# Generate values
w_vals = np.linspace(-10, 10, 1000)
f_vals = f(w_vals)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(w_vals, f_vals, label=r"$\frac{d h_j}{d \omega_j}$", color='blue')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.title("Derivative of $h_j$ with α = 1, λ = 0,001")
plt.xlabel("$\omega_j$")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
