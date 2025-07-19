import numpy as np


H = np.array([[1.0, 1.0 - 1e-12],
              [1.0 - 1e-12, 1.0 - 2e-12]])
b = np.array([1.0, 1.0])

# Print condition number
cond = np.linalg.cond(H)
print("Condition number:", cond)

# Try solving
try:
    x = np.linalg.solve(H, b)
    print("Solution x:", x)
except np.linalg.LinAlgError as e:
    print("Solve failed:", e)

# Print actual inverse for comparison
H_inv_exact = (1 / (-1e-8)) * np.array([[0.9998, -0.9999], [-0.9999, 1.0]])
print("Exact inverse H^{-1}:\n", H_inv_exact)

# Compare with NumPy inverse (if you want)
try:
    H_inv_np = np.linalg.inv(H)
    print("NumPy inverse:\n", H_inv_np)
except np.linalg.LinAlgError as e:
    print("Inverse failed:", e)
