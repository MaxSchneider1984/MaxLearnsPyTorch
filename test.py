import torch
print(F"torch.__version__: {torch.__version__}")

# Scalar
s = torch.tensor(7)
print("===== Scalar (7) =====")
print(F"s: {s}")
print(F"s.ndim: {s.ndim}")
print(F"s.item(): {s.item()}")

# Vector
v = torch.tensor([7, 7])
print("===== Vector ([7, 7]) =====")
print(F"v: {v}")
print(F"v.ndim: {v.ndim}")
print(F"v.shape: {v.shape}")

# MATRIX
M = torch.tensor([[7, 8], [9, 10]])
print("===== Matrix ([[7, 8], [9, 10]]) =====")
print(F"M: {M}")
print(F"M.ndim: {M.ndim}")
print(F"M.shape: {M.shape}")
print(F"M[1]: {M[1]}")
