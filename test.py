import torch
print(F"torch.__version__: {torch.__version__}")

# Scalar
scalar = torch.tensor(7)
print("scalar = torch.tensor(7)")
print(F"scalar: {scalar}")
print(F"scalar.ndim: {scalar.ndim}")
print(F"scalar.item(): {scalar.item()}")

# Vector
vector = torch.tensor([7, 7])
print("vector = torch.tensor([7, 7])")
print(F"vector: {vector}")
print(F"vector.ndim: {vector.ndim}")
print(F"vector.shape: {vector.shape}")

# MATRIX
MATRIX = torch.tensor([[7, 8], [9, 10]])
print("MATRIX = torch.tensor([[7, 8], [9, 10]])")
print(F"MATRIX.ndim: {MATRIX.ndim}")
print(F"MATRIX[1]: {MATRIX[1]}")
print(F"MATRIX.shape: {MATRIX.shape}")