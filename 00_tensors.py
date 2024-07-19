import torch
print(F"torch.__version__: {torch.__version__}")

# Scalar
s = torch.tensor(7)
print("===== Scalar =====")
print(F"s: {s}")
print(F"s.ndim: {s.ndim}")
print(F"s.item(): {s.item()}")

# Vector
v = torch.tensor([7, 7])
print("===== Vector =====")
print(F"v: {v}")
print(F"v.ndim: {v.ndim}")
print(F"v.shape: {v.shape}")

# Matrix
M = torch.tensor([[7, 8], [9, 10]])
print("===== Matrix =====")
print(F"M: {M}")
print(F"M.ndim: {M.ndim}")
print(F"M.shape: {M.shape}")
print(F"M[1]: {M[1]}")

# Tensor
T = torch.tensor([[[1, 2, 3],
                   [3, 6, 9],
                   [2, 4, 5]]])
print("===== Tensor =====")
print(F"T: {T}")
print(F"T.ndim: {T.ndim}")
print(F"T.shape: {T.shape}")
print(F"T[0]: {T[0]}")

# Random Tensors
# ...are used by neural networks as starting point and will be updated when they look at data to learn
R_T = torch.rand(3, 4)
print("===== Random Tensor =====")
print(F"R_T: {R_T}")
print(F"R_T.ndim: {R_T.ndim}")
print(F"R_T.shape: {R_T.shape}")
print(F"R_T[0]: {R_T[0]}")

# Random tensor with similar shape to an image tensor
R_T_IMAGE = torch.rand(size=(224, 224, 3)) # height, width, colour channels
print("===== Random Image Tensor =====")
print(F"R_T_IMAGE: {R_T_IMAGE}")
print(F"R_T_IMAGE.ndim: {R_T_IMAGE.ndim}")
print(F"R_T_IMAGE.shape: {R_T_IMAGE.shape}")
