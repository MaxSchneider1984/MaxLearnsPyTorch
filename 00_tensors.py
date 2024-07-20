import torch
import numpy as np

print(f"torch.__version__: {torch.__version__}")

print("\n===== Scalar =====")
s = torch.tensor(7)
print(s.item())
print(f"ndim: {s.ndim}")

print("\n===== Vector =====")
v = torch.tensor([7, 7])
print(v)
print(f"ndim: {v.ndim}")
print(f"shape: {v.shape[0]}")

print("\n===== Matrix =====")
M = torch.tensor([[7, 8], [9, 10]])
print(M)
print(f"ndim: {M.ndim}")
print(f"shape: {M.shape}")
print(f"[1]: {M[1]}")

print("\n===== Tensor =====")
T = torch.tensor([[[1, 2, 3],
                   [3, 6, 9],
                   [2, 4, 5]]])
print(T)
print(f"ndim: {T.ndim}")
print(f"shape: {T.shape}")
print(f"[0]: {T[0]}")

print("\n===== Random Tensor =====")
print("Random Tensors are used by neural networks as starting point and will be updated when they look at data to learn.")
R_T = torch.rand(3, 4)
print(R_T)
print(f"ndim: {R_T.ndim}")
print(f"shape: {R_T.shape}")
print(f"[0]: {R_T[0]}")

print("\n===== Random Image Tensor =====")
R_T_IMAGE = torch.rand(size=(224, 224, 3))
print(f"ndim: {R_T_IMAGE.ndim}")
print(f"shape: {R_T_IMAGE.shape} - height, width, colour channels")
print(R_T_IMAGE)

print("\n===== Zeros Tensor =====")
print("Zeros and ones in tensors are often used for masks.")
print(torch.zeros(size=(3, 4)))

print("\n===== Ones Tensor =====")
print(torch.ones(size=(3, 4)))

print("\n===== Range Tensor 1 to 10 =====")
ONE_TO_TEN = torch.arange(start=1, 
                          end=11, 
                          step=1)
print(ONE_TO_TEN)

print("\n===== Zeros Tensor Like Range =====")
print(torch.zeros_like(input=ONE_TO_TEN))

print("\n===== Tensor Datatypes: Float 32 =====")
FLOAT_32_T = torch.tensor([3.0, 6.0, 9.0], 
                          dtype=None, 
                          device=None, 
                          requires_grad=False)
print(FLOAT_32_T)
print(f"dtype: {FLOAT_32_T.dtype}")

print("\n===== Tensor Datatypes: Float 16 =====")
FLOAT_16_T = FLOAT_32_T.type(torch.float16)
print(FLOAT_16_T)
print(f"dtype: {FLOAT_16_T.dtype}")

print("\n===== Tensor Gradient Computation =====")
print("With requires_grad=True PyTorch tracks operations on a tensor, it builds a computational graph")
print("and when you call .backward() on the tensor PyTorch computes the gradients of loss, which is used when training neural networks.")
X = torch.tensor(2.0, requires_grad=True)
# Perform some operations on the tensor
Y = X ** 2
Z = Y + 3
# Compute the gradients
Z.backward()
print(f"Z = X^2 + 3 auto differentiation for X = 2.0: {X.grad}")

print("\n===== Manipulating Tensors / Tensor Operations =====")
T = torch.tensor([1, 4, 9])
print(f"T: {T}")

print("\n===== Tensor Addition =====")
T_ADD = T + 10 
print(f"T + 10: {T_ADD}")

print("\n===== Tensor Subtraction =====")
T_SUB = T - 10
print(f"T - 10: {T_SUB}")

print("\n===== Tensor Multiplication (element-wise) =====")
T_MUL = T * 10
print(f"T * 10: {T_MUL}")
T_MUL = T * T
print(f"T * T: {T_MUL}")

print("\n===== Tensor Division =====")
T_DIV = T / 10
print(f"T / 10: {T_DIV}")

print("\n===== Tensor Matrix Multiplication (Dot Product) =====")
print(f"T.matmul(T): {T.matmul(T)}")
calc_expl = "Matrix multiplication by hand: "
value = 0
for i in range(len(T)):
    element = T[i]
    value += element * element
    calc_expl += f"{element} * {element} ({element * element})"
    if i < len(T) - 1:
        calc_expl += " + "
print(f"{calc_expl} = {value}")
print("In matrix multiplication inner dimensions must match and outer dimensions determine the resulting shape.")
# (3, 2) @ (3, 2) won't work
# (2, 10) @ (10, 2) -> (2, 2)
# (3, 2) @ (2, 3) -> (3, 3)
print("Multiply random matrix (2, 10) with (10, 2):")
print((torch.rand(2, 10) @ torch.rand(10, 2)))
print("Multiply random matrix (3, 2) with (2, 3):")
print((torch.rand(3, 2) @ torch.rand(2, 3)))

print("\n===== Tensor Transpose =====")
T_A = torch.tensor([[1, 2],
                    [3, 4],
                    [5, 6]])
print(f"Tensor A: \n{T_A}")
T_B = torch.tensor([[7, 10],
                    [8, 11],
                    [9, 12]])
print(f"Tensor B: \n{T_B}")
print(f"In order to multiply two tensors of shape {T_A.shape} and {T_B.shape} we need to transpose one of them.")
print(f"Tensor B transposed: \n {T_B.T}")
print(f"...with the new shape: {T_B.T.shape} it can now be multiplied with tensor A:")
print(torch.matmul(T_A, T_B.T))

print("\n===== Tensor Aggregation =====")
X = torch.arange(1, 100, 10)
print(X)
print(f"Min: {X.min()}")
print(f"Max: {X.max()}")
print(f"Mean: {X.mean(dtype=torch.float32)} (needs type conversion e.g. to float32)")
print(f"Sum: {X.sum()}")
print(f"Position of min: {X.argmin()}")
print(f"Position of max: {X.argmax()}")

print("\n===== Tensor Reshaping =====")
X = torch.arange(1, 10)
print(f"(1, 9): \n{X}")
print(f"(9, 1): \n{X.reshape(9, 1)}")

print("\n===== Tensor View =====")
print("Views only work for contiguous tensors i.e. when their elements are stored sequentially in memory.")
print(f"(3, 3): \n{X.view(3, 3)}")

print("\n===== Tensor Stacking =====")
print(f"dim = 0: \n{torch.stack([X, X, X, X])}")
print(f"dim = 1: \n{torch.stack([X, X, X, X], dim=1)}")

print("\n===== Tensor Squeeze (remove dimensions of size 1) =====")
X = torch.zeros(2, 1, 4, 1, 3)
print(f"X is a zeros tensor of shape: {X.shape}")
print(f"X.squeeze().shape: {X.squeeze().shape} - removes all single dimensions")
print(f"X.squeeze(1).shape: {X.squeeze(1).shape} - removes the single dimension at index 1")
print(f"X.squeeze(2).shape: {X.squeeze(2).shape} - does not change the tensor as dimension at index 2 is not of size 1")
print(f"X.squeeze((1, 3)).shape: {X.squeeze((1, 3)).shape} - removes the single dimensions at index 1 and 3")

print("\n===== Tensor Unsqueeze (Add singleton dimension at specific index) =====")
print(f"X.unsqueeze(0).shape: {X.unsqueeze(0).shape} - only works with single integer")

print("\n===== Tensor Permute =====")
print(f"Permute rearranges the dimensions of a tensor.")
print(f"For example, when processing the random image data we created before we might need to change")
R_T_IMAGE_PERMUTED = R_T_IMAGE.permute(2, 0, 1)
print(f"its dimensions from: {R_T_IMAGE.shape} to {R_T_IMAGE_PERMUTED.shape} via .permute(2, 0, 1).")
print(f"When changing a value of the permuted tensor, e.g. from {R_T_IMAGE_PERMUTED[0, 0, 0]}")
R_T_IMAGE_PERMUTED[0, 0, 0] = 111
print(f"to {R_T_IMAGE_PERMUTED[0, 0, 0]} also the original tensor is changed: {R_T_IMAGE[0, 0, 0]}")

print("\n===== Indexing / Selecting Data from Tensors =====")
X = torch.arange(1, 10).reshape(1, 3, 3)
print(f"X: \n{X}")
print(f"X[0]: \n{X[0]}")
print(f"X[0, 0]: \n{X[0, 0]}")
print(f"X[0, 0, 0]: \n{X[0, 0, 0]}")
print(f"X[0, 2, 2]: \n{X[0, 2, 2]}")
print(f"X[0, :, 1]: \n{X[0, :, 1]} - ':' gets all values of a dimension")
print(f"X[0, :, :]: \n{X[0, :, :]}")
print(f"X[:, 2, 2]: \n{X[:, 2, 2]} - almost like [0, 2, 2] but compare the result!")
print(f"X[:, :, 2]: \n{X[:, :, 2]}")

print("\n===== PyTorch & NumPy =====")
A1 = np.arange(1.0, 8.0)
T1 = torch.from_numpy(A1)
print(f"Using an NumPy array: {A1} to create a tensor: {T1} of datatype: {T1.dtype}")
T2 = torch.ones(7)
A2 = T2.numpy()
print(f"When going the other way around, the tensor: {T2} leads to an array: {A2} of datatype {A2.dtype}")
print(f"When we change elements of the array the tensor is not impacted, as they don't share memory.")
