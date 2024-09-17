import numpy as np
import random

# Define the size of the matrix
n = 16
A = np.random.rand(n, n)
A = 10 * A
x = np.random.rand(n, 1)
x = 10 * x
for i in range(n):
    A[i, i] = np.sum(np.abs(A[i])) + random.uniform(1, 10)
b = A @ x

mat_filename = f"input_mat_{n}.txt"
vec_filename = f"input_vec_{n}.txt"
out_filename = f"expected_output_{n}.txt"
# Write A and b
with open(mat_filename, 'w') as f:
    f.write(str(n))
    f.write('\n')
    for i in range(0, n):
        current_str = ""
        for j in range(0, n):
            current_str = current_str + str(A[i, j]) + " "
        f.write(current_str)
        if i < n - 1:
            f.write('\n')

with open(vec_filename, 'w') as f:
    current_str = ""
    for i in range(0, n):
        current_str = current_str + str(b[i, 0]) + " "
    f.write(current_str)

with open(out_filename, 'w') as f:
    current_str = ""
    for i in range(0, n):
        current_str = current_str + str(x[i, 0]) + " "
    f.write(current_str)