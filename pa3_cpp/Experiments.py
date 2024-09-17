import os
import numpy as np


def inputGenerator(data_size):
    np.random.seed(100)
    n = data_size
    A = np.random.rand(n, n) * data_size
    x = np.random.rand(n, 1) * data_size
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + 1

    b = A @ x

    mat_filename = "input_mat.txt"
    vec_filename = "input_vec.txt"
    expect_res_filename = "expect_res.txt"
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

    with open(expect_res_filename, 'w') as f:
        current_str = ""
        for i in range(0, len(x)):
            current_str = current_str + str(x[i, 0]) + " "
        f.write(current_str)


os.system("make clean")
# Compile Code
os.system("make")

# exp 1: fixed data size
processor_nums = [1, 4, 9, 16]
data_size_base = 1152
inputGenerator(data_size_base)
os.system('echo "processor_num, runtime" >> log.txt')
for processor_num in processor_nums:
    os.system('echo -n "%d,">> log.txt' % processor_num)
    os.system("mpirun -np %d ./pjacobi input_mat.txt input_vec.txt output.txt >> log.txt" % processor_num)

# exp 2: fixed processor
processor_num = 16
data_sizes = [16, 32, 64, 128, 256, 512, 1024]
os.system('echo "data_size, processor_num, runtime" >> log_size.txt')
for data_size in data_sizes:
    inputGenerator(data_size)
    # Parallel Running
    os.system('echo -n "%d,%d,">> log_size.txt' % (data_size, processor_num))
    os.system("mpirun -np %d ./pjacobi input_mat.txt input_vec.txt output.txt >> log_size.txt" % processor_num)
    # Serial Running
    os.system('echo -n "%d,%d,">> log_size.txt' % (data_size, 1))
    os.system("mpirun -np 1 ./pjacobi input_mat.txt input_vec.txt output.txt >> log_size.txt")

