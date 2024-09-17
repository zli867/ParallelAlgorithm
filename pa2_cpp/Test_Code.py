import random
import os
import argparse


def generate_input(size, min_value, max_value):
    random_int_array = []
    for i in range(0, size):
        random_int_array.append(random.randint(min_value, max_value))
    return random_int_array

parser = argparse.ArgumentParser()
parser.add_argument('-np', help="Number of processors", default=8)
parser.add_argument('-n', help="Input size", default=1000000)
args = parser.parse_args()
size = int(args.n)

filename = f"pa2_input_{size}.txt"
min_value = -2147483648
max_value = 2147483647
try:
    f = open(filename, 'r')
    f.close()
except IOError:
    with open(filename, 'w') as f:
        f.write(str(size))
        f.write('\n')
        int_array = generate_input(size, min_value, max_value)
        for value in int_array:
            f.write(str(value))
            f.write(' ')

# Run code
os.system("make")
print(f"{args.np} processors, n = {args.n}")
os.system(f"mpirun -np {args.np} ./pqsort pa2_input_{size}.txt output.txt")

# test results
res_file = "output.txt"
with open(res_file) as f:
    lines = f.readlines()

res_sort_array = lines[0].split(" ")
res_sort_array.pop()
res_sort_int_array = []
for current_str in res_sort_array:
    res_sort_int_array.append(int(current_str))

prev = min_value
for num in res_sort_int_array:
    if prev > num:
        print("Unsorted!")
        exit()
    prev = num
print("Sorted!")