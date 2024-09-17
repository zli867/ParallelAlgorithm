## Machine environment: 
Code is executed on PACE COC-ICE cluster. We used the default environment on ICE cluster, which includes: gcc version 8.3.0, mvapich2/2.3.2.

## How the program works: 
* Compile the code: Go to the directory where Makefile is. Use the command: make to generate the executed object. Use the command: mpirun -np $i ./int_calc $n to run the program. $i: the number of processors. $n: the number of splits for calculating the integral.
* Clean: Use the command: make clean to delete all the objects or log files.