#include <mpi.h>
#include <cstdio>

double integral(int start_step, int end_step, int split_num){
    double integral_res = 0;
    double step_size = 1.0/split_num;
    for (int i = start_step; i <= end_step; i++){
        integral_res += 4/(1 + step_size * (i - 0.5) * step_size * (i - 0.5));
    }
    return integral_res / split_num;
}

int main(int argc, char** argv) {
    if (argc != 2){
        printf("The function needs argument: number of splits.\n");
        return -1;
    }
    int k = 0, n = 0;
    while (argv[1][k] != '\0'){
        if (argv[1][k] < '0' || argv[1][k] > '9'){
            printf("Invalid argument: number of splits.\n");
            return -1;
        }else{
            n = 10 * n + (argv[1][k] - '0');
        }
        k++;
    }

    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();

    int processor_num;
    MPI_Comm_size(MPI_COMM_WORLD, &processor_num);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Calculate start_step and end_step for each processor
    int start_step, end_step;
    int mode_res = n % processor_num;
    if (rank < mode_res){
        start_step = rank * (n / processor_num + 1) + 1;
        end_step = (rank + 1) * (n / processor_num + 1);
    }else{
        start_step = rank * (n / processor_num) + mode_res + 1;
        end_step = (rank + 1) * (n / processor_num) + mode_res;
    }

    double local_sum = integral(start_step, end_step, n);
    // Combine sum for each local processor
    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    if (rank == 0){
        printf("%.12lf, %lf\n", global_sum, end_time - start_time);
    }
    MPI_Finalize();
    return 0;
}