#include <iostream>
#include <mpi.h>
#include <fstream>
#include <cmath>
#include <iomanip>
#define NDIM 2
#define MAX_ITERNUM 1000000

using namespace std;

void equally_vector_splitter(int send_data_num[], int displs[], int num_count, int processor_num){
    int sum = 0;
    for (int i = 0; i < processor_num; i++){
        if (i < (num_count % processor_num)){
            send_data_num[i] = (num_count/processor_num) + 1;
        }else{
            send_data_num[i] = num_count/processor_num;
        }
        displs[i] = sum;
        sum += send_data_num[i];
    }
}

void matrix_transpose(double flatten_matrix[], int row, int col){
    double tmp_matrix[row][col];
    int idx = 0;
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++)
            tmp_matrix[i][j] = flatten_matrix[idx ++];
    }
    idx = 0;
    for (int i = 0; i < col; i ++){
        for (int j = 0; j < row; j++)
            flatten_matrix[idx ++] = tmp_matrix[j][i];
    }
}

void equally_row_splitter(int send_data_num[], int displs[], int processor_num, int matrix_size){
    int sum = 0;
    for (int i = 0; i < processor_num; i++){
        if (i < (matrix_size % processor_num)){
            send_data_num[i] = ((matrix_size/processor_num) + 1) * matrix_size;
        }else{
            send_data_num[i] = (matrix_size/processor_num) * matrix_size;
        }
        displs[i] = sum;
        sum += send_data_num[i];
    }
}

void equally_col_splitter(int send_data_num[], int displs[], int processor_num, int matrix_size, int row_size){
    int sum = 0;
    for (int i = 0; i < processor_num; i++){
        if (i < (matrix_size % processor_num)){
            send_data_num[i] = ((matrix_size/processor_num) + 1) * row_size;
        }else{
            send_data_num[i] = (matrix_size/processor_num) * row_size;
        }
        displs[i] = sum;
        sum += send_data_num[i];
    }
}

int main(int argc, char** argv)  {

    if (argc != 4){
        printf("The function needs two arguments: input file name and output file name\n");
        return -1;
    }

    int rank, size;
    int dims[NDIM] = {0, 0};
    int period[NDIM] = {1, 1};
    int coords[NDIM];
    MPI_Comm comm;
    // Cartesian Structure
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Dims_create(size, NDIM, dims);
    MPI_Cart_create(MPI_COMM_WORLD, NDIM, dims, period, 1, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, NDIM, coords);
    int root;
    int root_coord[] = {0, 0};
    MPI_Cart_rank(comm, root_coord, &root);

    // sub-communicators
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(comm, coords[0], coords[1], &row_comm);
    MPI_Comm_split(comm, coords[1], coords[0], &col_comm);

    // Read dimension
    int n;
    if (rank == root){
        ifstream inFile;
        inFile.open(argv[1]);
        if (!inFile) {
            cout << "Unable to open matrix file";
            exit(1);
        }
        inFile >> n;
        inFile.clear();
        inFile.close();
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    double A[n * n], b[n], final_x[n], d[n];
    if (rank == root){
        // Read Matrix A
        ifstream inFile;
        inFile.open(argv[1]);
        inFile >> n;
        int idx = 0;
        for (int i = 0; i < n * n; i ++) inFile >> A[i];
        inFile.clear();
        inFile.close();
        // Read vector
        inFile.open(argv[2]);
        if (!inFile) {
            cout << "Unable to open vector file";
            exit(1);
        }
        for (int row = 0; row < n; row ++) inFile >> b[row];
        for (int i = 0; i < n; i ++) d[i] = A[i * (n + 1)];
    }

    int q = (int) sqrt(size);
    double local_A[n/q + 1][n/q + 1], local_R[n/q + 1][n/q + 1], local_b[n/q + 1], local_d[n/q + 1], local_x[n/q + 1], local_Ax[n/q + 1], local_Rx[n/q + 1];
    int local_vec_size = coords[0] < n % q ? n / q  + 1: n / q;

    // block distribution vector and initialize x0, d
    if (coords[1] == 0){
        int send_data_num[q], displs[q];
        equally_vector_splitter(send_data_num, displs, n, q);
        MPI_Scatterv(b, send_data_num, displs, MPI_DOUBLE, local_b, local_vec_size, MPI_DOUBLE, 0, col_comm);
        MPI_Scatterv(d, send_data_num, displs, MPI_DOUBLE, local_d, local_vec_size, MPI_DOUBLE, 0, col_comm);
        for (int i = 0; i < local_vec_size; i ++) local_x[i] = 0;
    }


    // block distribution matrix and initialize R
    int start_row_idx, start_col_idx, row_size, col_size;
    if (coords[0] < n % q){
        start_row_idx = coords[0] * (n / q + 1);
        row_size = n / q  + 1;
    }else{
        start_row_idx = coords[0] * (n / q) + n % q;
        row_size = n / q;
    }
    if (coords[1] < n % q){
        start_col_idx = coords[1] * (n / q + 1);
        col_size = n / q  + 1;
    }else{
        start_col_idx = coords[1] * (n / q) + n % q;
        col_size = n / q;
    }

    double buf_A[n * n];
    int send_data_num[q], displs[q];
    // block distribution vector and initialize x0, d
    if (coords[1] == 0){
        equally_row_splitter(send_data_num, displs, q, n);
        MPI_Scatterv(A, send_data_num, displs, MPI_DOUBLE, buf_A, row_size * n, MPI_DOUBLE, 0, col_comm);
        matrix_transpose(buf_A, row_size, n);
        equally_col_splitter(send_data_num, displs, q, n, row_size);
    }
    MPI_Scatterv(buf_A, send_data_num, displs, MPI_DOUBLE, A, row_size * col_size, MPI_DOUBLE, 0, row_comm);

    // Generate local_A, Generate local_R
    int idx = 0;
    for (int col = 0; col < col_size; col ++){
        for (int row = 0; row < row_size; row ++){
            local_A[row][col] = A[idx ++];
            if (start_row_idx + row == start_col_idx + col){
                local_R[row][col] = 0;
            }else{
                local_R[row][col] = local_A[row][col];
            }
        }
    }

    double start_time = MPI_Wtime();
    // 0 for not converged, 1 for converged
    int converged = 0;
    // Parallel Jacobi Method
    for (int iter_num = 0; iter_num < MAX_ITERNUM; iter_num++){
        // vector matrix multiplication (Ax, Rx)
        // send local vector to diagonal (DEAD LOCK!!!!)
        if (coords[1] == 0 && coords[0] != 0){
            MPI_Send(local_x, local_vec_size, MPI_DOUBLE, coords[0], coords[0] + iter_num, row_comm);
        }
        if (coords[1] == coords[0] && coords[0] != 0){
            MPI_Recv(local_x, local_vec_size, MPI_DOUBLE, 0, coords[0] + iter_num, row_comm, MPI_STATUS_IGNORE);
        }

        // Broadcast to each row via col_comm
        MPI_Bcast(&local_vec_size, 1, MPI_INT, coords[1], col_comm);
        MPI_Bcast(local_x, local_vec_size, MPI_DOUBLE, coords[1], col_comm);

        // Multiplication Locally local_x and local_A, local_x and local_R
        double local_Ax_tmp[n/q + 1], local_Rx_tmp[n/q + 1];
        for (int i = 0; i < row_size; i++){
            local_Ax_tmp[i] = 0;
            local_Rx_tmp[i] = 0;
            for (int j = 0; j < col_size; j++){
                local_Ax_tmp[i] += local_A[i][j] * local_x[j];
                local_Rx_tmp[i] += local_R[i][j] * local_x[j];
            }
        }

        // Reduce
        MPI_Reduce(local_Ax_tmp, local_Ax, row_size, MPI_DOUBLE, MPI_SUM, 0, row_comm);
        MPI_Reduce(local_Rx_tmp, local_Rx, row_size, MPI_DOUBLE, MPI_SUM, 0, row_comm);

        // Calculate ||Ax - b|| locally
        if (coords[1] == 0){
            double local_sum = 0, global_sum;
            int col_comm_rank;
            for (int i = 0; i < row_size; i++) local_sum += (local_Ax[i] - local_b[i]) * (local_Ax[i] - local_b[i]);
            MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, col_comm);
            MPI_Comm_rank(col_comm, &col_comm_rank);
            if (col_comm_rank == 0 && sqrt(global_sum) < 1e-9){
                converged = 1;
            }
        }
        // If converged break
        MPI_Bcast(&converged, 1, MPI_INT, root, comm);
        if (converged == 1){
            for (int i = 0; i < row_size; i++) local_x[i] = (local_b[i] - local_Rx[i]) * (1 / local_d[i]);
            if (coords[1] == 0) local_vec_size = row_size;
            break;
        }
        // Update x
        if (coords[1] == 0){
            for (int i = 0; i < row_size; i++) local_x[i] = (local_b[i] - local_Rx[i]) * (1 / local_d[i]);
            // Restore local vector size
            local_vec_size = row_size;
        }
    }
    double end_time = MPI_Wtime();

    if (rank == root) cout << end_time - start_time << endl;
    // Gather all local_x and write to file (local_x -> final_x)
    if (coords[1] == 0){
        // receiv counts and displs
        int recvcounts[q], displs[q];
        displs[0] = 0;
        for (int i = 0; i < q; i++){
            int recvcount = i < n % q ? n / q + 1 : n / q;
            recvcounts[i] = recvcount;
            if (i < q - 1) displs[i + 1] = displs[i] + recvcount;
        }
        MPI_Gatherv(local_x, row_size, MPI_DOUBLE, final_x, recvcounts, displs, MPI_DOUBLE, 0, col_comm);
        int col_comm_rank;
        MPI_Comm_rank(col_comm, &col_comm_rank);
        // Write to target file
        if (col_comm_rank == 0){
            ofstream file(argv[3]);
            if (file.is_open()) {
                for (int i = 0; i < n; i++) file << setprecision(16) << final_x[i] << " ";
                file.close();
            }
            else {
                cout << "Error opening file!" << endl;
            }
        }
    }

    MPI_Finalize();
}