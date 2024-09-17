#include<iostream>
#include<fstream>
#include<mpi.h>
#include<cmath>
// DEBUG:
#include<cstdio>
#include<string>
#include <unistd.h>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 4){
        printf("The function needs three arguments: matrix file name, vector file name and output file name\n");
        return -1;
    }

    // Init
    MPI_Init(NULL, NULL);
    int rank;                   // This processor's rank in comm_cart
    int p;                      // Number of processors in comm_cart, which should be the same as in MPI_COMM_WORLD in the case
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int q = sqrt(p);            // All p processors forms a q*q grid
    int coords[2] = {q, q};     // Coordinator of this processor in q*q grid, except initially used as dims array
    int periods[2] = {1, 1};    // TODO: Not sure
    MPI_Comm comm_cart;         // Communicator of cart topology
                                // TODO: redorder: true or false?
    MPI_Cart_create(MPI_COMM_WORLD, 2, coords, periods, 1, &comm_cart);
    MPI_Comm_rank(comm_cart, &rank);
    MPI_Cart_coords(comm_cart, rank, 2, coords);

    // DEBUG: https://www.open-mpi.org/faq/?category=debugging
    // if (rank == 0) {
    //     volatile int i = 0;
    //     char hostname[256];
    //     gethostname(hostname, sizeof(hostname));
    //     printf("PID %d on %s ready for attach\n", getpid(), hostname);
    //     fflush(stdout);
    //     while (0 == i)
    //         sleep(5);
    // }


    // Root read n from file
    int n = 0;
    ifstream inFile;
    if (rank == 0) {
        inFile.open(argv[1]);
        if (!inFile) {
            cout << "Unable to open matrix file";
            exit(1);
        }
        inFile >> n;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, comm_cart);


    // Allocate local matrix block
    int block_size[2]; // Size of local matrix block
    block_size[0] = coords[0] < n%q ? (n + q - 1) / q : n / q;
    block_size[1] = coords[1] < n%q ? (n + q - 1) / q : n / q;
    double *mat_data = new double[block_size[0]*block_size[1]];
    double **mat = new double*[block_size[0]]; // Local matrix block. Note that mat is a coutinuous block
    for (int i = 0; i < block_size[0]; i++) {
        mat[i] = &mat_data[i * block_size[1]];
    }


    // Root read matrix from file
    double *matrix_data = new double[n*n]; 
    double **matrix = new double*[n]; // Entire input matrix of size n*n. Note that matrix is a continuous block
    for (int i = 0; i < n; i++) {
        matrix[i] = &matrix_data[i*n];
    }
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inFile >> matrix[i][j];
            }
        }
        inFile.close();
    }


    // Allocate root vector block and local vector block
    double *vector = new double[n];                         // The entire vector of size n, read from the file
    double *vec = new double[n];                            // The local vector on each processor, distributed by root.
    int vec_len = coords[0] < n%q ? (n + q - 1) / q : n / q;// The length of local vector owned by processors on the first column in the beginning


    // Root read vector from file
    if (rank == 0) {
        inFile.open(argv[2]);
        if (!inFile) {
            cout << "Unable to open vector file";
            exit(1);
        }
        for (int i = 0; i < n; i++) {
            inFile >> vector[i];
        }
        inFile.close();
    }


    // Distribute Matrix to each processor
    //
    // Attemption 1: Use scatter, which is wrong.
    //
    /*
    MPI_Datatype block_types[q][q]; // MPI datatype for matrix block in each processor
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            int array_of_sizes[2] = {n, n};
            int array_of_subsizes[2] = {
                i < n%q ? (n + q - 1) / q : n / q,
                j < n%q ? (n + q - 1) / q : n / q
            };
            int array_of_starts[2] = {
                i < n%q ? i * (n+q-1)/q : (n%q) * (n+q-1)/q + (i - n%q) * (n/q),
                j < n%q ? j * (n+q-1)/q : (n%q) * (n+q-1)/q + (j - n%q) * (n/q)
            };
            // https://learn.microsoft.com/zh-cn/message-passing-interface/mpi-type-create-subarray-function
            MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &block_types[i][j]);
            MPI_Type_commit(&block_types[i][j]);
        }
    }
    MPI_Scatter(&(matrix[0][0]), 1, block_types[0], &(mat[0][0]), block_size[0]*block_size[1], MPI_DOUBLE, 0, comm_cart);
    */
    //
    // Attemption 2: https://stackoverflow.com/questions/29325513/scatter-matrix-blocks-of-different-sizes-using-mpi
    // Segmentation fault, and impossible to debug because:
    // Missing separate debuginfo for the main executable file
    // Try: yum --enablerepo='*debug*' install /usr/lib/debug/.build-id/16/8d7a1584e358549e8df8795cfdf9a7debc1f0a
    //
    /*
    int sendcounts[p];
    int senddispls[p];
    MPI_Datatype sendtypes[p];
    int recvcounts[p];
    int recvdispls[p];
    MPI_Datatype recvtypes[p];
    
    for (int i = 0; i < p; i++) {
        recvcounts[i] = 0;
        recvdispls[i] = 0;
        recvtypes[i] = MPI_DOUBLE;
    
        sendcounts[i] = 0;
        senddispls[i] = 0;
        sendtypes[i] = MPI_DOUBLE;
    }
    recvcounts[0] = block_size[0] * block_size[1];
    
    if (rank == 0) {
        MPI_Datatype block_types[q][q]; // MPI datatype for matrix block in each processor
        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                int array_of_sizes[2] = {n, n};
                int array_of_subsizes[2] = {
                    i < n%q ? (n + q - 1) / q : n / q,
                    j < n%q ? (n + q - 1) / q : n / q
                };
                int array_of_starts[2] = {0, 0};
                // sdispls[i][j] = (i < n%q ? i * (n+q-1)/q : (n%q) * (n+q-1)/q + (i - n%q) * (n/q)) * n 
                //                 + j < n%q ? j * (n+q-1)/q : (n%q) * (n+q-1)/q + (j - n%q) * (n/q);
                // sendcounts[i][j] = 1;
                // https://learn.microsoft.com/zh-cn/message-passing-interface/mpi-type-create-subarray-function
                MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &block_types[i][j]);
                MPI_Type_commit(&block_types[i][j]);
            }
        }
        // now figure out the displacement and type of each processor's data
        for (int r = 0; r < p; r++) {
            int i = r / q, j = r % q;
            sendcounts[r] = 1;
            senddispls[r] =  ((i < n%q ? i * (n+q-1)/q : (n%q) * (n+q-1)/q + (i - n%q) * (n/q)) * n 
                                + j < n%q ? j * (n+q-1)/q : (n%q) * (n+q-1)/q + (j - n%q) * (n/q))
                            * sizeof(double);
            sendtypes[r] = block_types[i][j];
        }
    }
    MPI_Alltoallw(matrix, sendcounts, senddispls, sendtypes, 
        &(mat[0][0]), recvcounts, recvdispls, recvtypes, comm_cart);
    */
    //
    // Attempt 3: Serial Send
    //
    MPI_Datatype block_types[q][q]; // MPI datatype for matrix block in each processor
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            int array_of_sizes[2] = {n, n};
            int array_of_subsizes[2] = {
                i < n%q ? (n + q - 1) / q : n / q,
                j < n%q ? (n + q - 1) / q : n / q
            };
            int array_of_starts[2] = {
                i < n%q ? i * (n+q-1)/q : (n%q) * (n+q-1)/q + (i - n%q) * (n/q),
                j < n%q ? j * (n+q-1)/q : (n%q) * (n+q-1)/q + (j - n%q) * (n/q)
            };
            // https://learn.microsoft.com/zh-cn/message-passing-interface/mpi-type-create-subarray-function
            MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &block_types[i][j]);
            MPI_Type_commit(&block_types[i][j]);
        }
    }
    if (rank == 0) {
        for (int i = 0; i < q; i++) {
            for (int j = 0; j < q; j++) {
                int c[2] = {i, j};  // coords [i, j]
                int r = 0;          // rank r corresponing to coords [i, j]
                MPI_Cart_rank(comm_cart, c, &r);
                if (r == 0) {
                    for (int ii = 0; ii < block_size[0]; ii++) {
                        for (int jj = 0; jj < block_size[1]; jj++) {
                            mat[ii][jj] = matrix[ii][jj];
                        }
                    }
                    continue; // TRAP: You can't let root recv message sent by itself. This is deadlock!
                } else {
                    MPI_Send(matrix[0], 1, block_types[i][j], r, 0, comm_cart);
                }
            }
        }
    } else {
        MPI_Status status;
        MPI_Recv(&(mat[0][0]), block_size[0] * block_size[1], MPI_DOUBLE, 0, 0, comm_cart, &status);
    }


    // Distribute vector to each processor on the first column
    // But first we need to create new row-wise and column-wise communicator
    MPI_Comm comm_col, comm_row;
    MPI_Comm_split(comm_cart, coords[1], coords[0], &comm_col);
    MPI_Comm_split(comm_cart, coords[0], coords[1], &comm_row);
    int scounts[q];
    int displs[q];
    int displ = 0;
    for (int i = 0; i < q; i++) {
        displs[i] = displ;
        scounts[i] = i < n%q ? (n + q - 1) / q : n / q;
        displ += scounts[i];
    }
    MPI_Scatterv(vector, scounts, displs, MPI_DOUBLE, vec, vec_len, MPI_DOUBLE, 0, comm_col);


    // DEBUG: Check if matrix is distributed correctly
    // string prefix = "matrix-", suffix = ".txt";
    // string filename = prefix + to_string(rank) + suffix;
    // FILE *fp = fopen(filename.c_str(), "w");
    // for (int i = 0; i < block_size[0]; i++) {
    //     for (int j = 0; j < block_size[1]; j++) {
    //         fprintf(fp, "%f ", mat[i][j]);
    //     }
    //     fprintf(fp, "\n");
    // }
    // fclose(fp);


    // ---------- Data preparation stage ends. Now parallel matrix vector multiplication begins ----------
    

    // Align vector along the principal diagonal of the matrix
    if (coords[1] == 0 && coords[0] != 0) {
        MPI_Send(vec, vec_len, MPI_DOUBLE, coords[0], 0, comm_row);     // TODO: Have to change tag for different iterations
    } else if (coords[1] == coords[0] && coords[0] != 0) {
        MPI_Status status;
        MPI_Recv(vec, vec_len, MPI_DOUBLE, 0, 0, comm_row, &status);    // TODO: Have to change tag for different iterations
    }
    // TODO: Why this doesn't work?
    // MPI_Status status;
    // if (coords[0] != 0) {
    //     MPI_Sendrecv_replace(vec, vec_len, MPI_DOUBLE, coords[0], 0, 0, 0, comm_row, &status); // TODO: Have to change tag for different iterations
    // }


    // Broadcast vector shard (of length vec_len) from diagonal to each column
    MPI_Bcast(&vec_len, 1, MPI_INT, coords[1], comm_col);
    MPI_Bcast(vec, vec_len, MPI_DOUBLE, coords[1], comm_col);


    // DEBUG: Check if vector is distributed properly
    // prefix = "vector-";
    // filename = prefix + to_string(coords[0]) + "-" + to_string(coords[1]) + suffix;
    // fp = fopen(filename.c_str(), "w");
    // for (int i = 0; i < vec_len; i++) {
    //     fprintf(fp, "%f ", vec[i]);
    // }
    // fprintf(fp, "\n");
    // fclose(fp);


    // Perform sub-matrix sub-vector multiplication. Results are stored in res
    double *res_temp = new double[block_size[0]]; // Vector of length vec_len containing sub-matrix sub-vector multiplication result on each processor.
    double *res = new double[block_size[0]];      // Vector of length vec_len on first column of processors containing reduced matrix vector multiplication result
    for (int i = 0; i < block_size[0]; i++) {
        res_temp[i] = 0;
        for (int j = 0; j < block_size[1]; j++) { // vec_len == block_size[1]
            res_temp[i] += mat[i][j] * vec[j];
        }
    }
    MPI_Reduce(res_temp, res, block_size[0], MPI_DOUBLE, MPI_SUM, 0, comm_row);

    
    // DEBUG: Calculate matrix vector multiplication in serial to check if answer is right
    if (rank == 0) {
        double *res_serial = new double[n];
        for (int i = 0; i < n; i++) {
            res_serial[i] = 0;
            for (int j = 0; j < n; j++) {
                res_serial[i] += matrix[i][j] * vector[j];
            }
        }

        FILE *fp = fopen("result-serial.txt", "w");
        for (int i = 0; i < n; i++) {
            fprintf(fp, "%f ", res_serial[i]);
        }
        fclose(fp);

        delete[] res_serial;
    }


    // DEBUG: Print matrix vector multiplication result
    if (coords[1] == 0) {
        string prefix = "result-", suffix = ".txt";
        string filename = prefix + to_string(coords[0]) + suffix;
        FILE *fp = fopen(filename.c_str(), "w");
        for (int i = 0; i < block_size[0]; i++) {
            fprintf(fp, "%f ", res[i]);
        }
        fclose(fp);
    }


    // Free matrix and local matrix block
    delete[] matrix;
    delete[] matrix_data;
    
    delete[] mat;
    delete[] mat_data;

    delete[] vector;
    delete[] vec;

    delete[] res_temp;
    delete[] res;

    MPI_Comm_free(&comm_col);
    MPI_Comm_free(&comm_row);
    MPI_Comm_free(&comm_cart);
    MPI_Finalize();
    return 0;
}