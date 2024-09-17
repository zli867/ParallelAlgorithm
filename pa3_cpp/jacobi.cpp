#include<iostream>
#include<fstream>
#include<mpi.h>
#include<cmath>
#include<iomanip>
// DEBUG:
#include <unistd.h>


using namespace std;

const double termination_criteria = 1e-9;
const int max_iteration = 1e6;


// @param mat: Local matrix block
// @param block_size: An array of length 2 indicating local matrix block size
// @param vec: Local vector
// @param vec_len: Local vector length
// @param coords: An array of length 2 indicating this processor's coordinate in the mesh
// @param tag: Provide different tags for different paris of send/recv
// @param comm_row: Row communicator
// @param comm_col: Column communicator
//
// @result res: Vector of length vec_len on first column of processors containing reduced matrix vector multiplication result
void matrix_vector_mul(
    double **mat, int *block_size, 
    double *vec, int vec_len, 
    int *coords, 
    int tag,
    MPI_Comm comm_row, MPI_Comm comm_col, MPI_Comm comm_cart,
    double *res
) {
    // //DEBUG:
    // string prefix = "", suffix = ".txt";
    // string filename = prefix + to_string(coords[0]) + '-' + to_string(coords[1]) + suffix;
    // FILE *fp = fopen(filename.c_str(), "w");

    // MPI_Barrier(comm_cart);
    // fprintf(fp, "vec 1:\n");
    // for (int i = 0; i < vec_len; i++) {
    //     fprintf(fp, "%f ", vec[i]);
    // }
    // fprintf(fp, "\n\n");

    // Align vector along the principal diagonal of the matrix
    MPI_Status status;
    if (coords[1] == 0 && coords[0] != 0) {
        MPI_Send(vec, vec_len, MPI_DOUBLE, coords[0], tag, comm_row);     // Have to change tag for different iterations
    } else if (coords[1] == coords[0] && coords[0] != 0) {
        MPI_Recv(vec, vec_len, MPI_DOUBLE, 0, tag, comm_row, &status);    // Have to change tag for different iterations
    }

    // MPI_Barrier(comm_cart);
    // fprintf(fp, "vec 2:\n");
    // for (int i = 0; i < vec_len; i++) {
    //     fprintf(fp, "%f ", vec[i]);
    // }
    // fprintf(fp, "\n\n");


    // Broadcast vector shard (of length vec_len) from diagonal to each column
    MPI_Bcast(&vec_len, 1, MPI_INT, coords[1], comm_col);
    MPI_Bcast(vec, vec_len, MPI_DOUBLE, coords[1], comm_col);


    // Perform sub-matrix sub-vector multiplication. Results are stored in res
    double *res_temp = new double[block_size[0]]; // Vector of length vec_len containing sub-matrix sub-vector multiplication result on each processor.
    for (int i = 0; i < block_size[0]; i++) {
        res_temp[i] = 0;
        for (int j = 0; j < block_size[1]; j++) { // vec_len == block_size[1]
            res_temp[i] += mat[i][j] * vec[j];
        }
    }
    // DEBUG:
    // MPI_Barrier(comm_cart);
    // fprintf(fp, "mat:\n");
    // for (int i = 0; i < block_size[0]; i++) {
    //     for (int j = 0; j < block_size[1]; j++) {
    //         fprintf(fp, "%f ", mat[i][j]);
    //     }
    //     fprintf(fp, "\n");
    // }
    // fprintf(fp, "\n");

    // fprintf(fp, "vec 3:\n");
    // for (int i = 0; i < vec_len; i++) {
    //     fprintf(fp, "%f ", vec[i]);
    // }
    // fprintf(fp, "\n\n");

    // fprintf(fp, "res_temp:\n");
    // for (int i = 0; i < block_size[0]; i++) {
    //     fprintf(fp, "%f ", res_temp[i]);
    // }
    // fclose(fp);

    // MPI_Barrier(comm_cart);
    MPI_Reduce(res_temp, res, block_size[0], MPI_DOUBLE, MPI_SUM, 0, comm_row);
    delete[] res_temp;
}

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
    MPI_Comm comm_col, comm_row; // Create new row-wise and column-wise communicator
    MPI_Comm_split(comm_cart, coords[1], coords[0], &comm_col);
    MPI_Comm_split(comm_cart, coords[0], coords[1], &comm_row);


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


    // Generate counts and displs array that can be used by Scatterv and Gatherv
    int srcounts[q];
    int displs[q];
    int displ = 0;
    for (int i = 0; i < q; i++) {
        displs[i] = displ;
        srcounts[i] = i < n%q ? (n + q - 1) / q : n / q;
        displ += srcounts[i];
    }


    // Allocate local memory block for matrix A and R
    int block_size[2]; // Size of local matrix block
    block_size[0] = coords[0] < n%q ? (n + q - 1) / q : n / q;
    block_size[1] = coords[1] < n%q ? (n + q - 1) / q : n / q;
    double *mat_A_data = new double[block_size[0]*block_size[1]];
    double **mat_A = new double*[block_size[0]]; // Local matrix A block. Note that mat is a coutinuous block
    for (int i = 0; i < block_size[0]; i++) {
        mat_A[i] = &mat_A_data[i * block_size[1]];
    }
    double *mat_R_data = new double[block_size[0]*block_size[1]];
    double **mat_R = new double*[block_size[0]]; // Local matrix R block. Note that mat is a coutinuous block
    for (int i = 0; i < block_size[0]; i++) {
        mat_R[i] = &mat_R_data[i * block_size[1]];
    }


    // Root read matrix A from file
    double *matrix_A_data = new double[n*n]; 
    double **matrix_A = new double*[n]; // Entire input matrix A of size n*n. Note that matrix is a continuous block
    for (int i = 0; i < n; i++) {
        matrix_A[i] = &matrix_A_data[i*n];
    }
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inFile >> matrix_A[i][j];
            }
        }
        inFile.close();
    }


    // Allocate root and local vector b and x block
    int vec_len = coords[0] < n%q ? (n + q - 1) / q : n / q;    // The length of local vector b or x owned by processors on the first column in the beginning
    double *vector_b = new double[n];                           // The entire vector b of size n, read from the file
    double *vec_b = new double[vec_len];                        // The local vector b on each processor, distributed by root.
    double *vector_x = new double[n];                           // The entire vector x of size n, read from the file
    double *vec_x = new double[n];                              // The local vector x on each processor, distributed by root. Note that because vec_x will be broadcasted later, so allocate enough size n


    // Allocate root and local vector D block
    // Note D is actually a diagonal matrix, but stored as a vector
    double *vector_D = new double[n];   // Root D vector(matrix) of size n
    double *vec_D = new double[vec_len];// Local D vector(matrix) of size vec_len and vec_D is only used by first column of matrix


    // Calculate vector_D and distribute to vec_D to processors on first column
    for (int i = 0; i < n; i++) {
        vector_D[i] = matrix_A[i][i];
    }
    MPI_Scatterv(vector_D, srcounts, displs, MPI_DOUBLE, vec_D, vec_len, MPI_DOUBLE, 0, comm_col);


    // Root read vector b from file
    if (rank == 0) {
        inFile.open(argv[2]);
        if (!inFile) {
            cout << "Unable to open vector file";
            exit(1);
        }
        for (int i = 0; i < n; i++) {
            inFile >> vector_b[i];
        }
        inFile.close();
    }


    // Each processor initialize local vector x
    for (int i = 0; i < vec_len; i++) {
        vec_x[i] = 0;
    }


    // Distribute Matrix A to each processor
    // Attempt 3: Serial Send
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
                            mat_A[ii][jj] = matrix_A[ii][jj];
                        }
                    }
                    continue; // TRAP: You can't let root recv message sent by itself. This is deadlock!
                } else {
                    MPI_Send(matrix_A[0], 1, block_types[i][j], r, 0, comm_cart);
                }
            }
        }
    } else {
        MPI_Status status;
        MPI_Recv(&(mat_A[0][0]), block_size[0] * block_size[1], MPI_DOUBLE, 0, 0, comm_cart, &status);
    }


    // Generate local matrix R and D based on local matrix A
    for (int i = 0; i < block_size[0]; i++) {
        for (int j = 0; j < block_size[1]; j++) {
            int global_i = (coords[0] < n%q ? coords[0] * (n+q-1)/q : (n%q) * (n+q-1)/q + (coords[0] - n%q) * (n/q)) + i;
            int global_j = (coords[1] < n%q ? coords[1] * (n+q-1)/q : (n%q) * (n+q-1)/q + (coords[1] - n%q) * (n/q)) + j;
            if (global_i == global_j) {
                mat_R[i][j] = 0;
            } else {
                mat_R[i][j] = mat_A[i][j];
            }
        }
    }

    // DEBUG:
    // {
    //     string prefix = "D-", suffix = ".txt";
    //     string filename = prefix + to_string(coords[0]) + '-' + to_string(coords[1]) + suffix;
    //     FILE *fp = fopen(filename.c_str(), "w");

    //     for (int i = 0; i < vec_len; i++) {
    //         fprintf(fp, "%f ", vec_D[i]);
    //     }
    //     fprintf(fp, "\n");
    //     fclose(fp);
    // }
    // MPI_Barrier(comm_cart);


    // Distribute vector b to each processor on the first column
    MPI_Scatterv(vector_b, srcounts, displs, MPI_DOUBLE, vec_b, vec_len, MPI_DOUBLE, 0, comm_col);

    // DEBUG: https://www.open-mpi.org/faq/?category=debugging
    // if (coords[0] == 0 && coords[1] == 0) {
    //     volatile int i = 0;
    //     char hostname[256];
    //     gethostname(hostname, sizeof(hostname));
    //     printf("Processor %d-%d PID %d on %s ready for attach,\n", coords[0], coords[1], getpid(), hostname);
    //     fflush(stdout);
    //     while (0 == i)
    //         sleep(5);
    // }
    // MPI_Barrier(comm_cart); // Stall other processors


    // Jcaobi
    bool terminated = false;
    int iter_count = 0;
    do {
        // x = Rx
        matrix_vector_mul(mat_R, block_size, vec_x, vec_len, coords, iter_count * 2, comm_row, comm_col, comm_cart, vec_x);
        // DEBUG:
        MPI_Barrier(comm_cart);
        // x = D^{-1}(b - x)
        if (coords[1] == 0) {
            for (int i = 0; i < vec_len; i++) {
                vec_x[i] = (vec_b[i] - vec_x[i]) / vec_D[i];
            }
        }
        double *Ax = new double[vec_len];
        MPI_Barrier(comm_cart);
        matrix_vector_mul(mat_A, block_size, vec_x, vec_len, coords, iter_count * 2 + 1, comm_row, comm_col, comm_cart, Ax);
        MPI_Barrier(comm_cart);
        double distance = 0;
        double distance_total = 0;
        if (coords[1] == 0) {
            for (int i = 0; i < vec_len; i++) {
                distance += (Ax[i] - vec_b[i]) * (Ax[i] - vec_b[i]);
            }
            MPI_Reduce(&distance, &distance_total, 1, MPI_DOUBLE, MPI_SUM, 0, comm_col);
        }
        MPI_Barrier(comm_cart);
        delete[] Ax;
        distance_total = sqrt(distance_total);
        // DEBUG:
        if (rank == 0) {
            printf("iter %d, distance %f\n", iter_count, distance_total);
        }
        if (rank == 0) {
            terminated = distance_total < termination_criteria * termination_criteria;
        }
        MPI_Bcast(&terminated, 1, MPI_C_BOOL, 0, comm_cart);
        iter_count++;
    } while(!terminated && iter_count < max_iteration);


    // Gather x and output
    if (coords[1] == 0) {
        MPI_Gatherv(vec_x, vec_len, MPI_DOUBLE, vector_x, srcounts, displs, MPI_DOUBLE, 0, comm_col);
        if (coords[0] == 0) {
            ofstream outFile(argv[3]);
            if (!outFile) {
                cout << "Unable to open output file";
                exit(1);
            }
            for (int i = 0; i < n; i++) {
                outFile << setprecision(16) << vector_x[i] << " ";
            }
            outFile.close();
        }
    }


    // Free matrix and local matrix block
    delete[] matrix_A;
    delete[] matrix_A_data;
    
    delete[] mat_A;
    delete[] mat_A_data;
    delete[] mat_R;
    delete[] mat_R_data;

    delete[] vector_b;
    delete[] vector_x;
    delete[] vector_D;
    delete[] vec_b;
    delete[] vec_x;
    delete[] vec_D;

    MPI_Comm_free(&comm_col);
    MPI_Comm_free(&comm_row);
    MPI_Comm_free(&comm_cart);
    MPI_Finalize();
    return 0;
}