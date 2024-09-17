#include <mpi.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <unistd.h>
using namespace std;

// @param array: Integer array on this processor
// @param size: Number of integers on this processor
// @param &split_idx: Return by reference: array[0, split_idx) <= pivot,
//                                          array[split_idx, size) > pivot
void partition_array(int array[], int size, int &split_idx, int pivot){
    int l = -1;
    int r = size;
    while (l < r){
        do l++; while (l < size && array[l] <= pivot);
        do r--; while (r >= 0 && array[r] > pivot);
        if (l < r) swap(array[l], array[r]);
    }
    split_idx = l;
}

// @param num_count: Number of intergers in current communicator
// @param processor_num: Number of processors in current communicator
void equally_splitter(int receive_data_num[], int displs[], int num_count, int processor_num){
    int sum = 0;
    for (int i = 0; i < processor_num; i++){
        if (i < (num_count % processor_num)){
            receive_data_num[i] = (num_count/processor_num) + 1;
        }else{
            receive_data_num[i] = num_count/processor_num;
        }
        displs[i] = sum;
        sum += receive_data_num[i];
    }
}

// @param num_count: Number of integers in current communicator
// @param processor_num: Number of processors in current communicator
// @param &rank_num: Return by reference: the rank in this communicator of the processor which has the pivot
// @param &local_index: Return by reference: the index of this pivot on its processor
void get_pivot_loc(int num_count, int processor_num, int k, int &rank_num, int &local_index){
    if (k < (num_count % processor_num) * (num_count/ processor_num + 1)){
        rank_num = k/((num_count/processor_num) + 1);
        local_index = k % ((num_count/processor_num) + 1);
    }else{
        k = k - (num_count % processor_num) * (num_count/ processor_num + 1);
        rank_num = k/(num_count/processor_num) + (num_count % processor_num);
        local_index = k % (num_count/processor_num);
    }
}


int array_sum(int array[], int size){
    int sum = 0;
    for (int i = 0; i < size; i++) sum += array[i];
    return sum;
}

void displs_calculate(int size_array[], int displs[], int size, int start_idx){
    int sum = start_idx;
    for (int i = 0; i < size; i++){
        displs[i] = sum;
        sum += size_array[i];
    }
}

void print_array(int array[], int size){
    for (int i = 0; i < size; i++){
        cout << array[i] << " ";
    }
    cout << endl;
}

// @param part_size: An array of number of intergers that are in the lower or highr part on each processor
// @param send_num:
// @param size: Number of processors in current communicator
// @param allocate_processor_number: Allocated number of processors on lower or higher sub-problem
// @param rev: 
void calculate_send_num(int part_size[], int send_num[], int size, int allocate_processor_num, int local_rank, bool rev){
    int prev_sum = 0; // Number of integers in all processors that have lower-rank than this one
    for (int i = 0; i < local_rank; i++){
        prev_sum += part_size[i];
    }
    int current_sum = prev_sum + part_size[local_rank]; // Number of integers in processors from rank 0 to this one in this communicator
    int prev_target[size], current_target[size];
    // prev target
    for (int i = 0; i < size; i++){
        if (i < allocate_processor_num){
            if (i < prev_sum % allocate_processor_num){
                prev_target[i] = (prev_sum/allocate_processor_num) + 1;
            }else{
                prev_target[i] = (prev_sum/allocate_processor_num);
            }
        }else{
            prev_target[i] = 0;
        }
    }
    // current target
    for (int i = 0; i < size; i++){
        if (i < allocate_processor_num){
            if (i < current_sum % allocate_processor_num){
                current_target[i] = (current_sum/allocate_processor_num) + 1;
            }else{
                current_target[i] = (current_sum/allocate_processor_num);
            }
        }else{
            current_target[i] = 0;
        }
    }
    // send num
    for (int i = 0; i < size; i++) send_num[i] = current_target[i] - prev_target[i];
    if (rev){
        int l = 0, r = size - 1;
        while (l < r)swap(send_num[l ++], send_num[r --]);
    }
}

int main(int argc, char** argv) {
    if (argc != 3){
        printf("The function needs two arguments: input file name and output file name\n");
        return -1;
    }

    // DEBUG: https://www.open-mpi.org/faq/?category=debugging
    //
    // volatile int i = 0;
    // char hostname[256];
    // gethostname(hostname, sizeof(hostname));
    // printf("PID %d on %s ready for attach\n", getpid(), hostname);
    // fflush(stdout);
    // while (0 == i)
    //     sleep(5);

    MPI_Init(NULL, NULL);
    int processor_num;      // Number of processors in current communicator
    int global_rank;        // This processor's rank in MPI_COMM_WORLD
    int local_rank;         // This processor's rank in current communicator
    int local_size;         // Number of integers on this processor
    int comm_total_size;    // Number of integers in current communicator
    int total_processor_num;// Number of processors in MPI_COMM_WORLD
    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processor_num);

    double start_io_time = MPI_Wtime();
    // Read input files
    int num_count = 0; // Number of integers in original input file
    ifstream inFile;
    if (global_rank == 0) {
        inFile.open(argv[1]);
        if (!inFile) {
            cout << "Unable to open file";
            exit(1);
        }
        inFile >> num_count;
        
    }
    MPI_Bcast(&num_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int *local_num_arr = new int[num_count];
    int *recv_buf = new int[num_count];

    if (global_rank == 0) {
        int idx = 0;
        while (inFile >> recv_buf[idx ++]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    comm_total_size = num_count;

    MPI_Comm current_comm = MPI_COMM_WORLD;

    double start_distribution_time = MPI_Wtime();

    int receive_data_num[total_processor_num];
    int displs[total_processor_num];
    equally_splitter(receive_data_num, displs, comm_total_size, total_processor_num);
    MPI_Scatterv(recv_buf, receive_data_num, displs, MPI_INT, local_num_arr, receive_data_num[global_rank],MPI_INT, 0, MPI_COMM_WORLD);
    local_size = receive_data_num[global_rank];
    // Broadcast a seed
    int seed_num = time(NULL);
    MPI_Bcast(&seed_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    srand(seed_num);

    double start_time = MPI_Wtime();
    // recursion part
    while (true) {
        MPI_Comm_size(current_comm, &processor_num);
        MPI_Comm_rank(current_comm, &local_rank);

        if (processor_num <= 1){
            sort(local_num_arr, local_num_arr + comm_total_size);
            break;
        }

        // calculate current pivot
        int k = rand() % comm_total_size;
        int pivot, pivot_rank, pivot_local_index;
        get_pivot_loc(comm_total_size, processor_num, k, pivot_rank, pivot_local_index);
        if (local_rank == pivot_rank) {
            pivot = local_num_arr[pivot_local_index];
        }
        MPI_Bcast(&pivot, 1, MPI_INT, pivot_rank, current_comm);
        int split_idx; // lower_size, local_num_arr[0, split_idx) <= pivot, array[split_idx, local_size) > pivot
        partition_array(local_num_arr, local_size, split_idx, pivot);

        int upper_size = local_size - split_idx;
        int lower_part_size[processor_num]; // An array of number of integers <= pivot on each processor in current communicator 
        int upper_part_size[processor_num]; // An array of number of integers > pivot on each processor in current communicator 
        MPI_Allgather(&split_idx, 1, MPI_INT, lower_part_size, 1, MPI_INT, current_comm);
        MPI_Allgather(&upper_size, 1, MPI_INT, upper_part_size, 1, MPI_INT, current_comm);
        int m_lower = array_sum(lower_part_size, processor_num);
        int m_upper = array_sum(upper_part_size, processor_num);

        int lower_processor_num = (m_lower / (m_lower + m_upper)) * processor_num > 1 ? (m_lower / (m_lower + m_upper)) * processor_num: 1;
        lower_processor_num = lower_processor_num == processor_num ? processor_num - 1: lower_processor_num;
        int upper_processor_num = processor_num - lower_processor_num;

        // Alltoallv lower part, calculate send num array (receive num array can be calculated by Alltoall)
        // Lower part
        int sdispls[processor_num], rdispls[processor_num], send_num[processor_num], receiv_num[processor_num];
        calculate_send_num(lower_part_size, send_num, processor_num, lower_processor_num, local_rank, false);
        MPI_Alltoall(send_num,1,MPI_INT,receiv_num,1,MPI_INT,current_comm);
        displs_calculate(send_num, sdispls, processor_num, 0);
        displs_calculate(receiv_num, rdispls, processor_num, 0);
        MPI_Alltoallv(local_num_arr,send_num, sdispls,MPI_INT,recv_buf,receiv_num, rdispls,MPI_INT,current_comm);
        local_size = local_size - array_sum(send_num, processor_num) + array_sum(receiv_num, processor_num);

        // Upper part
        calculate_send_num(upper_part_size, send_num, processor_num, upper_processor_num, local_rank, true);
        MPI_Alltoall(send_num,1,MPI_INT,receiv_num,1,MPI_INT,current_comm);
        displs_calculate(send_num, sdispls, processor_num, split_idx);
        displs_calculate(receiv_num, rdispls, processor_num, 0);
        MPI_Alltoallv(local_num_arr,send_num, sdispls,MPI_INT,recv_buf,receiv_num, rdispls,MPI_INT,current_comm);
        local_size = local_size - array_sum(send_num, processor_num) + array_sum(receiv_num, processor_num);

        // copy recv_buf
        memcpy(local_num_arr, recv_buf, local_size * sizeof(int));
//        print_array(local_num_arr, local_size);

        int color = local_rank < lower_processor_num? 0: 1;
        MPI_Comm_split(current_comm, color, global_rank, &current_comm);
        if (color == 0){
            comm_total_size = m_lower;
        }else{
            comm_total_size = m_upper;
        }
    }

    double end_time = MPI_Wtime();
    // gather the (receiv_buf, receiv_buf + comm_total_size) from every processor and write to output.txt
    int local_array_size[total_processor_num], res_displs[total_processor_num];
    MPI_Gather(&comm_total_size, 1, MPI_INT, local_array_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    displs_calculate(local_array_size, res_displs, total_processor_num, 0);
    MPI_Gatherv(local_num_arr, comm_total_size, MPI_INT, recv_buf, local_array_size,  res_displs, MPI_INT, 0, MPI_COMM_WORLD);
    double end_distribution_time = MPI_Wtime();

    if (global_rank == 0){
        ofstream file(argv[2]);
        if (file.is_open()) {
            for (int i = 0; i < num_count; i++) file << recv_buf[i] << " ";
            file << endl;
            file << end_time - start_time;
            file.close();
        }
        else {
            cout << "Error opening file!" << endl;
        }
        double end_io_time = MPI_Wtime();
        cout << "Parallel sorting time: " << end_time - start_time << endl;
        cout << "Parallel sorting and distribution and gather time: " << end_distribution_time - start_distribution_time << endl;
        cout << "Total time w/ parallel soring, I/O, distribution and gather: " << end_io_time - start_io_time << endl;
    }
    delete[] local_num_arr;
    delete[] recv_buf;
    MPI_Finalize();
}
