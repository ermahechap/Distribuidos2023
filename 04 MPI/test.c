#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Check for at least 2 processes
    if (world_size < 2) {
        fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int tag1 = 0;
    int tag2 = 1;

    if (world_rank == 0) {
        // Process 0 sends two variables
        int variable1 = 123;
        double variable2 = 456.78;

        MPI_Send(&variable1, 1, MPI_INT, 1, tag1, MPI_COMM_WORLD);
        MPI_Send(&variable2, 1, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD);
    } else if (world_rank == 1) {
        // Process 1 receives the two variables
        int received_var1;
        double received_var2;

        MPI_Recv(&received_var1, 1, MPI_INT, 0, tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&received_var2, 1, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Process 1 received number %d and %f from process 0\n", received_var1, received_var2);
    }

    MPI_Finalize();
    return 0;
}