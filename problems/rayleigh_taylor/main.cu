#include <mpi.h>
#include "Problem.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    run_problem(argc, argv);
    MPI_Finalize();
    return 0;
}
