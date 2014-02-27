#include <fcntl.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <cmath>
#include <mpi.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <unistd.h>

#define FLOAT_SIZE 4

using std::cout;
using namespace thrust;

/* This macro was taken from the book CUDA by example. This code is used for
 * error checking */
static void HandleError(cudaError_t err, const char *file, int line ) { 
   if (err != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
      exit(1);
   }   
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

struct std_dev_help : public unary_function<float, float> {
   float standardMean;
   int *histoArr;
   std_dev_help(float mean, int *histo) {
      standardMean = mean;
      histoArr = histo;
   }
   __device__ float operator()(float &x) const {
      if (x > 99) {
         atomicAdd(&histoArr[99], 1);
      }
      else {
         atomicAdd(&histoArr[(int)x], 1);
      }
      return (x - standardMean) * (x - standardMean);
   }
};

void readElements(int fp, int numElements, host_vector<float> *elements) {
   float temp;
   for (int i = 0; i < numElements; i++) {
      if (read(fp, &temp, FLOAT_SIZE) == -1) {
         perror("read");
         exit(1);
      }
      elements.push_back(temp);
   }
}

int main(int argc, char **argv) {
   int fp;
   int numElements;
   int commSize;
   int commRank;
   host_vector<float> elements;

   if (argc < 3) {
      fprintf(stderr, "Usage: %s <infile> <histogram outfile>\n", *argv);
      exit(1);
   }

   // Launch the processes
   MPI_Init(&argc, &argv);

   // Get our MPI node number and node count
   MPI_Comm_size(MPI_COMM_WORLD, &commSize));
   MPI_Comm_rank(MPI_COMM_WORLD, &commRank));

   if (!commRank) {
      if ((fp = open(argv[1], O_RDONLY)) == -1) {
         perror("open");
         exit(1);
      }

      if (read(fp, &numElements, FLOAT_SIZE) == -1) {
         perror("read");
         exit(1);
      }
      int dataSizePerNode = numElements / 4;

      readElements(fp, numElements, &elements);
   }

   // Allocate a buffer on each node
   host_vector<float> elementsNode;

   for (int x = 0; x < dataSizePerNode; x++) {
      elementsNode.push_back(elements[x + (dataSizePerNode * commRank)]);
   }

   // Dispatch portions of input data to each node
   MPI_Scatter(elements, dataSizePerNode, MPI_FLOAT, 
    elementsNode, dataSizePerNode, MPI_FLOAT,
    0, MPI_COMM_WORLD);

   // Compute Mean
   device_vector<float> d_elements = elementsNode;
   float mean_node = reduce(d_elements.begin(), d_elements.end(), 0.0, plus<float>())
   
   int *dHistArr;
   HANDLE_ERROR(cudaMalloc(&dHistArr, sizeof(int) * 100));
   HANDLE_ERROR(cudaMemset(dHistArr, 0, sizeof(int) * 100));

   // Standard Deviation and create histogram
   float standardDeviation_node = transform_reduce(d_elements.begin(), 
    d_elements.end(), std_dev_help(mean, dHistArr), 0.0, plus<float>());

   // Compute the Min and Max
   float minimum_node = *min_element(d_elements.begin(), d_elements.end());
   float maximum_node = *max_element(d_elements.begin(), d_elements.end());
   
   int histo[100];
   HANDLE_ERROR(cudaMemcpy(histoArr, dHistArr, sizeof(int) * 100, 
    cudaMemcpyDeviceToHost));

   // Global variables to use
   float standardDeviation;
   float minimumHelper[4] = {0, 0, 0, 0};
   float maximumHelper[4] = {0, 0, 0, 0};
   float minimum;
   float maximum;
   float mean;
   int histoArr[100];

   MPI_Reduce(&mean_node, &mean, 2, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&standardDeviation_node, &standardDeviation, 2, 
    MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&minimum_node, &minimumHelper[commRank], 2, 
    MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&maximum_node, &maximumHelper[commRank], 2,
    MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Allreduce(histo, histoArr, 100, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   if (!commRank) {
      // final calculations
      mean /= numElements;
      standardDeviation = sqrt(standardDeviation / numElements);
      minimum = minimumHelper[0];
      maximum = maximumHelper[0];
      for (int find = 1; find < 4; find++) {
         if (minimumHelper[find] < minimum) {
            minimum = minimumHelper[find];
         }
         if (maximumHelper[find] > maximum) {
            maximum = maximumHelper[find];
         }
      }

      // Print Stats
      cout << "Count              : " << numElements << "\n";
      cout << "Minimum            : " << minimum << "\n";
      cout << "Maximum            : " << maximum << "\n";
      cout << "Mean               : " << mean << "\n"; 
      cout << "Standard Deviation : " << standardDeviation << "\n";

      // Write the Histogram File
      FILE *outfile = fopen(argv[2], "w");
      for (int x = 0; x < 100; x++) {
         fprintf(outfile, "%d, %d\n", x, histoArr[x]);
      }
   }
   MPI_Finalize();
   return 0;
}
