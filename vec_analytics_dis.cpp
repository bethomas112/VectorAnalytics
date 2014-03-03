#include <fcntl.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <cmath>
#include <mpi.h>
#include <unistd.h>

#include "vec_analytics_dis.h"

#define FLOAT_SIZE 4

using std::cout;
using std::vector;

void readElements(int fp, int numElements, float *elements, 
 int commRank) {
   float temp;
   lseek(fp, (FLOAT_SIZE * (numElements / 4)) * commRank, SEEK_CUR);
   if (read(fp, elements, FLOAT_SIZE * (numElements / 4)) == -1) {
      perror("read in readElements");
      exit(1);
   }
}

int main(int argc, char **argv) {
   int fp;
   int numElements;
   int commSize;
   int commRank;
   int dataSizePerNode;
   float *elements;

   if (argc < 3) {
      fprintf(stderr, "Usage: %s <infile> <histogram outfile>\n", *argv);
      exit(1);
   }

   if ((fp = open(argv[1], O_RDONLY)) == -1) {
      perror("open");
      exit(1);
   }
   if (read(fp, &numElements, FLOAT_SIZE) == -1) {
      perror("read");
      exit(1);
   }
   
   elements = (float *)malloc(sizeof(float) * (numElements / 4));
   if (!elements) {
      perror("malloc");
      exit(1);
   }

   // Launch the processes
   MPI_Init(&argc, &argv);

   // Get our MPI node number and node count
   MPI_Comm_size(MPI_COMM_WORLD, &commSize);
   MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

   readElements(fp, numElements, elements, commRank);
   dataSizePerNode = numElements / 4;

   // Compute Mean
   float mean_node = computeMean(elements, dataSizePerNode);   

   // Standard Deviation and create histogram
   int histo[100];
   float mean;

   MPI_Allreduce(&mean_node, &mean, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

   mean /=numElements;
  
   // Compute Std Deviation, min and max
   float minimum_node;
   float maximum_node;
   float standardDeviation_node = computeStdDevMinMax(elements, histo, mean,
    numElements / 4, &minimum_node, &maximum_node);

   // Global variables to use
   float standardDeviation;
   float minimum;
   float maximum;
   int histoArr[100];

   MPI_Reduce(&standardDeviation_node, &standardDeviation, 1, 
    MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&minimum_node, &minimum, 1, 
    MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&maximum_node, &maximum, 1,
    MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Allreduce(histo, histoArr, 100, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   if (!commRank) {
      // final calculations
      standardDeviation = sqrt(standardDeviation / numElements);

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
