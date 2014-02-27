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

void readElements(int fp, int numElements, vector<float> *elements, 
 int commRank) {
   float temp;
   lseek(fp, (FLOAT_SIZE * numElements / 4) * commRank, SEEK_CUR) 
   for (int i = 0; i < numElements / 4; i++) {
      if (read(fp, &temp, FLOAT_SIZE) == -1) {
         perror("read");
         exit(1);
      }
      (*elements).push_back(temp);
   }
}

int main(int argc, char **argv) {
   int fp;
   int numElements;
   int commSize;
   int commRank;
   int dataSizePerNode;
   vector<float> elements;

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


   // Launch the processes
   MPI_Init(&argc, &argv);

   // Get our MPI node number and node count
   MPI_Comm_size(MPI_COMM_WORLD, &commSize);
   MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

   readElements(fp, numElements, &elements, commRank);
   dataSizePerNode = numElements / 4;
   // Allocate a buffer on each node
   vector<float> elementsNode;
   for (int x = 0; x < dataSizePerNode; x++) {
      elementsNode.push_back(elements[x + (dataSizePerNode * commRank)]);
   }

   // Dispatch portions of input data to each node
   MPI_Scatter(elements, dataSizePerNode, MPI_FLOAT, 
    elementsNode, dataSizePerNode, MPI_FLOAT,
    0, MPI_COMM_WORLD);

   // Compute Mean
   float mean_node = computeMean(elementsNode);   

   // Standard Deviation and create histogram
   int histo[100];
   float mean;

   MPI_Reduce(&mean_node, &mean, 2, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   if (!commRank) {
      mean /=numElements;
   }  
   
   float standardDeviation_node = computeStdDeviation(histo, elementsNode, 
    mean);

   // Compute the Min and Max
   float minimum_node = getMin(elementsNode);
   float maximum_node = getMax(elementsNode);

   // Global variables to use
   float standardDeviation;
   float minimumHelper[4] = {0, 0, 0, 0};
   float maximumHelper[4] = {0, 0, 0, 0};
   float minimum;
   float maximum;
   int histoArr[100];

   MPI_Reduce(&standardDeviation_node, &standardDeviation, 2, 
    MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&minimum_node, &minimumHelper[commRank], 2, 
    MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&maximum_node, &maximumHelper[commRank], 2,
    MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Allreduce(histo, histoArr, 100, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

   if (!commRank) {
      // final calculations
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