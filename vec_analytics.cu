#include <fcntl.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <unistd.h>

#define FLOAT_SIZE 4
#define MAX_ELEMENTS 500000000
#define SPLIT_NUM 2

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

void readElements(int fp, int numElements, float *elements, int size) {
   if (read(fp, elements, size) == -1) {
      perror("read");
      exit(1);
   }
}

int main(int argc, char **argv) {
   int fp;
   int readSize;
   int numElements;
   float mean;
   float standardDeviation;
   float min = 101;
   float max = -1;
   float newMin, newMax;
   int count = 0;

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

   float *elements;
   
   if (numElements <= MAX_ELEMENTS) {
      readSize = numElements;
   }
   else {
      readSize = numElements / SPLIT_NUM;
   }
   printf("readSize: %d\n", readSize);
   elements = (float *)malloc(sizeof(float) * readSize);
   if (!elements) {
      perror("malloc");
      exit(1);
   }
  
   // Allocate space on device 
   device_vector<float> d_elements(readSize);
   int *dHistArr;
   HANDLE_ERROR(cudaMalloc(&dHistArr, sizeof(int) * 100));
   HANDLE_ERROR(cudaMemset(dHistArr, 0, sizeof(int) * 100));


   while (count < numElements) {
      // Read Elements in from file 
      readElements(fp, numElements, elements, readSize * FLOAT_SIZE);
      
      // Copy Elements to Device
      thrust::copy(elements, elements + readSize, d_elements.begin());
      // Compute Mean
      mean += reduce(d_elements.begin(), d_elements.end(), 0.0, 
       plus<float>());
      
      // Compute the Min and Max
      newMin = *min_element(d_elements.begin(), d_elements.end());
      newMax = *max_element(d_elements.begin(), d_elements.end());

      min = newMin < min ? newMin : min;
      max = newMax > max ? newMax : max;
      
      count += readSize;
   }

   lseek(fp, FLOAT_SIZE, SEEK_SET);
   count = 0;
   mean /= numElements;

   while (count < numElements) {

      if (readSize != numElements) {
         // Read Elements in from file 
         readElements(fp, numElements, elements, readSize * FLOAT_SIZE);
         // Copy Elements to Device
         thrust::copy(elements, elements + readSize, d_elements.begin());
      }

      // Standard Deviation and create histogram
      standardDeviation += transform_reduce(d_elements.begin(), 
       d_elements.end(), std_dev_help(mean, dHistArr), 0.0, plus<float>());
      count += readSize;
   }
   standardDeviation = sqrt(standardDeviation / numElements);
   
   int histoArr[100];
   HANDLE_ERROR(cudaMemcpy(histoArr, dHistArr, sizeof(int) * 100, 
    cudaMemcpyDeviceToHost));

   // Print Stats
   cout << "Count              : " << numElements << "\n";
   cout << "Minimum            : " << min << "\n";
   cout << "Maximum            : " << max << "\n";
   cout << "Mean               : " << mean << "\n"; 
   cout << "Standard Deviation : " << standardDeviation << "\n";

   // Write the Histogram File
   FILE *outfile = fopen(argv[2], "w");
   for (int x = 0; x < 100; x++) {
      fprintf(outfile, "%d, %d\n", x, histoArr[x]);
   }
   free(elements);
   return 0;
   
}
