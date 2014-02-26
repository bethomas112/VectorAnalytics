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

using namespace thrust;

struct std_dev_help : public unary_function<float, float> {
   float standardMean;
   int *histoArr;
   std_dev_help(float mean, int *histo) {
      standardMean = mean;
      histoArr = histo;
   }
   __device__ float operator()(float &x) const {
      atomicAdd(&histoArr[(int)x], 1);
      return (x - standardMean) * (x - standardMean);
   }
};

void readElements(int fp, int numElements, host_vector<float> *elements) {
   float temp;
   for (int i = 0; i < numElements; i++) {
      read(fp, &temp, FLOAT_SIZE);
      (*elements)[i] = temp;
   }
}

int main(int argc, char **argv) {
   int fp;
   int numElements;
   float mean;
   float standardDeviation;
   float min;
   float max;

   if (argc < 2) {
      fprintf(stderr, "Usage: %s <infile>\n", *argv);
      exit(1);
   }

   if ((fp = open(argv[1], O_RDONLY)) == -1) {
      perror("open");
      exit(1);
   }

   read(fp, &numElements, FLOAT_SIZE);
   host_vector<float> elements(numElements);
   
   readElements(fp, numElements, &elements);

   // Mean
   device_vector<float> d_elements = elements;
   mean = reduce(d_elements.begin(), d_elements.end(), 0.0, plus<float>()) / numElements;
   std::cout << mean << "\n";
   
   int *dHistArr;
   cudaMalloc(&dHistArr, sizeof(int) * 100);
   cudaMemset(dHistArr, 0, sizeof(int) * 100);
   // Standard Deviation and create histogram
   standardDeviation = sqrt(transform_reduce(d_elements.begin(), d_elements.end(), std_dev_help(mean, dHistArr), 0.0, plus<float>()) / numElements);
   std::cout << standardDeviation << "\n";

   min = *min_element(d_elements.begin(), d_elements.end());
   std::cout << min << "\n";

   max = *max_element(d_elements.begin(), d_elements.end());
   std::cout << max << "\n";
   
   int histoArr[100];
   cudaMemcpy(histoArr, dHistArr, sizeof(int) * 100, cudaMemcpyDeviceToHost);
   for (int x = 0; x < 100; x++) {
      std::cout << x << ", " << histoArr[x] << "\n";
   }
   return 0;
}
