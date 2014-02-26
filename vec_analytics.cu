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
   std_dev_help(float mean) {
      standardMean = mean;
   }
   __host__ __device__ T operator()(float &x) const {
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

   // Calculating mean
   device_vector<float> d_elements = elements;
   mean = reduce(d_elements.begin(), d_elements.end(), 0.0, plus<float>()) / numElements;
   std::cout << mean << "\n";

   standardDeviation = transform_reduce(d_elements.begin(), d_elements.end(), std_dev_help(mean), 0.0, plus<float>()) / numElements;
   std::cout << standardDeviation << "\n";

   return 0;
}
