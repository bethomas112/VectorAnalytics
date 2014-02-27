#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include "vec_analytics_dis.h"

using std::vector;

/* This macro was taken from the book CUDA by example. This code is used for
 * error checking */
static void HandleError(cudaError_t err, const char *file, int line ) { 
   if (err != cudaSuccess) {
      printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
      exit(1);
   }   
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

struct std_dev_help : public thrust::unary_function<float, float> {
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


float computeMean(vector<float> elementsNode) {
   thrust::device_vector<float> d_elements = elementsNode;
   return thrust::reduce(d_elements.begin(), d_elements.end(), 0.0, 
    thrust::plus<float>());
}

float computeStdDeviation(int *histo, vector<float> elementsNode,
 float mean) {
   int *dHistArr; 
   float stdDeviation;
   thrust::device_vector<float> d_elements = elementsNode;
   
   HANDLE_ERROR(cudaMalloc(&dHistArr, sizeof(int) * 100));
   HANDLE_ERROR(cudaMemset(dHistArr, 0, sizeof(int) * 100));
   
   stdDeviation = thrust::transform_reduce(d_elements.begin(), 
    d_elements.end(), std_dev_help(mean, dHistArr), 0.0, 
    thrust::plus<float>());
  
   HANDLE_ERROR(cudaMemcpy(histo, dHistArr, sizeof(int) * 100,
    cudaMemcpyDeviceToHost));
   
   HANDLE_ERROR(cudaFree(dHistArr));
   
   return stdDeviation;
}

float getMin(vector<float> elementsNode) {
   thrust::device_vector<float> d_elements = elementsNode;
   return *min_element(d_elements.begin(), d_elements.end());
}

float getMax(vector<float> elementsNode) {
   thrust::device_vector<float> d_elements = elementsNode;
   return *max_element(d_elements.begin(),d_elements.end());
}
