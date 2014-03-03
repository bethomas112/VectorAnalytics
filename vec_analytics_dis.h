#ifndef VEC_ANALYTICS_DIS_H
#define VEC_ANALYTICS_DIS_H
#include <vector>

using namespace std;


float computeMean(float *elements, int numElements);

float computeStdDevMinMax(float *elements, int *histo, float mean,
 int numElements, float *min, float *max);

#endif 
