#ifndef VEC_ANALYTICS_DIS_H
#define VEC_ANALYTICS_DIS_H
#include <vector>

using namespace std;


float computeMean(vector<float> elementsNode);

float computeStdDeviation(int *histo, vector<float> elementsNode);

float getMin(vector<float> elementsNode);
   
float getMax(vector<float> elementsNode);
#endif
