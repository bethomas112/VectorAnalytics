NVFLAGS=-g -O2 -o vec_analytics -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

all: vec_analy

vec_analy: vec_analytics.cu
	nvcc $(NVFLAGS) $^

clean: 
	-rm -r vec_analytics
