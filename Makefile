NVFLAGS=-g -O2 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
MPI_LIB= -L/usr/local/cuda/lib64 -lcudart
distCu= vec_analytics_dis.o

all: vec_analy

vec_analy: vec_analytics.cu
	nvcc $(NVFLAGS) $^ -o vec_analytics

dist: $(distCu) vec_analytics_dis.cpp
	mpic++ $^ -o vec_analytics_dis $(MPI_LIB)

$(distCu): vec_analytics_dis.cu
	nvcc $(NVFLAGS) $^ -c $(distCu)

clean: 
	-rm -f vec_analytics *.o
