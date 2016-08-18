#ifndef _UTILS_H
#define _UTILS_H

#define CHECK_CUDA_ERROR() do {						\
    cudaError_t error = cudaGetLastError();				\
    if (error != cudaSuccess){						\
      fprintf(stderr,"The last cuda error is %s\n",cudaGetErrorString(error)); \
      exit(EXIT_FAILURE);						\
    }									\
  }while(0)

#define ABORT(...) do {				\
  fprintf(stderr,"%s",__VA_ARGS__);		\
  fprintf(stderr,"Error in file %s, line %d, function %s\n",__FILE__, __LINE__, __func__); \
  fflush(stderr);							\
  exit(EXIT_FAILURE);							\
}while(0)

#define WARNING(...) do {				\
  fprintf(stderr,"%s",__VA_ARGS__);		\
  fprintf(stderr,"Warning in file %s, line %d, function %s\n",__FILE__, __LINE__, __func__); \
  fflush(stderr);							\
}while(0)

#endif
