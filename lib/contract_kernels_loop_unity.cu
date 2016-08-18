#include <contract.h>
#include <constants.h>
#include <device_opts_inline.h>
#include <utils.h>
#include <stdio.h>

using namespace contract;

// ==================================================== // 
// !!!!!!! for now the code will work only with 100 eigenVectors
// !!!!!!! for now the code will work only with submatrix side 25 ==> 25x25=625 threads
#define BLOCK_SIZE 25
#define NSIZE 100

//====================================================//

__global__ void calculate_loop_unity_kernel_float(float2* out, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMom, float2* tmp){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2
  
#include <calculate_loop_unity_core.h>
  
#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2
}

//================================================//
__global__ void calculate_loop_unity_kernel_double(double2* out, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMom, double2* tmp){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2
  
#include <calculate_loop_unity_core.h>
  
#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2
}

//===================================================//


template<typename Float2, typename Float>
static void calculate_loop_unity_kernel(cudaTextureObject_t texPropDiag, cudaTextureObject_t texMom, int Nt, Float* loop){
  
  int numBlocks = Nt * 4; // for spin combinations on the diagonal
  dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE,1);
  dim3 gridDim(numBlocks,1,1);

  Float *h_loop = NULL;
  h_loop = (Float*) malloc(numBlocks*2*sizeof(Float));
  if(h_loop == NULL)
    ABORT("Error allocating memory\n");
  Float *d_loop = NULL;
  cudaMalloc((void**)&d_loop, numBlocks*2*sizeof(Float));
  CHECK_CUDA_ERROR();
  
  Float *tmp = NULL;
  cudaMalloc((void**)&tmp, numBlocks*NSIZE*NSIZE*2*sizeof(Float));
  CHECK_CUDA_ERROR();

  //+++++++++++++
  if( typeid(Float2) == typeid(float2) ){
    calculate_loop_unity_kernel_float<<<gridDim,blockDim>>>((float2*) d_loop, texPropDiag, texMom, (float2*) tmp);
  }
  else if ( typeid(Float2) == typeid(double2) ){
    calculate_loop_unity_kernel_double<<<gridDim,blockDim>>>((double2*) d_loop, texPropDiag, texMom, (double2*) tmp);    
  }
  else
    ABORT("Something fishy is happening\n");
  //+++++++++++++
  cudaMemcpy(h_loop, d_loop, numBlocks*2*sizeof(Float), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR();

  //  Float *h_loop_reduced = (Float*) calloc(Nt*2,sizeof(Float));
  // if(h_loop_reduced == NULL) ABORT("Error allocating memory for reduction\n");

  memset(loop,0,Nt*2*sizeof(Float));

  for(int it = 0 ; it < Nt ; it++)
    for(int is = 0 ; is < 4 ; is++){
      loop[it*2+0] += h_loop[it*4*2+is*2+0];
      loop[it*2+1] += h_loop[it*4*2+is*2+1];
    }

  free(h_loop);
  cudaFree(d_loop);
  cudaFree(tmp);
  CHECK_CUDA_ERROR();
}

//=================================================//
void contract::run_ContractLoopUnity(cudaTextureObject_t texPropDiag, cudaTextureObject_t texMom, int Nt, void* loop, PRECISION prec){

  if(prec == SINGLE){
    calculate_loop_unity_kernel<float2,float>(texPropDiag, texMom, Nt, (float*) loop);
  }
  else if (prec == DOUBLE){
    calculate_loop_unity_kernel<double2,double>(texPropDiag, texMom, Nt, (double*) loop);
  }
  else{
    ABORT("Error: this precision in not implemented");
  }

}
