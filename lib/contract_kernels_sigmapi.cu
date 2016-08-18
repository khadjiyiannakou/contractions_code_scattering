#include <contract.h>
#include <constants.h>
#include <device_opts_inline.h>
#include <utils.h>
#include <stdio.h>

using namespace contract;

// =========== Constant memory references ================//

__constant__ short int c_spinIndices_sigmapi_hor_fish[16][2];
__constant__ float c_coef_sigmapi_hor_fish[16][2];

__constant__ short int c_spinIndices_sigmapi_triangle[128][5];
__constant__ short int c_ip_sigmapi_triangle[128];
__constant__ float c_coef_sigmapi_triangle[128][2];

//========================================================//

bool isConstantSigmaPiOn = false;

static void copy_constants_sigmapi(){
  cudaMemcpyToSymbol(c_spinIndices_sigmapi_hor_fish, spinIndices_sigmapi_hor_fish, 16*2*sizeof(short int));
  cudaMemcpyToSymbol(c_coef_sigmapi_hor_fish, coef_sigmapi_hor_fish, 16*2*sizeof(float));
  cudaMemcpyToSymbol(c_spinIndices_sigmapi_triangle, spinIndices_sigmapi_triangle, 128*5*sizeof(short int));
  cudaMemcpyToSymbol(c_ip_sigmapi_triangle, ip_sigmapi_triangle, 128*sizeof(short int));
  cudaMemcpyToSymbol(c_coef_sigmapi_triangle, coef_sigmapi_triangle, 128*2*sizeof(float));

  CHECK_CUDA_ERROR();

#ifdef DEVICE_DEBUG
  printf("Copy for rho constants to device finished\n");
#endif

}

//=================================================================//
// ==================================================== // 
// !!!!!!! for now the code will work only with 100 eigenVectors
// !!!!!!! for now the code will work only with submatrix side 25 ==> 25x25=625 threads
#define BLOCK_SIZE 25
#define NSIZE 100
//====================================================//

//====================================================//
__global__ void calculate_sigmapi_hor_fish_kernel_float(float2* out, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, float2* tmp1, float2* tmp2){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2

#include <calculate_sigmapi_hor_fish_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2
}
//==================================================//

//====================================================//
__global__ void calculate_sigmapi_hor_fish_kernel_double(double2* out, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, double2* tmp1, double2* tmp2){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2

#include <calculate_sigmapi_hor_fish_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2
}
//==================================================//
__global__ void calculate_sigmapi_triangle_kernel_float(float2* out, cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP1P2, int tf, float2* tmp1, float2* tmp2){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2

#include <calculate_sigmapi_triangle_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2
}
//====================================================//
__global__ void calculate_sigmapi_triangle_kernel_double(double2* out, cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP1P2, int tf, double2* tmp1, double2* tmp2){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2

#include <calculate_sigmapi_triangle_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2
}
//=======================================================//

template<typename Float2, typename Float> 
static void calculate_sigmapi_hor_fish_kernel(cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, int  Nt, Float* corr){
  if(!isConstantSigmaPiOn)
    ABORT("Error: You need to initialize device constants before calling Kernels\n");

  int numBlocks = Nt * 16; 
  dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE,1); // 625 threads
  dim3 gridDim(numBlocks,1,1);

  Float *h_partial_block = NULL;
  Float *d_partial_block = NULL;
  h_partial_block = (Float*) malloc(numBlocks*2*sizeof(Float));
  if(h_partial_block == NULL) ABORT("Error allocating memory\n");
  cudaMalloc((void**)&d_partial_block, numBlocks*2*sizeof(Float));
  CHECK_CUDA_ERROR();

  Float *tmp1 = NULL;
  Float *tmp2 = NULL;
  cudaMalloc((void**)&tmp1, numBlocks*NSIZE*NSIZE*2*sizeof(Float));
  CHECK_CUDA_ERROR();
  cudaMalloc((void**)&tmp2, numBlocks*NSIZE*NSIZE*2*sizeof(Float));
  CHECK_CUDA_ERROR();

  if( typeid(Float2) == typeid(float2) )
    calculate_sigmapi_hor_fish_kernel_float<<<gridDim,blockDim>>>((float2*) d_partial_block, texPropDiag, texMomP1, texMomP2, (float2*) tmp1, (float2*) tmp2);
  else if( typeid(Float2) == typeid(double2) )
    calculate_sigmapi_hor_fish_kernel_double<<<gridDim,blockDim>>>((double2*) d_partial_block, texPropDiag, texMomP1, texMomP2, (double2*) tmp1, (double2*) tmp2);
  else
    ABORT("Something fishy is happening\n");

  cudaMemcpy(h_partial_block, d_partial_block, numBlocks*2*sizeof(Float), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR();

  memset(corr,0,Nt*2*sizeof(Float));

  for(int ti = 0 ; ti < Nt ; ti++)
    for(int is = 0 ; is < 16 ; is++){
      corr[ti*2 + 0] += h_partial_block[ti*16*2 + is*2 + 0];
      corr[ti*2 + 1] += h_partial_block[ti*16*2 + is*2 + 1];
    }

  // clean memory
  free(h_partial_block);
  cudaFree(d_partial_block);
  cudaFree(tmp1);
  cudaFree(tmp2);
  CHECK_CUDA_ERROR();
}

template<typename Float2, typename Float> 
static void calculate_sigmapi_triangle_kernel(cudaTextureObject_t texProp,cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP1P2, int tf,int  Nt, Float* corr){
  if(!isConstantSigmaPiOn)
    ABORT("Error: You need to initialize device constants before calling Kernels\n");

  int numBlocks = Nt * 2 * 64; 
  dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE,1); // 625 threads
  dim3 gridDim(numBlocks,1,1);

  Float *h_partial_block = NULL;
  Float *d_partial_block = NULL;
  h_partial_block = (Float*) malloc(numBlocks*2*sizeof(Float));
  if(h_partial_block == NULL) ABORT("Error allocating memory\n");
  cudaMalloc((void**)&d_partial_block, numBlocks*2*sizeof(Float));
  CHECK_CUDA_ERROR();

  Float *tmp1 = NULL;
  Float *tmp2 = NULL;
  cudaMalloc((void**)&tmp1, numBlocks*NSIZE*NSIZE*2*sizeof(Float));
  CHECK_CUDA_ERROR();
  cudaMalloc((void**)&tmp2, numBlocks*NSIZE*NSIZE*2*sizeof(Float));
  CHECK_CUDA_ERROR();

  if( typeid(Float2) == typeid(float2) )
    calculate_sigmapi_triangle_kernel_float<<<gridDim,blockDim>>>((float2*) d_partial_block, texProp, texPropDiag, texMomP1, texMomP2, texMomP1P2, tf, (float2*) tmp1, (float2*) tmp2);
  else if( typeid(Float2) == typeid(double2) )
    calculate_sigmapi_triangle_kernel_double<<<gridDim,blockDim>>>((double2*) d_partial_block, texProp, texPropDiag, texMomP1, texMomP2, texMomP1P2, tf, (double2*) tmp1, (double2*) tmp2);
  else
    ABORT("Something fishy is happening\n");

  cudaMemcpy(h_partial_block, d_partial_block, numBlocks*2*sizeof(Float), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR();

  memset(corr,0,Nt*2*2*sizeof(Float));

  for(int ti = 0 ; ti < Nt ; ti++)
    for(int iop = 0 ; iop < 2 ; iop++)
      for(int is = 0 ; is < 64 ; is++){
	corr[ti*2*2 + iop*2 + 0] += h_partial_block[ti*2*64*2 + iop*64*2 + is*2 + 0];
	corr[ti*2*2 + iop*2 + 1] += h_partial_block[ti*2*64*2 + iop*64*2 + is*2 + 1];
      }

  // clean memory
  free(h_partial_block);
  cudaFree(d_partial_block);
  cudaFree(tmp1);
  cudaFree(tmp2);
  CHECK_CUDA_ERROR();
}

void contract::run_ContractSigmaPi_fish_hor(cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, int Nt, void* corr, PRECISION prec){

  if(prec == SINGLE){
    calculate_sigmapi_hor_fish_kernel<float2,float>(texPropDiag, texMomP1, texMomP2, Nt, (float*) corr);
  }
  else if (prec == DOUBLE){
    calculate_sigmapi_hor_fish_kernel<double2,double>(texPropDiag, texMomP1, texMomP2, Nt, (double*) corr);
  }
  else{
    ABORT("Error: this precision in not implemented");
  }

}

void contract::run_ContractSigmaPi_triangle(cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP1P2, int tf, int Nt, void* corr, PRECISION prec){

  if(prec == SINGLE){
    calculate_sigmapi_triangle_kernel<float2,float>(texProp, texPropDiag, texMomP1, texMomP2, texMomP1P2, tf, Nt, (float*) corr);
  }
  else if (prec == DOUBLE){
    calculate_sigmapi_triangle_kernel<double2,double>(texProp, texPropDiag, texMomP1, texMomP2, texMomP1P2, tf, Nt, (double*) corr);
  }
  else{
    ABORT("Error: this precision in not implemented");
  }

}

void contract::run_CopyConstantsSigmaPi(){
    if(isConstantSigmaPiOn){
    WARNING("Warning: Copy constants for sigmapi again will be skipped\n");
    return;
  }
  copy_constants_sigmapi();
  isConstantSigmaPiOn = true;
}
