#include <contract.h>
#include <constants.h>
#include <device_opts_inline.h>
#include <utils.h>
#include <stdio.h>

using namespace contract;


// =========== Constant memory references ================//

__constant__ short int c_spinIndices_rhopi[768][6]; // 8 Kb
__constant__ short int c_ip_rhopi[768]; // 1.5 Kb
__constant__ float c_coef_rhopi[768][2]; // 12 Kb
                                             // 21.5 Kb in total
// =======================================================// 

bool isConstantRhoPiPiOn = false;

static void copy_constants_rhopi(){

  cudaMemcpyToSymbol(c_spinIndices_rhopi, spinIndices_rhopi, 768*6*sizeof(short int));
  cudaMemcpyToSymbol(c_ip_rhopi, ip_rhopi, 768*sizeof(short int));
  cudaMemcpyToSymbol(c_coef_rhopi, coef_rhopi, 768*2*sizeof(float) );
  CHECK_CUDA_ERROR();
}

//=======================================================//
// !!!!!!! for now the code will work only with 100 eigenVectors
// !!!!!!! for now the code will work only with submatrix side 25 ==> 25x25=625 threads
#define BLOCK_SIZE 25
#define NSIZE 100

//=====================================================//
__global__ void calculate_rhopi_kernel_float(float2* out, cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP1P2, int ti, int tf, int idir, float2* tmp1, float2* tmp2){

#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2

#include <calculate_rhopi_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2


}

//==================================================//
__global__ void calculate_rhopi_kernel_double(double2* out, cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP1P2, int ti, int tf, int idir, double2* tmp1, double2* tmp2){

#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2

#include <calculate_rhopi_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2
}
// ==================================================//
template<typename Float2, typename Float>
static void calculate_rhopi_kernel(cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP1P2, int tf, int Nt, int idir, Float* corr){

  if(!isConstantRhoPiPiOn)
    ABORT("Error: You need to initialize device constants before calling Kernels\n");

  int numBlocks = 4 * 64; // 4 is for the four different rho operators and 64 is for the different non zero spin combinations
  // the different 3 spatial directions on the lattice can be chosen using idir

  dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE,1); // 625 threads
  dim3 gridDim(numBlocks,1,1);            // 256 blocks

  Float *h_partial_block = NULL;
  Float *d_partial_block = NULL;
  h_partial_block = (Float*) malloc(Nt*numBlocks*2*sizeof(Float));
  if(h_partial_block == NULL) ABORT("Error allocating memory\n");
  cudaMalloc((void**)&d_partial_block, Nt*numBlocks*2*sizeof(Float));
  CHECK_CUDA_ERROR();

  Float *tmp1 = NULL;
  Float *tmp2 = NULL;
  cudaMalloc((void**)&tmp1, numBlocks*NSIZE*NSIZE*2*sizeof(Float));
  CHECK_CUDA_ERROR();
  cudaMalloc((void**)&tmp2, numBlocks*NSIZE*NSIZE*2*sizeof(Float));
  CHECK_CUDA_ERROR();

  for(int ti = 0 ; ti < Nt ; ti++){
    if( typeid(Float2) == typeid(float2) )
      calculate_rhopi_kernel_float<<<gridDim,blockDim>>>((float2*) d_partial_block, texProp, texPropDiag,texMomP1,texMomP2,texMomP1P2, ti, tf, idir, (float2*) tmp1, (float2*) tmp2);
    else if( typeid(Float2) == typeid(double2) )
      calculate_rhopi_kernel_double<<<gridDim,blockDim>>>((double2*) d_partial_block, texProp, texPropDiag,texMomP1,texMomP2,texMomP1P2, ti, tf, idir, (double2*) tmp1, (double2*) tmp2);
    else
      ABORT("Something fishy is happening\n");

  }

  cudaMemcpy(h_partial_block, d_partial_block, Nt*numBlocks*2*sizeof(Float), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR();


  
  memset(corr,0,Nt*4*2*sizeof(Float));

  for(int ti = 0 ; ti < Nt ; ti++)
    for(int ico = 0 ; ico < 4 ; ico++)
	for(int is = 0 ; is < 64 ; is++){
	  corr[ti*4*2 + ico*2 + 0] += h_partial_block[ti*4*64*2 + ico*64*2 + is*2 + 0];
	  corr[ti*4*2 + ico*2 + 1] += h_partial_block[ti*4*64*2 + ico*64*2 + is*2 + 1];
	}
  

  // clean memory
  free(h_partial_block);
  cudaFree(d_partial_block);
  cudaFree(tmp1);
  cudaFree(tmp2);
  CHECK_CUDA_ERROR();


}
//===================================================//
void contract::run_ContractRhoPi(cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP1P2, int tf, int Nt, int idir, void* corr, PRECISION prec){

  if(prec == SINGLE){
    calculate_rhopi_kernel<float2,float>(texProp,texPropDiag,texMomP1,texMomP2,texMomP1P2,tf,Nt,idir,(float*) corr);
  }
  else if (prec == DOUBLE){
    calculate_rhopi_kernel<double2,double>(texProp,texPropDiag,texMomP1,texMomP2,texMomP1P2,tf,Nt,idir,(double*) corr);
  }
  else{
    ABORT("Error: this precision in not implemented");
  }

}
//===================================================//
void contract::run_CopyConstantsRhoPi(){
  if(isConstantRhoPiPiOn){
    WARNING("Warning: Copy constants for rho-pi again will be skipped\n");
    return;
  }
  copy_constants_rhopi();
  isConstantRhoPiPiOn = true;
}
//==================================================//


  /*
  if(tf == 0){
    for(int is = 0 ; is < 64 ; is++)
      printf("%+e %+e\n",h_partial_block[0*4*64*2 + 0*64*2 + is*2 + 0], h_partial_block[0*4*64*2 + 0*64*2 + is*2 + 1]);
    exit(-1);
  }
  */

  /*
  if(tf == 1 ){
    for(int ico = 0 ; ico < 4 ; ico++)
      printf("%+e %+e\n",corr[0*4*2 + ico*2 + 0], corr[0*4*2 + ico*2 + 1]);
    exit(-1);
  }
  */
