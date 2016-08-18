#include <contract.h>
#include <constants.h>
#include <device_opts_inline.h>
#include <utils.h>
#include <stdio.h>

using namespace contract;

// =========== Constant memory references ================//

__constant__ short int c_spinIndices_sigma[64][4];
__constant__ short int c_lp_rp_sigma[64][2];
__constant__ float c_coef_sigma[64][2]; 
                                      
// ======================================================//

bool isConstantSigmaOn = false;


static void copy_constants_sigma(){

  cudaMemcpyToSymbol(c_spinIndices_sigma, spinIndices_sigma, 64*4*sizeof(short int));
  cudaMemcpyToSymbol(c_lp_rp_sigma, lp_rp_sigma, 64*2*sizeof(short int));
  cudaMemcpyToSymbol(c_coef_sigma, coef_sigma, 64*2*sizeof(float) );
  CHECK_CUDA_ERROR();

#ifdef DEVICE_DEBUG
  printf("Copy for rho constants to device finished\n");
#endif
  
}


// ==================================================== // 
// !!!!!!! for now the code will work only with 100 eigenVectors
// !!!!!!! for now the code will work only with submatrix side 25 ==> 25x25=625 threads
#define BLOCK_SIZE 25
#define NSIZE 100

//====================================================//

__global__ void calculate_sigma_one_fish_kernel_float(float2* out, cudaTextureObject_t texProp, cudaTextureObject_t texMom, int tf, float2* tmp1, float2* tmp2){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2

#include <calculate_sigma_one_fish_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2
}

//==================================================//
__global__ void calculate_sigma_one_fish_kernel_double(double2* out, cudaTextureObject_t texProp, cudaTextureObject_t texMom, int tf, double2* tmp1, double2* tmp2){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2


#include <calculate_sigma_one_fish_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2
}
//==================================================//

template<typename Float2, typename Float> 
static void calculate_sigma_one_fish_kernel(cudaTextureObject_t texProp, cudaTextureObject_t texMom, int tf,int  Nt, Float* corr){
  if(!isConstantSigmaOn)
    ABORT("Error: You need to initialize device constants before calling Kernels\n");
  
  int numBlocks = Nt * 2 * 2 * 16; // these are the different contractions we need to calculate for the sigma-sigma
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
    calculate_sigma_one_fish_kernel_float<<<gridDim,blockDim>>>((float2*) d_partial_block, texProp, texMom, tf, (float2*) tmp1, (float2*) tmp2);
  else if( typeid(Float2) == typeid(double2) )
    calculate_sigma_one_fish_kernel_double<<<gridDim,blockDim>>>((double2*) d_partial_block, texProp, texMom, tf, (double2*) tmp1, (double2*) tmp2);
  else
    ABORT("Something fishy is happening\n");


  cudaMemcpy(h_partial_block, d_partial_block, numBlocks*2*sizeof(Float), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR();

  //  for(int i = 0 ; i < 16 ; i++)
  //   printf("%+e %+e\n",h_partial_block[1*4*16*2 + 0*16*2 + i*2 + 0],h_partial_block[1*4*16*2 + 0*16*2 + i*2+1]);
  // exit(-1);
  
  memset(corr,0,Nt*2*2*2*sizeof(Float));

  for(int ti = 0 ; ti < Nt ; ti++)
    for(int ico = 0 ; ico < 4 ; ico++)
	for(int is = 0 ; is < 16 ; is++){
	  corr[ti*4*2 + ico*2 + 0] += h_partial_block[ti*4*16*2 + ico*16*2 + is*2 + 0]; 
	  corr[ti*4*2 + ico*2 + 1] += h_partial_block[ti*4*16*2 + ico*16*2 + is*2 + 1]; 
	}

  //  for(int i = 0 ; i < Nt ; i++)
  //   printf("\n %+e %+e\n",corr[i*4*2 + 0*2 +0],corr[i*4*2 + 0*2 +1]);

  // exit(-1);

  // clean memory
  free(h_partial_block);
  cudaFree(d_partial_block);
  cudaFree(tmp1);
  cudaFree(tmp2);
  CHECK_CUDA_ERROR();
}

void contract::run_ContractSigma(cudaTextureObject_t texProp, cudaTextureObject_t texMom, int tf,int Nt,  void* corr, PRECISION prec ){

  if(prec == SINGLE){
    calculate_sigma_one_fish_kernel<float2,float>(texProp, texMom, tf, Nt, (float*) corr);
  }
  else if (prec == DOUBLE){
    calculate_sigma_one_fish_kernel<double2,double>(texProp, texMom, tf, Nt, (double*) corr);
  }
  else{
    ABORT("Error: this precision in not implemented");
  }

}

void contract::run_CopyConstantsSigma(){
  if(isConstantSigmaOn){
    WARNING("Warning: Copy constants for sigma again will be skipped\n");
    return;
  }
  copy_constants_sigma();
  isConstantSigmaOn = true;
}


  /*
  for(int ico = 0 ; ico < 16 ; ico++)
    for(int igi = 0 ; igi < 3 ; igi++)
      printf("%+e %+e\n",corr[ico*3*2 + igi*2 + 0], corr[ico*3*2 + igi*2 + 1]);

  exit(-1);
  */


//   cudaDeviceSynchronize();
//    exit(-1);

/*
    if(ti != 1 || tf != 1 )
      continue;

    printf("kale %d %d\n",ti,tf);
*/

  /*
  for(int igi = 0 ; igi < 3 ; igi++)
    for(int is = 0 ; is < 16 ; is++){
      printf("%d %+e %+e\n",igi,h_partial_block[igi*16*2 + is*2 + 0], h_partial_block[igi*16*2 + is*2 + 1]);
    }

  exit(-1);
  */


  //  Float *tmp1_h = (Float*) malloc(numBlocks*NSIZE*NSIZE*2*sizeof(Float)); 
  // Float *tmp2_h = (Float*) malloc(numBlocks*NSIZE*NSIZE*2*sizeof(Float)); 

    /*
    cudaMemcpy(tmp1_h,tmp1,numBlocks*NSIZE*NSIZE*2*sizeof(Float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmp2_h,tmp2,numBlocks*NSIZE*NSIZE*2*sizeof(Float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    FILE *ptr_tmp1, *ptr_tmp2;
    ptr_tmp1 = fopen("tmp1.dat","w");
    ptr_tmp2 = fopen("tmp2.dat","w");

    for(int r = 0 ;r < 100; r++)
      for(int c = 0 ; c < 100 ; c++){
	fprintf(ptr_tmp1,"%+e %+e\n",tmp1_h[r*100*2 + c*2 + 0], tmp1_h[r*100*2 + c*2 + 1] );
	fprintf(ptr_tmp2,"%+e %+e\n",tmp2_h[r*100*2 + c*2 + 0], tmp2_h[r*100*2 + c*2 + 1]);
      }
    CHECK_CUDA_ERROR(); 

    exit(-1);

    */

  //  printf("%+e %+e\n",h_partial_block[0], h_partial_block[1]);
  // exit(-1);
