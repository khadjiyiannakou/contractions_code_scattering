#include <contract.h>
#include <constants.h>
#include <device_opts_inline.h>
#include <utils.h>
#include <stdio.h>

using namespace contract;

// =========== Constant memory references ================//

__constant__ short int c_spinIndices_rho[768][4]; // 6 Kb
__constant__ short int c_lp_rp_rho[768][2]; // 3 Kb
__constant__ float c_coef_rho[768][2]; // 12 Kb
                                       // 21 Kb in total 
// ======================================================//

bool isConstantRhoOn = false;


static void copy_constants_rho(){

  cudaMemcpyToSymbol(c_spinIndices_rho, spinIndices_rho, 768*4*sizeof(short int));
  cudaMemcpyToSymbol(c_lp_rp_rho, lp_rp_rho, 768*2*sizeof(short int));
  cudaMemcpyToSymbol(c_coef_rho, coef_rho, 768*2*sizeof(float) );
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

__global__ void calculate_rho_kernel_float(float2* out, cudaTextureObject_t texProp, cudaTextureObject_t texMom, int ti, int tf, float2* tmp1, float2* tmp2){
#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2

#include <calculate_rho_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2
}

//==================================================//
__global__ void calculate_rho_kernel_double(double2* out, cudaTextureObject_t texProp, cudaTextureObject_t texMom, int ti, int tf, double2* tmp1, double2* tmp2){
#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2


#include <calculate_rho_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2
}
//==================================================//

template<typename Float2, typename Float> 
static void calculate_rho_kernel(cudaTextureObject_t texProp, cudaTextureObject_t texMom, int tf,int  Nt, Float* corr){
  if(!isConstantRhoOn)
    ABORT("Error: You need to initialize device constants before calling Kernels\n");
  
  // 4 * 4 ==> these are the different combinations one can constuct using 4 different interpolating fields for the rho
  // 3 is for the three spatial directions on the lattice
  // 16 are the non zero combinations when one does the spin indices contractions
  int numBlocks = 4 * 4 * 3 * 16; // these are the different contractions we need to calculate for the rho-rho
  dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE,1); // 625 threads
  dim3 gridDim(numBlocks,1,1); // 768 blocks

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
      calculate_rho_kernel_float<<<gridDim,blockDim>>>((float2*) d_partial_block, texProp, texMom, ti, tf, (float2*) tmp1, (float2*) tmp2);
    else if( typeid(Float2) == typeid(double2) )
      calculate_rho_kernel_double<<<gridDim,blockDim>>>((double2*) d_partial_block, texProp, texMom, ti, tf, (double2*) tmp1, (double2*) tmp2);
    else
      ABORT("Something fishy is happening\n");

  }

  cudaMemcpy(h_partial_block, d_partial_block, Nt*numBlocks*2*sizeof(Float), cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR();
  
  memset(corr,0,Nt*4*4*3*2*sizeof(Float));

  for(int ti = 0 ; ti < Nt ; ti++)
    for(int ico = 0 ; ico < 16 ; ico++)
      for(int igi = 0 ; igi < 3 ; igi++)
	for(int is = 0 ; is < 16 ; is++){
	  corr[ti*16*3*2 + ico*3*2 + igi*2 + 0] += h_partial_block[ti*16*3*16*2 + ico*3*16*2 + igi*16*2 + is*2 + 0]; 
	  corr[ti*16*3*2 + ico*3*2 + igi*2 + 1] += h_partial_block[ti*16*3*16*2 + ico*3*16*2 + igi*16*2 + is*2 + 1]; 
	}

  // clean memory
  free(h_partial_block);
  cudaFree(d_partial_block);
  cudaFree(tmp1);
  cudaFree(tmp2);
  CHECK_CUDA_ERROR();
}

void contract::run_ContractRho(cudaTextureObject_t texProp, cudaTextureObject_t texMom, int tf,int Nt,  void* corr, PRECISION prec ){

  if(prec == SINGLE){
    calculate_rho_kernel<float2,float>(texProp, texMom, tf, Nt, (float*) corr);
  }
  else if (prec == DOUBLE){
    calculate_rho_kernel<double2,double>(texProp, texMom, tf, Nt, (double*) corr);
  }
  else{
    ABORT("Error: this precision in not implemented");
  }

}

void contract::run_CopyConstantsRho(){
  if(isConstantRhoOn){
    WARNING("Warning: Copy constants for rho again will be skipped\n");
    return;
  }
  copy_constants_rho();
  isConstantRhoOn = true;
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
