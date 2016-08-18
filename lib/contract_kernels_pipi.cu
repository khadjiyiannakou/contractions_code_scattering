#include <contract.h>
#include <constants.h>
#include <device_opts_inline.h>
#include <utils.h>
#include <stdio.h>

using namespace contract;


// =========== Constant memory references ================//
__constant__ short int c_spinIndices_pipi_square[256][6]; // 3 Kb
__constant__ float c_coef_pipi_square[256][2];              // 2 Kb

__constant__ short int c_spinIndices_pipi_doubleTriangle[256][6]; // 3 Kb
__constant__ float c_coef_pipi_doubleTriangle[256][2];              // 2 Kb


__constant__ short int c_spinIndices_pipi_doubleTriangle_hor[256][4]; // 2Kb
__constant__ float c_coef_pipi_doubleTriangle_hor[256][2];            // 2Kb
                                                                  // 14 Kb total
// ======================================================//

bool isConstantPiPiPiPiOn = false;

static void copy_constants_pipi(){

  cudaMemcpyToSymbol(c_spinIndices_pipi_square, spinIndices_pipi_square, 256*6*sizeof(short int));
  cudaMemcpyToSymbol(c_spinIndices_pipi_doubleTriangle, spinIndices_pipi_doubleTriangle, 256*6*sizeof(short int));
  cudaMemcpyToSymbol(c_spinIndices_pipi_doubleTriangle_hor, spinIndices_pipi_doubleTriangle_hor, 256*4*sizeof(short int));

  cudaMemcpyToSymbol(c_coef_pipi_square, coef_pipi_square, 256*2*sizeof(float));
  cudaMemcpyToSymbol(c_coef_pipi_doubleTriangle, coef_pipi_doubleTriangle, 256*2*sizeof(float));
  cudaMemcpyToSymbol(c_coef_pipi_doubleTriangle_hor, coef_pipi_doubleTriangle_hor, 256*2*sizeof(float));
  
  CHECK_CUDA_ERROR();
}

//=======================================================//
// !!!!!!! for now the code will work only with 100 eigenVectors
// !!!!!!! for now the code will work only with submatrix side 25 ==> 25x25=625 threads
#define BLOCK_SIZE 25
#define NSIZE 100

//=====================================================//

//=====================================================//
__global__ void calculate_pipi_doubleTriangleHor_kernel_float(float2* out, cudaTextureObject_t texProp, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, float2* tmp1, float2* tmp2){

#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2

#include <calculate_pipi_doubleTriangleHor_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2

}
//==================================================//

//=====================================================//
__global__ void calculate_pipi_doubleTriangleHor_kernel_double(double2* out, cudaTextureObject_t texProp, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, double2* tmp1, double2* tmp2){

#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2

#include <calculate_pipi_doubleTriangleHor_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2

}
//==================================================//

//=====================================================//
__global__ void calculate_pipi_square_kernel_float(float2* out, cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, float2* tmp1, float2* tmp2){

#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2

#include <calculate_pipi_square_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2

}
//==================================================//

//=====================================================//
__global__ void calculate_pipi_square_kernel_double(double2* out, cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, double2* tmp1, double2* tmp2){

#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2

#include <calculate_pipi_square_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2

}

//=====================================================//
__global__ void calculate_pipi_doubleTriangle_kernel_float(float2* out, cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, float2* tmp1, float2* tmp2){

#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2

#include <calculate_pipi_doubleTriangle_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2

}
//==================================================//

//=====================================================//
__global__ void calculate_pipi_doubleTriangle_kernel_double(double2* out, cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, double2* tmp1, double2* tmp2){

#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2

#include <calculate_pipi_doubleTriangle_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2

}
// =================================================//
__global__ void calculate_pipi_starfish_kernel_float(float2* out, cudaTextureObject_t texProp, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, float2* tmp1, float2* tmp2){

#define FLOAT2 float2
#define FLOAT float
#define FETCH_FLOAT2 fetch_float2

#include <calculate_pipi_starfish_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2

}
//==================================================//
// =================================================//
__global__ void calculate_pipi_starfish_kernel_double(double2* out, cudaTextureObject_t texProp, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, double2* tmp1, double2* tmp2){

#define FLOAT2 double2
#define FLOAT double
#define FETCH_FLOAT2 fetch_double2

#include <calculate_pipi_starfish_core.h>

#undef FLOAT2
#undef FLOAT
#undef FETCH_FLOAT2

}
//==================================================//

template<typename Float2, typename Float>
static void calculate_pipi_kernel(cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, int Nt, Float* corr){

  if(!isConstantPiPiPiPiOn)
    ABORT("Error: You need to initialize device constants before calling Kernels\n");

  int numBlocks_square = Nt * 256;           // 256 non-zero spin combinations
  int numBlocks_doubleTriangle = Nt * 256;
  int numBlocks_star = Nt * 16;             // 16 non-zero spin combinations
  int numBlocks_fish = Nt * 16;

  dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE,1); // 625 threads
  dim3 gridDim_square(numBlocks_square,1,1);
  dim3 gridDim_doubleTriangle(numBlocks_doubleTriangle,1,1);
  dim3 gridDim_star(numBlocks_star,1,1);
  dim3 gridDim_fish(numBlocks_fish,1,1);

  Float *h_square = NULL;
  Float *h_doubleTriangle = NULL;
  Float *h_star = NULL;
  Float *h_fish = NULL;

  h_square = (Float*) malloc(numBlocks_square*2*sizeof(Float));
  h_doubleTriangle = (Float*) malloc(numBlocks_doubleTriangle*2*sizeof(Float));
  h_star = (Float*) malloc(numBlocks_star*2*2*sizeof(Float)); // two traces to store
  h_fish = (Float*) malloc(numBlocks_fish*2*2*sizeof(Float)); // two traces to store
  if(h_square == NULL || h_doubleTriangle == NULL || h_star == NULL || h_fish == NULL)
    ABORT("Error allocating memory\n");

  Float *d_square = NULL;
  Float *d_doubleTriangle = NULL;
  Float *d_star = NULL;
  Float *d_fish = NULL;
  
  cudaMalloc((void**)&d_square, numBlocks_square*2*sizeof(Float));
  cudaMalloc((void**)&d_doubleTriangle, numBlocks_doubleTriangle*2*sizeof(Float));
  cudaMalloc((void**)&d_star, numBlocks_star*2*2*sizeof(Float));
  cudaMalloc((void**)&d_fish, numBlocks_fish*2*2*sizeof(Float));
  CHECK_CUDA_ERROR();

  Float *tmp1 = NULL;
  Float *tmp2 = NULL;
  
  cudaMalloc((void**)&tmp1, numBlocks_square*NSIZE*NSIZE*2*sizeof(Float));
  CHECK_CUDA_ERROR();
  cudaMalloc((void**)&tmp2, numBlocks_square*NSIZE*NSIZE*2*sizeof(Float));
  CHECK_CUDA_ERROR();

  //++++
  if( typeid(Float2) == typeid(float2) ){
    calculate_pipi_square_kernel_float<<<gridDim_square,blockDim>>>((float2*) d_square, texProp, texPropDiag, texMomP1, texMomP2, texMomP3, texMomP4, tf, (float2*) tmp1, (float2*) tmp2);
    calculate_pipi_doubleTriangle_kernel_float<<<gridDim_doubleTriangle,blockDim>>>((float2*) d_doubleTriangle, texProp, texPropDiag, texMomP1, texMomP2, texMomP3, texMomP4, tf, (float2*) tmp1, (float2*) tmp2);
    calculate_pipi_starfish_kernel_float<<<gridDim_star,blockDim>>>((float2*) d_star, texProp, texMomP1, texMomP2, texMomP3, texMomP4, tf, (float2*) tmp1, (float2*) tmp2);
    calculate_pipi_starfish_kernel_float<<<gridDim_fish,blockDim>>>((float2*) d_fish, texProp, texMomP1, texMomP2, texMomP4, texMomP3, tf, (float2*) tmp1, (float2*) tmp2);
  }
  else if ( typeid(Float2) == typeid(double2) ){
    calculate_pipi_square_kernel_double<<<gridDim_square,blockDim>>>((double2*) d_square, texProp, texPropDiag, texMomP1, texMomP2, texMomP3, texMomP4, tf, (double2*) tmp1, (double2*) tmp2);
    calculate_pipi_doubleTriangle_kernel_double<<<gridDim_doubleTriangle,blockDim>>>((double2*) d_doubleTriangle, texProp, texPropDiag, texMomP1, texMomP2, texMomP3, texMomP4, tf, (double2*) tmp1, (double2*) tmp2);
    calculate_pipi_starfish_kernel_double<<<gridDim_star,blockDim>>>((double2*) d_star, texProp, texMomP1, texMomP2, texMomP3, texMomP4, tf, (double2*) tmp1, (double2*) tmp2);
    calculate_pipi_starfish_kernel_double<<<gridDim_fish,blockDim>>>((double2*) d_fish, texProp, texMomP1, texMomP2, texMomP4, texMomP3, tf, (double2*) tmp1, (double2*) tmp2);
  }
  else
    ABORT("Something fishy is happening\n");
  //++++
  cudaMemcpy(h_square, d_square, numBlocks_square*2*sizeof(Float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_doubleTriangle, d_doubleTriangle, numBlocks_doubleTriangle*2*sizeof(Float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_star, d_star, numBlocks_star*2*2*sizeof(Float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fish, d_fish, numBlocks_fish*2*2*sizeof(Float), cudaMemcpyDeviceToHost);

  CHECK_CUDA_ERROR();

  Float *h_square_reduce = (Float*) calloc(Nt*2,sizeof(Float));
  Float *h_doubleTriangle_reduce = (Float*) calloc(Nt*2,sizeof(Float));
  if(h_square_reduce == NULL || h_doubleTriangle_reduce == NULL)ABORT("Error allocating memory for reduction\n");

  for(int ti = 0 ; ti < Nt ; ti++)
    for(int is = 0 ; is < 256 ; is++){
      h_square_reduce[ti*2 + 0] += h_square[ti*256*2 + is*2 + 0];
      h_square_reduce[ti*2 + 1] += h_square[ti*256*2 + is*2 + 1];

      h_doubleTriangle_reduce[ti*2 + 0] += h_doubleTriangle[ti*256*2 + is*2 + 0];
      h_doubleTriangle_reduce[ti*2 + 1] += h_doubleTriangle[ti*256*2 + is*2 + 1];
    }

  Float *h_star_reduce = (Float*) calloc(2*Nt*2,sizeof(Float));
  Float *h_fish_reduce = (Float*) calloc(2*Nt*2,sizeof(Float));
  if(h_star_reduce == NULL || h_fish_reduce == NULL) ABORT("Error allocating memory for reduction\n");

  Float *h_star_trtr = (Float*) calloc(Nt*2,sizeof(Float));
  Float *h_fish_trtr = (Float*) calloc(Nt*2,sizeof(Float));

  for(int ti = 0 ; ti < Nt ; ti++)
    for(int is = 0 ; is < 16 ; is++)
      for(int tr = 0 ; tr < 2 ; tr++){
	h_star_reduce[tr*Nt*2 + ti*2 + 0] += h_star[tr*Nt*16*2 + ti*16*2 + is*2 + 0];
	h_star_reduce[tr*Nt*2 + ti*2 + 1] += h_star[tr*Nt*16*2 + ti*16*2 + is*2 + 1];

	h_fish_reduce[tr*Nt*2 + ti*2 + 0] += h_fish[tr*Nt*16*2 + ti*16*2 + is*2 + 0];
	h_fish_reduce[tr*Nt*2 + ti*2 + 1] += h_fish[tr*Nt*16*2 + ti*16*2 + is*2 + 1];
    }

  for(int ti = 0 ; ti < Nt ; ti++){
    h_star_trtr[ti*2+0] = h_star_reduce[0*Nt*2 + ti*2 + 0]*h_star_reduce[1*Nt*2 + ti*2 + 0] - h_star_reduce[0*Nt*2 + ti*2 + 1]*h_star_reduce[1*Nt*2 + ti*2 + 1];
    h_star_trtr[ti*2+1] = h_star_reduce[0*Nt*2 + ti*2 + 0]*h_star_reduce[1*Nt*2 + ti*2 + 1] + h_star_reduce[0*Nt*2 + ti*2 + 1]*h_star_reduce[1*Nt*2 + ti*2 + 0];

    h_fish_trtr[ti*2+0] = h_fish_reduce[0*Nt*2 + ti*2 + 0]*h_fish_reduce[1*Nt*2 + ti*2 + 0] - h_fish_reduce[0*Nt*2 + ti*2 + 1]*h_fish_reduce[1*Nt*2 + ti*2 + 1];
    h_fish_trtr[ti*2+1] = h_fish_reduce[0*Nt*2 + ti*2 + 0]*h_fish_reduce[1*Nt*2 + ti*2 + 1] + h_fish_reduce[0*Nt*2 + ti*2 + 1]*h_fish_reduce[1*Nt*2 + ti*2 + 0];
  }

  memset(corr, 0, Nt*5*2*sizeof(Float)); // 5 because we have 5 diagrams

  /*
  for(int ti = 0 ; ti < Nt ; ti++){
    corr[ti*2+0] = - (2.*h_square_reduce[ti*2 + 0] - 2.*h_doubleTriangle_reduce[ti*2 + 0] + h_star_trtr[ti*2 + 0] - h_fish_trtr[ti*2 + 0]);
    corr[ti*2+1] = - (2.*h_square_reduce[ti*2 + 1] - 2.*h_doubleTriangle_reduce[ti*2 + 1] + h_star_trtr[ti*2 + 1] - h_fish_trtr[ti*2 + 1]);
  }
  */

  for(int ti = 0 ; ti < Nt ; ti++){
    corr[ti*5*2 + 0*2 +0] = -2.*h_square_reduce[ti*2 + 0];
    corr[ti*5*2 + 0*2 +1] = -2.*h_square_reduce[ti*2 + 1];

    corr[ti*5*2 + 1*2 +0] = 2.*h_doubleTriangle_reduce[ti*2 + 0];
    corr[ti*5*2 + 1*2 +1] = 2.*h_doubleTriangle_reduce[ti*2 + 1];

    corr[ti*5*2 + 2*2 +0] = -h_star_trtr[ti*2 + 0];
    corr[ti*5*2 + 2*2 +1] = -h_star_trtr[ti*2 + 1];

    corr[ti*5*2 + 3*2 +0] = h_fish_trtr[ti*2 + 0];
    corr[ti*5*2 + 3*2 +1] = h_fish_trtr[ti*2 + 1];

    corr[ti*5*2 + 4*2 +0] = 0.; // for I=1 there are only 4 diagrams
    corr[ti*5*2 + 4*2 +1] = 0.;
  }


  free(h_star_trtr);
  free(h_fish_trtr);
  free(h_star_reduce);
  free(h_fish_reduce);
  free(h_square_reduce);
  free(h_doubleTriangle_reduce);
  cudaFree(tmp1);
  cudaFree(tmp2);
  cudaFree(d_square);
  cudaFree(d_doubleTriangle);
  cudaFree(d_star);
  cudaFree(d_fish);
  CHECK_CUDA_ERROR();
  free(h_square);
  free(h_doubleTriangle);
  free(h_star);
  free(h_fish);
}

//=========================================================//

template<typename Float2, typename Float>
static void calculate_pipi_kernel_I0(cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, int Nt, Float* corr){

  if(!isConstantPiPiPiPiOn)
    ABORT("Error: You need to initialize device constants before calling Kernels\n");

  int numBlocks_square = Nt * 256;           // 256 non-zero spin combinations
  int numBlocks_doubleTriangle = Nt * 256;
  int numBlocks_doubleTriangle_hor = Nt * 256;
  int numBlocks_star = Nt * 16;             // 16 non-zero spin combinations
  int numBlocks_fish = Nt * 16;

  dim3 blockDim(BLOCK_SIZE,BLOCK_SIZE,1); // 625 threads
  dim3 gridDim_square(numBlocks_square,1,1);
  dim3 gridDim_doubleTriangle(numBlocks_doubleTriangle,1,1);
  dim3 gridDim_doubleTriangle_hor(numBlocks_doubleTriangle_hor,1,1);

  dim3 gridDim_star(numBlocks_star,1,1);
  dim3 gridDim_fish(numBlocks_fish,1,1);

  Float *h_square = NULL;
  Float *h_doubleTriangle = NULL;
  Float *h_doubleTriangle_hor = NULL;
  Float *h_star = NULL;
  Float *h_fish = NULL;

  h_square = (Float*) malloc(numBlocks_square*2*sizeof(Float));
  h_doubleTriangle = (Float*) malloc(numBlocks_doubleTriangle*2*sizeof(Float));
  h_doubleTriangle_hor = (Float*) malloc(numBlocks_doubleTriangle_hor*2*sizeof(Float));

  h_star = (Float*) malloc(numBlocks_star*2*2*sizeof(Float)); // two traces to store
  h_fish = (Float*) malloc(numBlocks_fish*2*2*sizeof(Float)); // two traces to store
  if(h_square == NULL || h_doubleTriangle == NULL || h_doubleTriangle_hor == NULL || h_star == NULL || h_fish == NULL)
    ABORT("Error allocating memory\n");

  Float *d_square = NULL;
  Float *d_doubleTriangle = NULL;
  Float *d_doubleTriangle_hor = NULL;
  Float *d_star = NULL;
  Float *d_fish = NULL;
  
  cudaMalloc((void**)&d_square, numBlocks_square*2*sizeof(Float));
  cudaMalloc((void**)&d_doubleTriangle, numBlocks_doubleTriangle*2*sizeof(Float));
  cudaMalloc((void**)&d_doubleTriangle_hor, numBlocks_doubleTriangle_hor*2*sizeof(Float));

  cudaMalloc((void**)&d_star, numBlocks_star*2*2*sizeof(Float));
  cudaMalloc((void**)&d_fish, numBlocks_fish*2*2*sizeof(Float));
  CHECK_CUDA_ERROR();

  Float *tmp1 = NULL;
  Float *tmp2 = NULL;
  
  cudaMalloc((void**)&tmp1, numBlocks_square*NSIZE*NSIZE*2*sizeof(Float));
  CHECK_CUDA_ERROR();
  cudaMalloc((void**)&tmp2, numBlocks_square*NSIZE*NSIZE*2*sizeof(Float));
  CHECK_CUDA_ERROR();

  //++++
  if( typeid(Float2) == typeid(float2) ){
    calculate_pipi_square_kernel_float<<<gridDim_square,blockDim>>>((float2*) d_square, texProp, texPropDiag, texMomP1, texMomP2, texMomP3, texMomP4, tf, (float2*) tmp1, (float2*) tmp2);
    calculate_pipi_doubleTriangle_kernel_float<<<gridDim_doubleTriangle,blockDim>>>((float2*) d_doubleTriangle, texProp, texPropDiag, texMomP1, texMomP2, texMomP3, texMomP4, tf, (float2*) tmp1, (float2*) tmp2);
    calculate_pipi_doubleTriangleHor_kernel_float<<<gridDim_doubleTriangle_hor,blockDim>>>((float2*) d_doubleTriangle_hor, texProp, texMomP1, texMomP2, texMomP3, texMomP4, tf, (float2*) tmp1, (float2*) tmp2);
    calculate_pipi_starfish_kernel_float<<<gridDim_star,blockDim>>>((float2*) d_star, texProp, texMomP1, texMomP2, texMomP3, texMomP4, tf, (float2*) tmp1, (float2*) tmp2);
    calculate_pipi_starfish_kernel_float<<<gridDim_fish,blockDim>>>((float2*) d_fish, texProp, texMomP1, texMomP2, texMomP4, texMomP3, tf, (float2*) tmp1, (float2*) tmp2);
  }
  else if ( typeid(Float2) == typeid(double2) ){
    calculate_pipi_square_kernel_double<<<gridDim_square,blockDim>>>((double2*) d_square, texProp, texPropDiag, texMomP1, texMomP2, texMomP3, texMomP4, tf, (double2*) tmp1, (double2*) tmp2);
    calculate_pipi_doubleTriangle_kernel_double<<<gridDim_doubleTriangle,blockDim>>>((double2*) d_doubleTriangle, texProp, texPropDiag, texMomP1, texMomP2, texMomP3, texMomP4, tf, (double2*) tmp1, (double2*) tmp2);
    calculate_pipi_doubleTriangleHor_kernel_double<<<gridDim_doubleTriangle_hor,blockDim>>>((double2*) d_doubleTriangle_hor, texProp, texMomP1, texMomP2, texMomP3, texMomP4, tf, (double2*) tmp1, (double2*) tmp2);
    calculate_pipi_starfish_kernel_double<<<gridDim_star,blockDim>>>((double2*) d_star, texProp, texMomP1, texMomP2, texMomP3, texMomP4, tf, (double2*) tmp1, (double2*) tmp2);
    calculate_pipi_starfish_kernel_double<<<gridDim_fish,blockDim>>>((double2*) d_fish, texProp, texMomP1, texMomP2, texMomP4, texMomP3, tf, (double2*) tmp1, (double2*) tmp2);
  }
  else
    ABORT("Something fishy is happening\n");
  //++++
  cudaMemcpy(h_square, d_square, numBlocks_square*2*sizeof(Float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_doubleTriangle, d_doubleTriangle, numBlocks_doubleTriangle*2*sizeof(Float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_doubleTriangle_hor, d_doubleTriangle_hor, numBlocks_doubleTriangle_hor*2*sizeof(Float), cudaMemcpyDeviceToHost);

  cudaMemcpy(h_star, d_star, numBlocks_star*2*2*sizeof(Float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_fish, d_fish, numBlocks_fish*2*2*sizeof(Float), cudaMemcpyDeviceToHost);

  CHECK_CUDA_ERROR();

  Float *h_square_reduce = (Float*) calloc(Nt*2,sizeof(Float));
  Float *h_doubleTriangle_reduce = (Float*) calloc(Nt*2,sizeof(Float));
  Float *h_doubleTriangle_hor_reduce = (Float*) calloc(Nt*2,sizeof(Float));
  
  if(h_square_reduce == NULL || h_doubleTriangle_reduce == NULL || h_doubleTriangle_hor_reduce == NULL)ABORT("Error allocating memory for reduction\n");

  for(int ti = 0 ; ti < Nt ; ti++)
    for(int is = 0 ; is < 256 ; is++){
      h_square_reduce[ti*2 + 0] += h_square[ti*256*2 + is*2 + 0];
      h_square_reduce[ti*2 + 1] += h_square[ti*256*2 + is*2 + 1];

      h_doubleTriangle_reduce[ti*2 + 0] += h_doubleTriangle[ti*256*2 + is*2 + 0];
      h_doubleTriangle_reduce[ti*2 + 1] += h_doubleTriangle[ti*256*2 + is*2 + 1];

      h_doubleTriangle_hor_reduce[ti*2 + 0] += h_doubleTriangle_hor[ti*256*2 + is*2 + 0];
      h_doubleTriangle_hor_reduce[ti*2 + 1] += h_doubleTriangle_hor[ti*256*2 + is*2 + 1];
    }

  Float *h_star_reduce = (Float*) calloc(2*Nt*2,sizeof(Float));
  Float *h_fish_reduce = (Float*) calloc(2*Nt*2,sizeof(Float));
  if(h_star_reduce == NULL || h_fish_reduce == NULL) ABORT("Error allocating memory for reduction\n");

  Float *h_star_trtr = (Float*) calloc(Nt*2,sizeof(Float));
  Float *h_fish_trtr = (Float*) calloc(Nt*2,sizeof(Float));

  for(int ti = 0 ; ti < Nt ; ti++)
    for(int is = 0 ; is < 16 ; is++)
      for(int tr = 0 ; tr < 2 ; tr++){
	h_star_reduce[tr*Nt*2 + ti*2 + 0] += h_star[tr*Nt*16*2 + ti*16*2 + is*2 + 0];
	h_star_reduce[tr*Nt*2 + ti*2 + 1] += h_star[tr*Nt*16*2 + ti*16*2 + is*2 + 1];

	h_fish_reduce[tr*Nt*2 + ti*2 + 0] += h_fish[tr*Nt*16*2 + ti*16*2 + is*2 + 0];
	h_fish_reduce[tr*Nt*2 + ti*2 + 1] += h_fish[tr*Nt*16*2 + ti*16*2 + is*2 + 1];
    }

  for(int ti = 0 ; ti < Nt ; ti++){
    h_star_trtr[ti*2+0] = h_star_reduce[0*Nt*2 + ti*2 + 0]*h_star_reduce[1*Nt*2 + ti*2 + 0] - h_star_reduce[0*Nt*2 + ti*2 + 1]*h_star_reduce[1*Nt*2 + ti*2 + 1];
    h_star_trtr[ti*2+1] = h_star_reduce[0*Nt*2 + ti*2 + 0]*h_star_reduce[1*Nt*2 + ti*2 + 1] + h_star_reduce[0*Nt*2 + ti*2 + 1]*h_star_reduce[1*Nt*2 + ti*2 + 0];

    h_fish_trtr[ti*2+0] = h_fish_reduce[0*Nt*2 + ti*2 + 0]*h_fish_reduce[1*Nt*2 + ti*2 + 0] - h_fish_reduce[0*Nt*2 + ti*2 + 1]*h_fish_reduce[1*Nt*2 + ti*2 + 1];
    h_fish_trtr[ti*2+1] = h_fish_reduce[0*Nt*2 + ti*2 + 0]*h_fish_reduce[1*Nt*2 + ti*2 + 1] + h_fish_reduce[0*Nt*2 + ti*2 + 1]*h_fish_reduce[1*Nt*2 + ti*2 + 0];
  }

  memset(corr, 0, Nt*5*2*sizeof(Float));

  /*
  for(int ti = 0 ; ti < Nt ; ti++){
    corr[ti*2+0] =  (-1./3.)*h_square_reduce[ti*2 + 0] - (1./3.)*h_doubleTriangle_reduce[ti*2 + 0] - (5./3.)*h_doubleTriangle_hor_reduce[ti*2 + 0] + h_star_trtr[ti*2 + 0] + h_fish_trtr[ti*2 + 0];
    corr[ti*2+1] =  (-1./3.)*h_square_reduce[ti*2 + 1] - (1./3.)*h_doubleTriangle_reduce[ti*2 + 1] - (5./3.)*h_doubleTriangle_hor_reduce[ti*2 + 1] + h_star_trtr[ti*2 + 1] + h_fish_trtr[ti*2 + 1];
  }
  */

  for(int ti = 0 ; ti < Nt ; ti++){
    corr[ti*5*2 + 0*2 +0] = (-1./3.)*h_square_reduce[ti*2 + 0];
    corr[ti*5*2 + 0*2 +1] = (-1./3.)*h_square_reduce[ti*2 + 1];

    corr[ti*5*2 + 1*2 +0] = -(1./3.)*h_doubleTriangle_reduce[ti*2 + 0];
    corr[ti*5*2 + 1*2 +1] = -(1./3.)*h_doubleTriangle_reduce[ti*2 + 1];

    corr[ti*5*2 + 2*2 +0] = h_star_trtr[ti*2 + 0];
    corr[ti*5*2 + 2*2 +1] = h_star_trtr[ti*2 + 1];

    corr[ti*5*2 + 3*2 +0] = h_fish_trtr[ti*2 + 0];
    corr[ti*5*2 + 3*2 +1] = h_fish_trtr[ti*2 + 1];

    corr[ti*5*2 + 4*2 +0] = -(5./3.)*h_doubleTriangle_hor_reduce[ti*2 + 0];
    corr[ti*5*2 + 4*2 +1] = -(5./3.)*h_doubleTriangle_hor_reduce[ti*2 + 1];
  }



  free(h_star_trtr);
  free(h_fish_trtr);
  free(h_star_reduce);
  free(h_fish_reduce);
  free(h_square_reduce);
  free(h_doubleTriangle_reduce);
  free(h_doubleTriangle_hor_reduce);
  cudaFree(tmp1);
  cudaFree(tmp2);
  cudaFree(d_square);
  cudaFree(d_doubleTriangle);
  cudaFree(d_doubleTriangle_hor);
  cudaFree(d_star);
  cudaFree(d_fish);
  CHECK_CUDA_ERROR();
  free(h_square);
  free(h_doubleTriangle);
  free(h_doubleTriangle_hor);
  free(h_star);
  free(h_fish);
}



//===================================================//
void contract::run_ContractPiPi_I0(cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, int Nt, void* corr, PRECISION prec){

  if(prec == SINGLE){
    calculate_pipi_kernel_I0<float2,float>(texProp,texPropDiag,texMomP1,texMomP2,texMomP3,texMomP4,tf,Nt,(float*) corr);
  }
  else if (prec == DOUBLE){
    calculate_pipi_kernel_I0<double2,double>(texProp,texPropDiag,texMomP1,texMomP2,texMomP3,texMomP4,tf,Nt,(double*) corr);
  }
  else{
    ABORT("Error: this precision in not implemented");
  }  

}
//===================================================//
void contract::run_ContractPiPi(cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, int Nt, void* corr, PRECISION prec){

  if(prec == SINGLE){
    calculate_pipi_kernel<float2,float>(texProp,texPropDiag,texMomP1,texMomP2,texMomP3,texMomP4,tf,Nt,(float*) corr);
  }
  else if (prec == DOUBLE){
    calculate_pipi_kernel<double2,double>(texProp,texPropDiag,texMomP1,texMomP2,texMomP3,texMomP4,tf,Nt,(double*) corr);
  }
  else{
    ABORT("Error: this precision in not implemented");
  }

}
//===================================================//
void contract::run_CopyConstantsPiPi(){
  if(isConstantPiPiPiPiOn){
    WARNING("Warning: Copy constants for pi-pi again will be skipped\n");
    return;
  }
  copy_constants_pipi();
  isConstantPiPiPiPiOn = true;
}
//==================================================//
