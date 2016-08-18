int bx = blockIdx.x;

int tx = threadIdx.x;
int ty = threadIdx.y;

short int s1,s2,s3,s5,s6;
short int ip;
FLOAT2 coef;

s1 = c_spinIndices_rhopi[idir*gridDim.x+bx][0];
s2 = c_spinIndices_rhopi[idir*gridDim.x+bx][1];
s3 = c_spinIndices_rhopi[idir*gridDim.x+bx][2];
//s4 = c_spinIndices_rhopi[idir*gridDim.x+bx][3];
s5 = c_spinIndices_rhopi[idir*gridDim.x+bx][4];
s6 = c_spinIndices_rhopi[idir*gridDim.x+bx][5];

ip = c_ip_rhopi[idir*gridDim.x+bx];

coef.x = (FLOAT) c_coef_rhopi[idir*gridDim.x+bx][0];
coef.y = (FLOAT) c_coef_rhopi[idir*gridDim.x+bx][1];

int Nxb = NSIZE / BLOCK_SIZE;
int Nyb = NSIZE / BLOCK_SIZE;

//int aBegin;
//int aEnd;
//int aStep;
//int bBegin;
//int bStep;
FLOAT2 Csub;
int offset_Prop;
int offset_MomMatrices;
int offset_MomMatrix;

__shared__ FLOAT2 As[BLOCK_SIZE][BLOCK_SIZE];
__shared__ FLOAT2 Bs[BLOCK_SIZE][BLOCK_SIZE];

//offset_MomMatrices = tf*NSIZE*NSIZE;
offset_MomMatrices = tf*5*NSIZE*NSIZE +  ip*NSIZE*NSIZE;
offset_Prop = ti*4*4*NSIZE*NSIZE +  s5*4*NSIZE*NSIZE + s6*NSIZE*NSIZE; 
// ==================================================================//
#pragma unroll
 for(int ixb = 0 ; ixb < Nxb ; ixb++)
#pragma unroll
   for(int iyb = 0 ; iyb < Nyb ; iyb++){
 
     int aBegin = NSIZE * BLOCK_SIZE * iyb;
     int aEnd   = aBegin + NSIZE - 1;
     int aStep  = BLOCK_SIZE;
     int bBegin = BLOCK_SIZE * ixb;
     int bStep  = BLOCK_SIZE * NSIZE;
     Csub.x=0.;
     Csub.y=0.;
     for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
       {

	 As[ty][tx] =  FETCH_FLOAT2(texMomP1P2, offset_MomMatrices + a + NSIZE * ty + tx);
	 Bs[ty][tx] =  FETCH_FLOAT2(texProp, offset_Prop + b + NSIZE * ty + tx);

	 __syncthreads();

#pragma unroll
	 for (int k = 0; k < BLOCK_SIZE; ++k)
	   {
	     Csub.x += As[ty][k].x * Bs[k][tx].x - As[ty][k].y * Bs[k][tx].y;
	     Csub.y += As[ty][k].x * Bs[k][tx].y + As[ty][k].y * Bs[k][tx].x;
	   }
	 __syncthreads();
       }

     int c = NSIZE * BLOCK_SIZE * iyb + BLOCK_SIZE * ixb;
     tmp1[bx*NSIZE*NSIZE + c + NSIZE * ty + tx] = Csub;
   } // close for loop running on submatrices
__syncthreads();
// ================================================= //

offset_MomMatrix = ti*NSIZE*NSIZE;
// ==================================================================//
#pragma unroll
 for(int ixb = 0 ; ixb < Nxb ; ixb++)
#pragma unroll
   for(int iyb = 0 ; iyb < Nyb ; iyb++){
 
     int aBegin = NSIZE * BLOCK_SIZE * iyb;
     int aEnd   = aBegin + NSIZE - 1;
     int aStep  = BLOCK_SIZE;
     int bBegin = BLOCK_SIZE * ixb;
     int bStep  = BLOCK_SIZE * NSIZE;
     Csub.x=0.;
     Csub.y=0.;
     for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
       {

	 As[ty][tx] = tmp1[bx*NSIZE*NSIZE + a + NSIZE * ty + tx];
	 Bs[ty][tx] =  FETCH_FLOAT2(texMomP2, offset_MomMatrix + b + NSIZE * ty + tx);

	 __syncthreads();

#pragma unroll
	 for (int k = 0; k < BLOCK_SIZE; ++k)
	   {
	     Csub.x += As[ty][k].x * Bs[k][tx].x - As[ty][k].y * Bs[k][tx].y;
	     Csub.y += As[ty][k].x * Bs[k][tx].y + As[ty][k].y * Bs[k][tx].x;
	   }
	 __syncthreads();
       }

     int c = NSIZE * BLOCK_SIZE * iyb + BLOCK_SIZE * ixb;
     tmp2[bx*NSIZE*NSIZE + c + NSIZE * ty + tx] = Csub;
   } // close for loop running on submatrices
__syncthreads();
// ================================================= //

offset_Prop = ti*4*4*NSIZE*NSIZE +  s1*4*NSIZE*NSIZE + s2*NSIZE*NSIZE;
// ==================================================================//
#pragma unroll
 for(int ixb = 0 ; ixb < Nxb ; ixb++)
#pragma unroll
   for(int iyb = 0 ; iyb < Nyb ; iyb++){
 
     int aBegin = NSIZE * BLOCK_SIZE * iyb;
     int aEnd   = aBegin + NSIZE - 1;
     int aStep  = BLOCK_SIZE;
     int bBegin = BLOCK_SIZE * ixb;
     int bStep  = BLOCK_SIZE * NSIZE;
     Csub.x=0.;
     Csub.y=0.;
     for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
       {

	 As[ty][tx] = tmp2[bx*NSIZE*NSIZE + a + NSIZE * ty + tx];
	 Bs[ty][tx] =  FETCH_FLOAT2(texPropDiag, offset_Prop + b + NSIZE * ty + tx);

	 __syncthreads();

#pragma unroll
	 for (int k = 0; k < BLOCK_SIZE; ++k)
	   {
	     Csub.x += As[ty][k].x * Bs[k][tx].x - As[ty][k].y * Bs[k][tx].y;
	     Csub.y += As[ty][k].x * Bs[k][tx].y + As[ty][k].y * Bs[k][tx].x;
	   }
	 __syncthreads();
       }

     int c = NSIZE * BLOCK_SIZE * iyb + BLOCK_SIZE * ixb;
     tmp1[bx*NSIZE*NSIZE + c + NSIZE * ty + tx] = Csub;
   } // close for loop running on submatrices
__syncthreads();
// ================================================= //

offset_MomMatrix = ti*NSIZE*NSIZE;
// ==================================================================//
#pragma unroll
 for(int ixb = 0 ; ixb < Nxb ; ixb++)
#pragma unroll
   for(int iyb = 0 ; iyb < Nyb ; iyb++){
 
     int aBegin = NSIZE * BLOCK_SIZE * iyb;
     int aEnd   = aBegin + NSIZE - 1;
     int aStep  = BLOCK_SIZE;
     int bBegin = BLOCK_SIZE * ixb;
     int bStep  = BLOCK_SIZE * NSIZE;
     Csub.x=0.;
     Csub.y=0.;
     for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
       {

	 As[ty][tx] = tmp1[bx*NSIZE*NSIZE + a + NSIZE * ty + tx];
	 Bs[ty][tx] =  FETCH_FLOAT2(texMomP1, offset_MomMatrix + b + NSIZE * ty + tx);

	 __syncthreads();

#pragma unroll
	 for (int k = 0; k < BLOCK_SIZE; ++k)
	   {
	     Csub.x += As[ty][k].x * Bs[k][tx].x - As[ty][k].y * Bs[k][tx].y;
	     Csub.y += As[ty][k].x * Bs[k][tx].y + As[ty][k].y * Bs[k][tx].x;
	   }
	 __syncthreads();
       }

     int c = NSIZE * BLOCK_SIZE * iyb + BLOCK_SIZE * ixb;
     tmp2[bx*NSIZE*NSIZE + c + NSIZE * ty + tx] = Csub;
   } // close for loop running on submatrices
__syncthreads();
// ===========================================================//

offset_Prop = ti*4*4*NSIZE*NSIZE +  s3*4*NSIZE*NSIZE + s2*NSIZE*NSIZE;

Csub.x=0.;
Csub.y=0.;
Bs[ty][tx].x = 0.;
Bs[ty][tx].y = 0.;
__syncthreads();

 for(int ixb = 0 ; ixb < Nxb ; ixb++)
   for(int iyb = 0 ; iyb < Nyb ; iyb++){
     int c = NSIZE * BLOCK_SIZE * iyb + BLOCK_SIZE * ixb;
     As[ty][tx] = tmp2[bx*NSIZE*NSIZE + c + NSIZE * ty + tx] * conj(FETCH_FLOAT2(texProp, offset_Prop + c + NSIZE * ty + tx));
     Bs[ty][tx].x = 0.;
     Bs[ty][tx].y = 0.;
     __syncthreads();

     // accumulation proccess
     if(ty == 0){
#pragma unroll
       for(int ii = 0 ; ii < BLOCK_SIZE ; ii++){
	 Bs[0][tx].x += As[ii][tx].x;
	 Bs[0][tx].y += As[ii][tx].y;
       }
     }
     __syncthreads();
     if(tx == 0 && ty == 0){
#pragma unroll
       for(int ii = 0 ; ii < BLOCK_SIZE ; ii++){
	 Csub.x += Bs[0][ii].x;
	 Csub.y += Bs[0][ii].y;
       }
     }
     __syncthreads();
   }

// only master thread writes the result
if(tx == 0 && ty == 0){
    out[ti*gridDim.x +  bx] = coef * Csub;
 }


/*
if(bx == 0 && ti == 1 && tf == 0 && tx == 0 && ty == 0){
  printf("(%d %d) , (%d,%d)\n",s1,s6,ti,tf);
  for(int i = 0 ; i < 100 ; i++){
    FLOAT2 temp = FETCH_FLOAT2(texProp, offset_Prop + 0 + NSIZE * 0 + i);
    printf("%+e %+e\n",temp.x, temp.y);
  }

  asm("trap;");
  }
*/
