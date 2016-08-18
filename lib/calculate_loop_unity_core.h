int bx = blockIdx.x;

int tx = threadIdx.x;
int ty = threadIdx.y;

// total number of blocks needed is NT*4;
short int s1 = bx%4;
short int it = bx/4;

int Nxb = NSIZE / BLOCK_SIZE;
int Nyb = NSIZE / BLOCK_SIZE;

FLOAT2 Csub;
int offset_Prop;
int offset_Mom;

__shared__ FLOAT2 As[BLOCK_SIZE][BLOCK_SIZE];
__shared__ FLOAT2 Bs[BLOCK_SIZE][BLOCK_SIZE];

offset_Mom = it*NSIZE*NSIZE;
offset_Prop = it*4*4*NSIZE*NSIZE + s1*4*NSIZE*NSIZE + s1*NSIZE*NSIZE; // in the special case where we use unity we only need the diagonal elements in the spin subspace, We need texPropDiag for the loop

//Csub.x=0.;
//Csub.y=0.;

//====================================================================//
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

	 As[ty][tx] =  FETCH_FLOAT2(texPropDiag, offset_Prop + a + NSIZE * ty + tx);
	 Bs[ty][tx] =  FETCH_FLOAT2(texMom, offset_Mom + b + NSIZE * ty + tx);

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
     tmp[bx*NSIZE*NSIZE + c + NSIZE * ty + tx] = Csub;
   } // close for loop running on submatrices
__syncthreads();
//============================================================================//

Csub.x=0.;
Csub.y=0.;

if(tx == 0 && ty == 0){
#pragma unroll
  for(int ii = 0 ; ii < NSIZE ; ii++){
    Csub.x += tmp[bx*NSIZE*NSIZE + NSIZE * ii + ii].x;
    Csub.y += tmp[bx*NSIZE*NSIZE + NSIZE * ii + ii].y;
  }

  out[bx] = Csub;
 }



/*
for(int ixb = 0 ; ixb < Nxb ; ixb++)
  for(int iyb = 0 ; iyb < Nyb ; iyb++){
    int c = NSIZE * BLOCK_SIZE * iyb + BLOCK_SIZE * ixb;
    As[ty][tx] = FETCH_FLOAT2(texPropDiag, offset_Prop + c + NSIZE * ty + tx);


  }
*/
