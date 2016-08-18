int bx = blockIdx.x;

int tx = threadIdx.x;
int ty = threadIdx.y;


short int s1,s2,s3,s4;
short int lp,rp;
short int ti;
FLOAT2 coef;
short int numSpinsComb = 64;

s1=c_spinIndices_sigma[bx%numSpinsComb][0];
s2=c_spinIndices_sigma[bx%numSpinsComb][1];
s3=c_spinIndices_sigma[bx%numSpinsComb][2];
s4=c_spinIndices_sigma[bx%numSpinsComb][3];

ti = bx / numSpinsComb;

lp = c_lp_rp_sigma[bx%numSpinsComb][0];
rp = c_lp_rp_sigma[bx%numSpinsComb][1];

coef.x = (FLOAT) c_coef_sigma[bx%numSpinsComb][0];
coef.y = (FLOAT) c_coef_sigma[bx%numSpinsComb][1];

int Nxb = NSIZE / BLOCK_SIZE;
int Nyb = NSIZE / BLOCK_SIZE;

FLOAT2 Csub;
int offset_Prop;
int offset_Mom;

__shared__ FLOAT2 As[BLOCK_SIZE][BLOCK_SIZE];
__shared__ FLOAT2 Bs[BLOCK_SIZE][BLOCK_SIZE];



offset_Prop = ti*4*4*NSIZE*NSIZE +  s3*4*NSIZE*NSIZE + s4*NSIZE*NSIZE;
offset_Mom = ti*2*NSIZE*NSIZE + rp*NSIZE*NSIZE;
// ================================================== //
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

	 As[ty][tx] =  FETCH_FLOAT2(texProp, offset_Prop + a + NSIZE * ty + tx);
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
     tmp1[bx*NSIZE*NSIZE + c + NSIZE * ty + tx] = Csub;
   } // close for loop running on submatrices

//=========================================================//

offset_Prop = ti*4*4*NSIZE*NSIZE + s2*4*NSIZE*NSIZE + s1*NSIZE*NSIZE;
offset_Mom = tf*2*NSIZE*NSIZE +  lp*NSIZE*NSIZE;

// ================================================== //
 for(int ixb = 0 ; ixb < Nxb ; ixb++)
   for(int iyb = 0 ; iyb < Nyb ; iyb++){
     int     aBegin = NSIZE * BLOCK_SIZE * iyb;
     int aEnd   = aBegin + NSIZE - 1;
     int aStep  = BLOCK_SIZE;
     int bBegin = BLOCK_SIZE * ixb;
     int     bStep  = BLOCK_SIZE * NSIZE;
     Csub.x=0.;
     Csub.y=0.;
     for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
       {
	 As[ty][tx] = FETCH_FLOAT2(texMom, offset_Mom + a + NSIZE * ty + tx);
	 Bs[ty][tx] = FETCH_FLOAT2(texProp, offset_Prop + b + NSIZE * ty + tx);
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
// ================================================= //

Csub.x=0.;
Csub.y=0.;
Bs[ty][tx].x = 0.;
Bs[ty][tx].y = 0.;
__syncthreads();


 for(int ixb = 0 ; ixb < Nxb ; ixb++)
   for(int iyb = 0 ; iyb < Nyb ; iyb++){
     int c = NSIZE * BLOCK_SIZE * iyb + BLOCK_SIZE * ixb;
     As[ty][tx] = conj(tmp2[bx*NSIZE*NSIZE + c + NSIZE * ty + tx])*tmp1[bx*NSIZE*NSIZE + c + NSIZE * ty + tx];
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
    out[bx] = coef * Csub;
 }
