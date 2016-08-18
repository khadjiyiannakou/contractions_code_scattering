#include <vector>
#include <stdio.h>
#include "smeared_matrix_object.h"
#include "matrix.h"
#include "comm/comm_low.h"
#include "options.h"
#include "layout_minsurface.h"
#include "qcd_fields.h"
#include "laplacean_eigensolver.h"
#include "smearing.h"
#include "boundary_flip.h"
#include "timer.h"
#include "lapmult.h"
#include "gamma_mult.h"
#include "inject.h"

using namespace qcd;

double_complex rhoj(int j, int i, int r, int c, bool dag=false){
  static matrix g[3] = { gamma_matrix.g1(), gamma_matrix.g2(), gamma_matrix.g3()};
  matrix res(4,4);


  switch(j){
  case 1:
    if(dag) 
      return -g[i](r,c);
    return g[i](r,c);
    break;
  case 2:
    res = gamma_matrix.g4()*g[i];
    if(dag) return res(r,c);
    return res(r,c);
    break;
  case 3:
    if(dag) return -g[i](r,c);
    return g[i](r,c);
    break;
  case 4:
    res = gamma_matrix.g4()*gamma_matrix.g4();
    if(dag) return -0.5*res(r,c);
    return 0.5*res(r,c);
    break;
  default:
    fprintf(stderr,"I found a mistake\n");
    exit(EXIT_FAILURE);
  }

}


void compute_lp_rp(int l , int k , int i , int *lp_rp){

  if(l==0 && k==0){lp_rp[0]=0;lp_rp[1]=0;}
  if(l==0 && k==1){lp_rp[0]=0;lp_rp[1]=0;}
  if(l==0 && k==2){lp_rp[0]=1;lp_rp[1]=0;}
  if(l==0 && k==3){lp_rp[0]=2+i;lp_rp[1]=0;}

  if(l==1 && k==0){lp_rp[0]=0;lp_rp[1]=0;}
  if(l==1 && k==1){lp_rp[0]=0;lp_rp[1]=0;}
  if(l==1 && k==2){lp_rp[0]=1;lp_rp[1]=0;}
  if(l==1 && k==3){lp_rp[0]=2+i;lp_rp[1]=0;}

  if(l==2 && k==0){lp_rp[0]=0;lp_rp[1]=1;}
  if(l==2 && k==1){lp_rp[0]=0;lp_rp[1]=1;}
  if(l==2 && k==2){lp_rp[0]=1;lp_rp[1]=1;}
  if(l==2 && k==3){lp_rp[0]=2+i;lp_rp[1]=1;}

  if(l==3 && k==0){lp_rp[0]=0;lp_rp[1]=2+i;}
  if(l==3 && k==1){lp_rp[0]=0;lp_rp[1]=2+i;}
  if(l==3 && k==2){lp_rp[0]=1;lp_rp[1]=2+i;}
  if(l==3 && k==3){lp_rp[0]=2+i;lp_rp[1]=2+i;}

}



int main(){

    matrix g5(4,4);
    g5 = gamma_matrix.g5();

    double_complex coef;
    int j;
    FILE *ptr_f;
    ptr_f=fopen("constants_rho.h","w");

    if(ptr_f == NULL){
      fprintf(stderr,"Error open file\n");
      exit(EXIT_FAILURE);
    }

    int lp_rp[2];
    int count = 0;
    //////////////////////////////////////////////////////////
    fprintf(ptr_f,"short int spinIndices[768][4] = { ");
    count = 0;
    for(int l = 0 ; l < 4 ; l++)
      for(int k = 0 ; k < 4 ; k++)
	for(int i = 0 ; i < 3 ; i++){
	  j=0;
	  for(int s1=0; s1<4; ++s1)
	    for(int s2=0; s2<4; ++s2)
	      for(int s3=0; s3<4; ++s3)
		for(int s4=0; s4<4; ++s4)
		  { 
		    
		    if(l*4+k < 12)
		      coef = -rhoj(k+1,i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rhoj(l+1,i, s1, s3);
		    else
		      coef = rhoj(k+1,i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rhoj(l+1,i, s1, s3);	      
		    if( abs(coef) != 0){
		      compute_lp_rp(l,k,i,lp_rp);
		      //		      printf("l(%d)-k(%d)-i(%d)-j(%d)-(%d,%d,%d,%d) \t %d %d \t (%+f,%+f)\n",l,k,i,j,s1,s2,s3,s4,lp_rp[0],lp_rp[1],coef.real,coef.imag);
		      //		      printf("%d \t %d %d %d %d \t %d %d \t %+f %+f\n",count,s1,s2,s3,s4,lp_rp[0],lp_rp[1],coef.real,coef.imag);
		      fprintf(ptr_f,"%d,%d,%d,%d,",s1,s2,s3,s4);
		      j++;
		      count++;
		    }
		  }
	}
    fprintf(ptr_f,"};\n\n");
    //////////////////////////////////////////////////////
    fprintf(ptr_f,"short int lp_rp[768][2] = { ");
    count = 0;
    for(int l = 0 ; l < 4 ; l++)
      for(int k = 0 ; k < 4 ; k++)
	for(int i = 0 ; i < 3 ; i++){
	  j=0;
	  for(int s1=0; s1<4; ++s1)
	    for(int s2=0; s2<4; ++s2)
	      for(int s3=0; s3<4; ++s3)
		for(int s4=0; s4<4; ++s4)
		  { 
		    if(l*4+k < 12)
		      coef = -rhoj(k+1,i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rhoj(l+1,i, s1, s3);
		    else
		      coef = rhoj(k+1,i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rhoj(l+1,i, s1, s3);	      		

		    if( abs(coef) != 0){
		      compute_lp_rp(l,k,i,lp_rp);
		      //		      printf("l(%d)-k(%d)-i(%d)-j(%d)-(%d,%d,%d,%d) \t %d %d \t (%+f,%+f)\n",l,k,i,j,s1,s2,s3,s4,lp_rp[0],lp_rp[1],coef.real,coef.imag);
		      //		      printf("%d \t %d %d %d %d \t %d %d \t %+f %+f\n",count,s1,s2,s3,s4,lp_rp[0],lp_rp[1],coef.real,coef.imag);
		      fprintf(ptr_f,"%d,%d,",lp_rp[0],lp_rp[1]);
		      j++;
		      count++;
		    }
		  }
	}
    fprintf(ptr_f,"};\n\n");

    //////////////////////////////////////////////////////
    fprintf(ptr_f,"float coef[768][2] = { ");
    count = 0;
    for(int l = 0 ; l < 4 ; l++)
      for(int k = 0 ; k < 4 ; k++)
	for(int i = 0 ; i < 3 ; i++){
	  j=0;
	  for(int s1=0; s1<4; ++s1)
	    for(int s2=0; s2<4; ++s2)
	      for(int s3=0; s3<4; ++s3)
		for(int s4=0; s4<4; ++s4)
		  { 
		
		    if(l*4+k < 12)
		      coef = -rhoj(k+1,i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rhoj(l+1,i, s1, s3);
		    else
		      coef = rhoj(k+1,i, s4, s2, true)*g5(s2, s2)*g5(s1, s1)*rhoj(l+1,i, s1, s3);	      

		    if( abs(coef) != 0){
		      compute_lp_rp(l,k,i,lp_rp);
		      //		      printf("l(%d)-k(%d)-i(%d)-j(%d)-(%d,%d,%d,%d) \t %d %d \t (%+f,%+f)\n",l,k,i,j,s1,s2,s3,s4,lp_rp[0],lp_rp[1],coef.real,coef.imag);
		      //		      printf("%d \t %d %d %d %d \t %d %d \t %+f %+f\n",count,s1,s2,s3,s4,lp_rp[0],lp_rp[1],coef.real,coef.imag);
		      fprintf(ptr_f,"%+3.2f,%+3.2f,",coef.real,coef.imag);
		      j++;
		      count++;
		    }
		  }
	}
    fprintf(ptr_f,"};\n\n");


  return 0;
}
