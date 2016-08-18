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


int compute_ip(int idir, int iop){
  int ll;

  switch(iop){
  case 0:
    ll=0;
    break;
  case 1:
    ll=0;
    break;
  case 2:
    ll=1;
    break;
  case 3:
    ll=2+idir;
    break;
  default:
    fprintf(stderr,"I found a mistake\n");
    exit(EXIT_FAILURE);
    break;
  }

  //  if(iop==0) ll=0;
  // if(iop==1) ll=0;
  // if(iop==2) ll=1;
  // if(iop==3) ll=2+idir;

  return ll;
}


int main(){

  matrix g5(4,4);
  g5 = gamma_matrix.g5();

  double_complex coef;

  FILE *ptr_f;
  ptr_f=fopen("constants_rhopi.h","w");

  if(ptr_f == NULL){
    fprintf(stderr,"Error open file\n");
    exit(EXIT_FAILURE);
  }

  int count = 0;

  fprintf(ptr_f,"static short int spinIndices_rhopi[768][6] = { ");
  count=0;
  for(int idir = 0 ; idir < 3 ; idir++)
    for(int iop = 0 ; iop < 4 ; iop++)
      for(int s1=0; s1<4; ++s1)
	for(int s2=0; s2<4; ++s2)
	  for(int s3=0; s3<4; ++s3)
	    for(int s4=0; s4<4; ++s4)
	      for(int s5=0; s5<4; ++s5)
		for(int s6=0; s6<4; ++s6){

		  coef = rhoj(iop+1,idir,s4,s5)*g5(s3,s4)*g5(s6,s1);
		  if(abs(coef) != 0){
		    int ip = compute_ip(idir,iop);
		    //		    printf("%d %d (%d,%d,%d,%d,%d,%d) \t %d \t %+f %+f\n",idir,iop,s1,s2,s3,s4,s5,s6,ip,coef.real,coef.imag);
		    fprintf(ptr_f,"%d,%d,%d,%d,%d,%d,",s1,s2,s3,s4,s5,s6);
		    count++;
		  }
		}
  fprintf(ptr_f,"};\n\n");

  fprintf(ptr_f,"static short int ip_rhopi[768] = { ");
  count=0;
  for(int idir = 0 ; idir < 3 ; idir++)
    for(int iop = 0 ; iop < 4 ; iop++)
      for(int s1=0; s1<4; ++s1)
	for(int s2=0; s2<4; ++s2)
	  for(int s3=0; s3<4; ++s3)
	    for(int s4=0; s4<4; ++s4)
	      for(int s5=0; s5<4; ++s5)
		for(int s6=0; s6<4; ++s6){

		  coef = rhoj(iop+1,idir,s4,s5)*g5(s3,s4)*g5(s6,s1);
		  if(abs(coef) != 0){
		    int ip = compute_ip(idir,iop);
		    //		    printf("%d %d (%d,%d,%d,%d,%d,%d) \t %d \t %+f %+f\n",idir,iop,s1,s2,s3,s4,s5,s6,ip,coef.real,coef.imag);
		    fprintf(ptr_f,"%d,",ip);
		    count++;
		  }
		}
  fprintf(ptr_f,"};\n\n");

  fprintf(ptr_f,"static float coef_rhopi[768][2] = { ");
  count=0;
  for(int idir = 0 ; idir < 3 ; idir++)
    for(int iop = 0 ; iop < 4 ; iop++)
      for(int s1=0; s1<4; ++s1)
	for(int s2=0; s2<4; ++s2)
	  for(int s3=0; s3<4; ++s3)
	    for(int s4=0; s4<4; ++s4)
	      for(int s5=0; s5<4; ++s5)
		for(int s6=0; s6<4; ++s6){

		  coef = rhoj(iop+1,idir,s4,s5)*g5(s3,s4)*g5(s6,s1);
		  if(abs(coef) != 0){
		    int ip = compute_ip(idir,iop);
		    //		    printf("%d %d (%d,%d,%d,%d,%d,%d) \t %d \t %+f %+f\n",idir,iop,s1,s2,s3,s4,s5,s6,ip,coef.real,coef.imag);
		    fprintf(ptr_f,"%+3.2f,%+3.2f,",coef.real,coef.imag);
		    count++;
		  }
		}
  fprintf(ptr_f,"};\n\n");


  return 0;
}
