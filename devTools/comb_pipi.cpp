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




int main(){

  matrix g5(4,4);
  g5 = gamma_matrix.g5();

  double_complex coef;

  FILE *ptr_f;
  ptr_f=fopen("constants_pipi.h","w");

  if(ptr_f == NULL){
    fprintf(stderr,"Error open file\n");
    exit(EXIT_FAILURE);
  }

  int count = 0;

  //////////////////// square //////////////////////

  fprintf(ptr_f,"static short int spinIndices_pipi_square[256][6] = { ");
  count=0;

  for(int s1=0; s1<4; ++s1)
    for(int s2=0; s2<4; ++s2)
      for(int s3=0; s3<4; ++s3)
	for(int s4=0; s4<4; ++s4)
	  for(int s5=0; s5<4; ++s5)
	    for(int s6=0; s6<4; ++s6){
	      
	      coef = g5(s4,s5)*g5(s6,s1);
	      if(abs(coef) != 0){
		fprintf(ptr_f,"%d,%d,%d,%d,%d,%d,",s1,s2,s3,s4,s5,s6);
		count++;
	      }
	    }
  fprintf(ptr_f,"};\n\n");



  fprintf(ptr_f,"static float coef_pipi_square[256][2] = { ");
  count=0;

  for(int s1=0; s1<4; ++s1)
    for(int s2=0; s2<4; ++s2)
      for(int s3=0; s3<4; ++s3)
	for(int s4=0; s4<4; ++s4)
	  for(int s5=0; s5<4; ++s5)
	    for(int s6=0; s6<4; ++s6){


	      coef = g5(s4,s5)*g5(s6,s1);
	      if(abs(coef) != 0){
		fprintf(ptr_f,"%+3.2f,%+3.2f,",coef.real,coef.imag);
		count++;
	      }
	    }
  fprintf(ptr_f,"};\n\n");

  ///////////// doubleTriangle

  fprintf(ptr_f,"static short int spinIndices_pipi_doubleTriangle[256][6] = { ");
  count=0;

  for(int s1=0; s1<4; ++s1)
    for(int s2=0; s2<4; ++s2)
      for(int s3=0; s3<4; ++s3)
	for(int s4=0; s4<4; ++s4)
	  for(int s5=0; s5<4; ++s5)
	    for(int s6=0; s6<4; ++s6){
	      
	      coef = g5(s2,s3)*g5(s6,s1);
	      if(abs(coef) != 0){
		fprintf(ptr_f,"%d,%d,%d,%d,%d,%d,",s1,s2,s3,s4,s5,s6);
		count++;
	      }
	    }
  fprintf(ptr_f,"};\n\n");



  fprintf(ptr_f,"static float coef_pipi_doubleTriangle[256][2] = { ");
  count=0;

  for(int s1=0; s1<4; ++s1)
    for(int s2=0; s2<4; ++s2)
      for(int s3=0; s3<4; ++s3)
	for(int s4=0; s4<4; ++s4)
	  for(int s5=0; s5<4; ++s5)
	    for(int s6=0; s6<4; ++s6){


	      coef = g5(s2,s3)*g5(s6,s1);
	      if(abs(coef) != 0){
		fprintf(ptr_f,"%+3.2f,%+3.2f,",coef.real,coef.imag);
		count++;
	      }
	    }
  fprintf(ptr_f,"};\n\n");

  ///////////////////// double triangle horizontal
  fprintf(ptr_f,"static short int spinIndices_pipi_doubleTriangle_hor[256][4] = { ");
  count=0;  

  for(int s1 = 0 ; s1 < 4 ; s1++)
    for(int s2 = 0 ; s2 < 4 ; s2++)
      for(int s3 = 0 ; s3 < 4 ; s3++)
	for(int s4 = 0 ; s4 < 4 ; s4++)
	  fprintf(ptr_f,"%d,%d,%d,%d,",s1,s2,s3,s4);
  fprintf(ptr_f,"};\n\n");

  fprintf(ptr_f,"static float coef_pipi_doubleTriangle_hor[256][2] = { ");
  count=0;  
  for(int s1 = 0 ; s1 < 4 ; s1++)
    for(int s2 = 0 ; s2 < 4 ; s2++)
      for(int s3 = 0 ; s3 < 4 ; s3++)
	for(int s4 = 0 ; s4 < 4 ; s4++)
	  fprintf(ptr_f,"%+3.2f,%+3.2f,",1.0,0.0);
  fprintf(ptr_f,"};\n\n");
  ///////////////////
  return 0;
}
