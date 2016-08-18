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
    int j;
    FILE *ptr_f;
    ptr_f=fopen("constants_sigmapi.h","w");

    if(ptr_f == NULL){
      fprintf(stderr,"Error open file\n");
      exit(EXIT_FAILURE);
    }

    int ip;
    int count = 0;
    //////////////////////////////////////////////////////////
    fprintf(ptr_f,"static short int spinIndices_sigmapi_hor_fish[16][2] = { ");
    count = 0;

    for(int s1=0; s1<4; ++s1)
      for(int s2=0; s2<4; ++s2){ 
	fprintf(ptr_f,"%d,%d,",s1,s2);
      }
    fprintf(ptr_f,"};\n\n");

    //////////////////////////////////////////////////////
    fprintf(ptr_f,"static float coef_sigmapi_hor_fish[16][2] = { ");
    count = 0;

    for(int s1=0; s1<4; ++s1)
      for(int s2=0; s2<4; ++s2){     
	fprintf(ptr_f,"%+3.2f,%+3.2f,",1.0,0.0);
      }    
    
    fprintf(ptr_f,"};\n\n");
    //////////////////////////////////////////////////////


    //////////////////////////////////////////////////////

    fprintf(ptr_f,"static short int spinIndices_sigmapi_triangle[128][5] = { ");
    for(int ik = 0 ; ik < 2 ; ik++)
      for(int s1=0; s1<4; ++s1)
        for(int s2=0; s2<4; ++s2)
          for(int s3=0; s3<4; ++s3)
            for(int s4=0; s4<4; ++s4)
              for(int s5=0; s5<4; ++s5){
		coef = g5(s1,s2)*g5(s3,s4);
		if(abs(coef) != 0){
		  fprintf(ptr_f,"%d,%d,%d,%d,%d,",s1,s2,s3,s4,s5);
		}
	      }
    fprintf(ptr_f,"};\n\n");

    ///////////////////////////////////////////////////////
    fprintf(ptr_f,"static short int ip_sigmapi_triangle[128] = { ");
    for(int ik = 0 ; ik < 2 ; ik++)
      for(int s1=0; s1<4; ++s1)
        for(int s2=0; s2<4; ++s2)
          for(int s3=0; s3<4; ++s3)
            for(int s4=0; s4<4; ++s4)
              for(int s5=0; s5<4; ++s5){
		coef = g5(s1,s2)*g5(s3,s4);
		if(abs(coef) != 0){
		  fprintf(ptr_f,"%d,",ik);
		}
	      }
    fprintf(ptr_f,"};\n\n");
    /////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////
    fprintf(ptr_f,"static float coef_sigmapi_triangle[128][2] = { ");
    for(int ik = 0 ; ik < 2 ; ik++)
      for(int s1=0; s1<4; ++s1)
        for(int s2=0; s2<4; ++s2)
          for(int s3=0; s3<4; ++s3)
            for(int s4=0; s4<4; ++s4)
              for(int s5=0; s5<4; ++s5){
		coef = g5(s1,s2)*g5(s3,s4);
		if(abs(coef) != 0){
		  fprintf(ptr_f,"%+3.2f,%+3.2f,",coef.real,coef.imag);
		}
	      }
    fprintf(ptr_f,"};\n\n");
    /////////////////////////////////////////////////////

    fclose(ptr_f);
  return 0;
}
