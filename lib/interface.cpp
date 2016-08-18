#include <contract.cpp>
#include <sys/time.h>

using namespace contract;
// ======================================================================================== //
template<typename Float>
void calculateRho(contractInfo info, Float* ptr_prop , double *tmp_momMatrix, std::string outfile){

  //  initContract(info);  
 
  // create and read prop
  //  LapH_prop<Float> *propH = new LapH_prop<Float>(BOTH,LAPH_PROP_CLASS);
  //  propH->readPropH(info.PropH_fileName); // read data on host memory
  LapH_prop<Float> *propH = new LapH_prop<Float>(DEVICE,LAPH_PROP_CLASS,ptr_prop);


  // create and read mom matrices
  Mom_matrices<Float> *momMatrices = new Mom_matrices<Float>(BOTH,MOM_MATRICES_CLASS);

    for(int it = 0 ; it < info.Nt ; it++)
      for(int i = 0 ; i < 5 ; i++)
	for(int iv1 = 0 ; iv1 < info.NeV ; iv1++)
	  for(int iv2 = 0 ; iv2 < info.NeV ; iv2++)
	    for(int ir = 0 ; ir < 2 ; ir++)
	      momMatrices->H_elem()[(it*5*info.NeV*info.NeV + i*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = 
		(Float) tmp_momMatrix[(i*info.Nt*info.NeV*info.NeV + it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir]; // read data on host memory


  // create contraction object for rho
    Contract_rho_rho<Float> *rhoObj = new Contract_rho_rho<Float>(HOST,CONTRACT_RHO_RHO_CLASS, outfile);

  // calculate floating point arithmetics
  // ( 2*8*100^3               +               8*100*100             )*   (768)   *    (Nt*Nt)
  // | mult 2 complex                       tr[A*B^\dagger]          |  # blocks |  from all to    |
  // |  matrices                                                     |           |  all timeslices |

  // for Nt=32 , Flo = 12645826560000 
  //  long int Flo = 12645826560000;
  long int Flo = 12349440000LL*info.Nt*info.Nt;
  struct timeval start, end;
  gettimeofday(&start, NULL);
  rhoObj->contractRho(*momMatrices, *propH);
  gettimeofday(&end, NULL);
  double elapsedTime;
  elapsedTime = ( (end.tv_sec + end.tv_usec / 1000000.0) - (start.tv_sec  + start.tv_usec / 1000000.0));
  if(typeid(Float) == typeid(float))
    printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in single precision\n", elapsedTime, Flo/elapsedTime/1.e9);
  else if(typeid(Float) == typeid(double))
    printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in double precision\n", elapsedTime, Flo/elapsedTime/1.e9);
  else
    ABORT("Error precision must be either single or double\n");
  //  printf("%+e %+e\n",rhoObj->H_elem()[0], rhoObj->H_elem()[1]);

  delete propH;
  delete momMatrices;
  delete rhoObj;
}
// ================================================================================================================//
template<typename Float>
void calculateRhoPi( int idir, contractInfo info, Float* ptr_prop, double *tmpMomP1, double *tmpMomP2, double *tmpMomP1P2, std::string outfile){

  //  initContract(info);  
 
  // create and read prop
  LapH_prop<Float> *propH = new LapH_prop<Float>(DEVICE_EXTRA,LAPH_PROP_CLASS,ptr_prop);   // for rho-pipi contractions we need the diagonal elemens thus need extra memory
  //  propH->readPropH(info.PropH_fileName); // read data on host memory
  
  // create and read mom matrices for rho
  Mom_matrices<Float> *momMatricesP1P2 = new Mom_matrices<Float>(BOTH,MOM_MATRICES_CLASS);
  for(int it = 0 ; it < info.Nt ; it++)
    for(int i = 0 ; i < 5 ; i++)
      for(int iv1 = 0 ; iv1 < info.NeV ; iv1++)
	for(int iv2 = 0 ; iv2 < info.NeV ; iv2++)
	  for(int ir = 0 ; ir < 2 ; ir++)
	    momMatricesP1P2->H_elem()[(it*5*info.NeV*info.NeV + i*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = 
	      (Float) tmpMomP1P2[(i*info.Nt*info.NeV*info.NeV + it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir]; // read data on host memory
  
  Mom_matrix<Float> *momMatrixP1 = new Mom_matrix<Float>(BOTH, MOM_MATRIX_CLASS);
  Mom_matrix<Float> *momMatrixP2 = new Mom_matrix<Float>(BOTH, MOM_MATRIX_CLASS);
  
  for(int it = 0 ; it < info.Nt ; it++)
    for(int iv1 = 0 ; iv1 < info.NeV ; iv1++)
      for(int iv2 = 0 ; iv2 < info.NeV ; iv2++)
	for(int ir = 0 ; ir < 2 ; ir++){
	  momMatrixP1->H_elem()[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = (Float) tmpMomP1[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir];
	  momMatrixP2->H_elem()[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = (Float) tmpMomP2[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir];
      }


  // create contraction object for rho
  Contract_rho_pipi<Float> *rhoPiObj = new Contract_rho_pipi<Float>(HOST,CONTRACT_RHO_PION_PION_CLASS, outfile);

  // calculate floating point arithmetics
  // ( 4*8*100^3               +               8*100*100             )*   (256)   *    (Nt*Nt)
  // | mult 4 complex                       tr[A*B^\dagger]          |  # blocks |  from all to    |
  // |  matrices                                                     |           |  all timeslices |

  long int Flo = 8212480000LL*info.Nt*info.Nt;
  struct timeval start, end;
  gettimeofday(&start, NULL);
  rhoPiObj->contractRhoPi(idir, *momMatrixP1, *momMatrixP2, *momMatricesP1P2, *propH); 
  gettimeofday(&end, NULL);
  double elapsedTime;
  elapsedTime = ( (end.tv_sec + end.tv_usec / 1000000.0) - (start.tv_sec  + start.tv_usec / 1000000.0));

  if(typeid(Float) == typeid(float))
    printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in single precision\n", elapsedTime, Flo/elapsedTime/1.e9);
  else if(typeid(Float) == typeid(double))
    printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in double precision\n", elapsedTime, Flo/elapsedTime/1.e9);
  else
    ABORT("Error precision must be either single or double\n");


  delete rhoPiObj;
  delete propH;
  delete momMatrixP1;
  delete momMatrixP2;
  delete momMatricesP1P2;
}

// ======================================================================================================= //
template<typename Float>
void calculatePiPi(contractInfo info,Float* ptr_prop, double *tmpMomP1, double *tmpMomP2, double *tmpMomP3, double *tmpMomP4, std::string outfile, ISOSPIN Iso){

  //  initContract(info);  
 
  // create and read prop
  LapH_prop<Float> *propH = new LapH_prop<Float>(DEVICE_EXTRA,LAPH_PROP_CLASS, ptr_prop);   // for pipi-pipi contractions we need the diagonal elemens thus need extra memory
  //  propH->readPropH(info.PropH_fileName); // read data on host memory
    
  Mom_matrix<Float> *momMatrixP1 = new Mom_matrix<Float>(BOTH, MOM_MATRIX_CLASS);
  Mom_matrix<Float> *momMatrixP2 = new Mom_matrix<Float>(BOTH, MOM_MATRIX_CLASS);
  Mom_matrix<Float> *momMatrixP3 = new Mom_matrix<Float>(BOTH, MOM_MATRIX_CLASS);
  Mom_matrix<Float> *momMatrixP4 = new Mom_matrix<Float>(BOTH, MOM_MATRIX_CLASS);
  

  for(int it = 0 ; it < info.Nt ; it++)
    for(int iv1 = 0 ; iv1 < info.NeV ; iv1++)
      for(int iv2 = 0 ; iv2 < info.NeV ; iv2++)
	for(int ir = 0 ; ir < 2 ; ir++){
	  momMatrixP1->H_elem()[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = (Float) tmpMomP1[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir];
	  momMatrixP2->H_elem()[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = (Float) tmpMomP2[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir];
	  momMatrixP3->H_elem()[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = (Float) tmpMomP3[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir];
	  momMatrixP4->H_elem()[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = (Float) tmpMomP4[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir];
	}


  Contract_pipi_pipi<Float> *piPiObj = new Contract_pipi_pipi<Float>(HOST,CONTRACT_PION_PION_PION_PION_CLASS, outfile);

  // calculate floating point arithmetics

  // square diagram
  // ( 6*8*100^3               +               8*100*100             )*   (256*Nt)   *    (Nt)
  // | mult 6 complex                       tr[A*B^\dagger]          |  # blocks |  from all to    |
  // |  matrices                                                     |           |  all timeslices |

  // double triangle diagram
  // ( 6*8*100^3               +               8*100*100             )*   (256*Nt)   *    (Nt)
  // | mult 6 complex                       tr[A*B^\dagger]          |  # blocks |  from all to    |
  // |  matrices                                                     |           |  all timeslices |

  // included only when I=0
  // double triangle_hor diagram
  // ( 6*8*100^3               +               8*100*100             )*   (256*Nt)   *    (Nt)
  // | mult 6 complex                       tr[A*B^\dagger]          |  # blocks |  from all to    |
  // |  matrices                                                     |           |  all timeslices |

  // star diagram * 2 needed for two traces
  // ( 2*8*100^3               +               8*100*100             )*   (16*Nt)   *    (Nt)
  // | mult 2 complex                       tr[A*B^\dagger]          |  # blocks |  from all to    |
  // |  matrices                                                     |           |  all timeslices |

  // fish diagram * 2 needed for two traces
  // ( 2*8*100^3               +               8*100*100             )*   (16*Nt)   *    (Nt)
  // | mult 2 complex                       tr[A*B^\dagger]          |  # blocks |  from all to    |
  // |  matrices                                                     |           |  all timeslices |
  


  long int Flo;
  if(Iso == I1)
    Flo = (12308480000LL + 12308480000LL + 514560000LL + 514560000LL)*info.Nt*info.Nt;
  else if(Iso == I0)
    Flo = (12308480000LL + 12308480000LL + 12308480000LL + 514560000LL + 514560000LL)*info.Nt*info.Nt;
  else
    ABORT("This Isospin is not implemeted\n");

  struct timeval start, end;
  gettimeofday(&start, NULL);
  piPiObj->contractPiPi(*momMatrixP1, *momMatrixP2, *momMatrixP3, *momMatrixP4, *propH, Iso); 
  gettimeofday(&end, NULL);
  double elapsedTime;
  elapsedTime = ( (end.tv_sec + end.tv_usec / 1000000.0) - (start.tv_sec  + start.tv_usec / 1000000.0));
  //  printf("Elapsed time is %lf sec\n",elapsedTime);
  if(typeid(Float) == typeid(float))
    printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in single precision\n", elapsedTime, Flo/elapsedTime/1.e9);
  else if(typeid(Float) == typeid(double))
    printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in double precision\n", elapsedTime, Flo/elapsedTime/1.e9);
  else
    ABORT("Error precision must be either single or double\n");


  delete piPiObj;
  delete propH;
  delete momMatrixP1;
  delete momMatrixP2;
  delete momMatrixP3;
  delete momMatrixP4;
}


template<typename Float>
void calculateSigma(contractInfo info, Float* ptr_prop , double *tmp_momMatrix, std::string outfile){

  LapH_prop<Float> *propH = new LapH_prop<Float>(DEVICE_EXTRA,LAPH_PROP_CLASS,ptr_prop);
  Mom_matrix<Float> *momMatrix = new Mom_matrix<Float>(BOTH,MOM_MATRIX_CLASS);
  Contract_loop_unity<Float> *loopObj = new Contract_loop_unity<Float>(HOST,CONTRACT_LOOP_UNITY, outfile);

  for(int i = 0 ; i < 4 ; i++){ // we need to calculate the loop for e^{+ip} k gia e^{-ip} and for the smeared case
    
    for(int it = 0 ; it < info.Nt ; it++)
      for(int iv1 = 0 ; iv1 < info.NeV ; iv1++)
	for(int iv2 = 0 ; iv2 < info.NeV ; iv2++)
	  for(int ir = 0 ; ir < 2 ; ir++)
	    momMatrix->H_elem()[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = 
	      (Float) tmp_momMatrix[(i*info.Nt*info.NeV*info.NeV + it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir]; // read data on host memory

    // loop
    // (  8*100*100  )*         (4*Nt)
    // | tr[A*B^\dagger]  |  # blocks | 
    // |  matrices                    

    long int Flo = (320000LL)*info.Nt;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    loopObj->contractLoopUnity(*momMatrix,*propH);
    gettimeofday(&end, NULL);
    double elapsedTime;
    elapsedTime = ( (end.tv_sec + end.tv_usec / 1000000.0) - (start.tv_sec  + start.tv_usec / 1000000.0));
    if(typeid(Float) == typeid(float))
      printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in single precision\n", elapsedTime, Flo/elapsedTime/1.e9);
    else if(typeid(Float) == typeid(double))
      printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in double precision\n", elapsedTime, Flo/elapsedTime/1.e9);
    else
      ABORT("Error precision must be either single or double\n");
    //  printf("%+e %+e\n",rhoObj->H_elem()[0], rhoObj->H_elem()[1]);


  }

  Mom_matrices_x2<Float> *momMatrices_x2 = new Mom_matrices_x2<Float>(BOTH,MOM_MATRICES_CLASS_x2);

  for(int it = 0 ; it < info.Nt ; it++)
    for(int i = 0 ; i < 2 ; i++)
      for(int iv1 = 0 ; iv1 < info.NeV ; iv1++)
	for(int iv2 = 0 ; iv2 < info.NeV ; iv2++)
	  for(int ir = 0 ; ir < 2 ; ir++)
	    momMatrices_x2->H_elem()[(it*2*info.NeV*info.NeV + i*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = 
	      (Float) tmp_momMatrix[((i*2+1)*info.Nt*info.NeV*info.NeV + it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir]; // we choose only e^{-ip} and e^{-ip}_smeared


  // calculate floating point arithmetics
  // ( 2*8*100^3               +               8*100*100             )*   (Nt*64)   *    (Nt)
  // | mult 2 complex                       tr[A*B^\dagger]          |  # blocks |       all   |
  // |  matrices                                                     |                timeslices |

  Contract_sigma_sigma<Float> *sigmaObj = new Contract_sigma_sigma<Float>(HOST,CONTRACT_SIGMA_SIGMA_CLASS, outfile);

  long int Flo = (1029120000LL)*info.Nt*info.Nt;
  struct timeval start, end;
  gettimeofday(&start, NULL);
  sigmaObj->contractSigma(*momMatrices_x2, *propH);
  gettimeofday(&end, NULL);
  double elapsedTime;
  elapsedTime = ( (end.tv_sec + end.tv_usec / 1000000.0) - (start.tv_sec  + start.tv_usec / 1000000.0));

  if(typeid(Float) == typeid(float))
    printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in single precision\n", elapsedTime, Flo/elapsedTime/1.e9);
  else if(typeid(Float) == typeid(double))
    printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in double precision\n", elapsedTime, Flo/elapsedTime/1.e9);
  else
    ABORT("Error precision must be either single or double\n");


  delete sigmaObj;
  delete momMatrices_x2;
  delete propH;
  delete momMatrix;
  delete loopObj;

}


template<typename Float>
void calculateSigmaPi(contractInfo info, Float* ptr_prop, double *tmpMomP1, double *tmpMomP2, double *tmpMomP1P2, std::string outfile){

  LapH_prop<Float> *propH = new LapH_prop<Float>(DEVICE_EXTRA,LAPH_PROP_CLASS,ptr_prop);
  Mom_matrix<Float> *momMatrixP1 = new Mom_matrix<Float>(BOTH,MOM_MATRIX_CLASS);
  Mom_matrix<Float> *momMatrixP2 = new Mom_matrix<Float>(BOTH,MOM_MATRIX_CLASS);
  Mom_matrices_x2<Float> *momMatricesP1P2_x2 = new Mom_matrices_x2<Float>(BOTH,MOM_MATRICES_CLASS_x2);
  Contract_loop_unity<Float> *loopObj = new Contract_loop_unity<Float>(HOST,CONTRACT_LOOP_UNITY, outfile);
  Contract_sigma_pipi<Float> *sigmapiObj = new Contract_sigma_pipi<Float>(HOST,CONTRACT_SIGMA_PION_PION_CLASS,outfile);


  for(int i = 0 ; i < 2 ; i++){ // we need to calculate the loop for e^{+ip1p2} and the smeared case 
    
    for(int it = 0 ; it < info.Nt ; it++)
      for(int iv1 = 0 ; iv1 < info.NeV ; iv1++)
	for(int iv2 = 0 ; iv2 < info.NeV ; iv2++)
	  for(int ir = 0 ; ir < 2 ; ir++)
	    momMatrixP1->H_elem()[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = 
	      (Float) tmpMomP1P2[(i*info.Nt*info.NeV*info.NeV + it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir]; // read data on host memory

    // loop
    // (  8*100*100  )*         (4*Nt)
    // | tr[A*B^\dagger]  |  # blocks | 
    // |  matrices                    

    long int Flo = (320000LL)*info.Nt;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    loopObj->contractLoopUnity(*momMatrixP1,*propH);
    gettimeofday(&end, NULL);
    double elapsedTime;
    elapsedTime = ( (end.tv_sec + end.tv_usec / 1000000.0) - (start.tv_sec  + start.tv_usec / 1000000.0));
    if(typeid(Float) == typeid(float))
      printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in single precision\n", elapsedTime, Flo/elapsedTime/1.e9);
    else if(typeid(Float) == typeid(double))
      printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in double precision\n", elapsedTime, Flo/elapsedTime/1.e9);
    else
      ABORT("Error precision must be either single or double\n");
  }



  for(int it = 0 ; it < info.Nt ; it++)
    for(int i = 0 ; i < 2 ; i++)
      for(int iv1 = 0 ; iv1 < info.NeV ; iv1++)
	for(int iv2 = 0 ; iv2 < info.NeV ; iv2++)
	  for(int ir = 0 ; ir < 2 ; ir++)
	    momMatricesP1P2_x2->H_elem()[(it*2*info.NeV*info.NeV + i*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = 
	      (Float) tmpMomP1P2[(i*info.Nt*info.NeV*info.NeV + it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir]; // read data on host memory
  
  
  for(int it = 0 ; it < info.Nt ; it++)
    for(int iv1 = 0 ; iv1 < info.NeV ; iv1++)
      for(int iv2 = 0 ; iv2 < info.NeV ; iv2++)
	for(int ir = 0 ; ir < 2 ; ir++){
	  momMatrixP1->H_elem()[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = (Float) tmpMomP1[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir];
	  momMatrixP2->H_elem()[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir] = (Float) tmpMomP2[(it*info.NeV*info.NeV + iv1*info.NeV + iv2)*2+ir];
      }




  // calculate floating point arithmetics

  // two point
  // ( 2*8*100^3               +               8*100*100             )* (Nt*16)  |
  // | mult 2 complex                       tr[A*B^\dagger]          |  # blocks |
  // |  matrices                                                     |            

  // triangle
  // ( 4*8*100^3               +               8*100*100             )*(Nt*128)   *    Nt
  // | mult 4 complex                       tr[A*B^\dagger]          |  # blocks |                 |
  // |  matrices                                                     |           |  all timeslices |

  long int Flo = 257280000LL*info.Nt + 4106240000LL*info.Nt*info.Nt;
  struct timeval start, end;
  gettimeofday(&start, NULL);
  sigmapiObj->contractSigmaPi(*momMatrixP1, *momMatrixP2, *momMatricesP1P2_x2, *propH); 
  gettimeofday(&end, NULL);
  double elapsedTime;
  elapsedTime = ( (end.tv_sec + end.tv_usec / 1000000.0) - (start.tv_sec  + start.tv_usec / 1000000.0));

  if(typeid(Float) == typeid(float))
    printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in single precision\n", elapsedTime, Flo/elapsedTime/1.e9);
  else if(typeid(Float) == typeid(double))
    printf("Elapsed time is %lf sec, kernel performance is %lf GFlops in double precision\n", elapsedTime, Flo/elapsedTime/1.e9);
  else
    ABORT("Error precision must be either single or double\n");


  delete propH;
  delete momMatrixP1;
  delete momMatrixP2;
  delete momMatricesP1P2_x2;
  delete loopObj;
  delete sigmapiObj;
}

// ======================================================================================================= //
void calculateRho(contractInfo info, void *ptr_prop, double *tmp_momMatrix, PRECISION prec, std::string outfile){
  if(prec == SINGLE)
    calculateRho<float>(info, (float*) ptr_prop , tmp_momMatrix, outfile);
  else
    calculateRho<double>(info, (double*) ptr_prop, tmp_momMatrix, outfile);
}

void calculateRhoPi( int idir, contractInfo info, void *ptr_prop, double *tmpMomP1, double *tmpMomP2, double *tmpMomP1P2, PRECISION prec, std::string outfile){
  if(prec == SINGLE)
    calculateRhoPi<float>(idir,info,(float*) ptr_prop, tmpMomP1, tmpMomP2, tmpMomP1P2, outfile);
  else
    calculateRhoPi<double>(idir,info,(double*) ptr_prop,tmpMomP1, tmpMomP2, tmpMomP1P2, outfile);
}

void calculatePiPi(contractInfo info, void *ptr_prop, double *tmpMomP1, double *tmpMomP2, double *tmpMomP3, double *tmpMomP4, PRECISION prec, std::string outfile, ISOSPIN Iso){
  if(prec == SINGLE)
    calculatePiPi<float>(info,(float*) ptr_prop, tmpMomP1, tmpMomP2, tmpMomP3, tmpMomP4, outfile, Iso);
  else
    calculatePiPi<double>(info,(double*) ptr_prop,tmpMomP1, tmpMomP2, tmpMomP3, tmpMomP4, outfile, Iso);
}

void calculateSigma(contractInfo info, void *ptr_prop, double *tmp_momMatrix, PRECISION prec, std::string outfile){
  if(prec == SINGLE)
    calculateSigma<float>(info, (float*) ptr_prop , tmp_momMatrix, outfile);
  else
    calculateSigma<double>(info, (double*) ptr_prop, tmp_momMatrix, outfile);  
}

void calculateSigmaPi(contractInfo info, void *ptr_prop, double *tmpMomP1, double *tmpMomP2, double *tmpMomP1P2, PRECISION prec, std::string outfile){
  if(prec == SINGLE)
    calculateSigmaPi<float>(info,(float*) ptr_prop, tmpMomP1, tmpMomP2, tmpMomP1P2, outfile);
  else
    calculateSigmaPi<double>(info,(double*) ptr_prop,tmpMomP1, tmpMomP2, tmpMomP1P2, outfile);
}
// ======================================================================================================= //
