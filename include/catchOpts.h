#ifndef CATCHOPTS_H
#define CATCHOPTS_H

#include <utils.h>
#include <sys/stat.h>
#include <sstream>
#include <contract.h>

namespace contract{
  // declare global variables for the options
  
  extern  std::string latname;
  extern  std::string eigename; 
  extern std::string propname;
  extern std::string momname;
  extern std::string outname;

  extern PRECISION Prec;

  extern int devID;
  extern int Nthreads;
  extern int Nx;
  extern int Ny;
  extern int Nz;
  extern int Nt;
  extern int Nvec;
  extern int Nmom;
  extern int momComb;
  extern int i_dir;

  extern int *px1;
  extern int *px2;
  extern int *px3;
  extern int *px4;

  extern int *py1;
  extern int *py2;
  extern int *py3;
  extern int *py4;

  extern int *pz1;
  extern int *pz2;
  extern int *pz3;
  extern int *pz4;
  
  void catchOpts(int argc, char **argv);
  void destroyOpts(); // this will free any memory allocated from the catchOpts
}

#endif

