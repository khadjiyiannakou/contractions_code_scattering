#ifndef INTERFACE_H
#define INTERFACE_H

#include <contract.h>

void calculateRho(contract::contractInfo,void*,double*,contract::PRECISION, std::string outfile);
void calculateRhoPi(int idir, contract::contractInfo,void*,double*, double*, double*,contract::PRECISION, std::string outfile);
void calculatePiPi(contract::contractInfo,void*,double*, double*, double*, double*,contract::PRECISION, std::string outfile, contract::ISOSPIN);
void calculateSigma(contract::contractInfo,void*,double*,contract::PRECISION, std::string outfile);
void calculateSigmaPi(contract::contractInfo,void*,double*, double*, double*,contract::PRECISION, std::string outfile);
#endif
