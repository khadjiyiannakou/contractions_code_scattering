#ifndef CONTRACT_H
#define CONTRACT_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <typeinfo>
#include <string>
#include <string.h>
#include <fstream>
#include <iostream>
#include <cmath>

namespace contract {

  typedef struct{
    int NeV;
    int Nt;
    std::string PropH_fileName;
    int dev;
  }contractInfo;

  // =====================================================//
  enum ALLOCATION_FLAG{NONE,HOST,DEVICE,BOTH,BOTH_EXTRA, DEVICE_EXTRA};
  enum CLASS_ENUM{BASE_CLASS, GAUGE_CLASS, VECTOR_CLASS, COLOR_VECTOR_CLASS, LAPH_EIGENVECTORS_CLASS,
		  LAPH_PROP_CLASS, MOM_MATRIX_CLASS, MOM_MATRICES_CLASS, CONTRACT_RHO_RHO_CLASS,
		  CONTRACT_RHO_PION_PION_CLASS, CONTRACT_PION_PION_PION_PION_CLASS,
                  CONTRACT_LOOP_UNITY,  MOM_MATRICES_CLASS_x2, CONTRACT_SIGMA_SIGMA_CLASS,
                  CONTRACT_SIGMA_PION_PION_CLASS};
  enum PRECISION{SINGLE, DOUBLE};
  enum ISOSPIN{I0,I1};
  // ============= Forward definition of classes ======== //
  template<typename Float> class Base;
  template<typename Float> class Mom_matrices;
  template<typename Float> class Mom_matrices_x2;
  template<typename Float> class Mom_matrix;
  template<typename Float> class LapH_prop;
  template<typename Float> class Contract_rho_rho;
  template<typename Float> class Contract_rho_pipi;
  template<typename Float> class Contract_pipi_pipi;
  template<typename Float> class Contract_loop_unity;
  template<typename Float> class Contract_sigma_sigma;
  template<typename Float> class Contract_sigma_pipi;
  // ==================================================== //
  void initContract(contractInfo info);
  void destroyContract();
  void* readPropH(const std::string& fileName, PRECISION prec); // returns a pointer to the data

  void run_ContractRho(cudaTextureObject_t texProp, cudaTextureObject_t texMom, int tf,int Nt,  void* corr, PRECISION prec);
  void run_CopyConstantsRho();
  
  void run_ContractRhoPi(cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP1P2, int tf, int Nt, int idir, void* corr, PRECISION prec);
  void run_CopyConstantsRhoPi();

  void run_ContractPiPi(cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, int Nt, void* corr, PRECISION prec);

  void run_ContractLoopUnity(cudaTextureObject_t texPropDiag, cudaTextureObject_t texMom, int Nt, void* loop, PRECISION prec);

  void run_CopyConstantsPiPi();

  void run_ContractPiPi_I0(cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP3, cudaTextureObject_t texMomP4, int tf, int Nt, void* corr, PRECISION prec);

  void run_ContractSigma(cudaTextureObject_t texProp, cudaTextureObject_t texMom, int tf,int Nt,  void* corr, PRECISION prec);
  void run_CopyConstantsSigma();

  void run_CopyConstantsSigmaPi();
  void run_ContractSigmaPi_fish_hor(cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, int Nt, void* corr, PRECISION prec);
  void run_ContractSigmaPi_triangle(cudaTextureObject_t texProp, cudaTextureObject_t texPropDiag, cudaTextureObject_t texMomP1, cudaTextureObject_t texMomP2, cudaTextureObject_t texMomP1P2, int tf, int Nt, void* corr, PRECISION prec);
  //=====================================================================//
  template<typename Float>   class Base {    
  protected:
    int NeV;
    int Nt;

    long int totalLength_host;
    long int totalLength_device;

    size_t totalLength_host_bytes;
    size_t totalLength_device_bytes;

    Float *h_elem;
    Float *d_elem;
    Float *d_elem_extra;

    bool isAllocHost;
    bool isAllocDevice;
    bool isAllocDeviceExtra;

    bool isTextureOn;
    bool isTextureExtraOn;

    void create_host();
    void create_device();
    void create_device_extra();

    void destroy_host();
    void destroy_device();
    void destroy_device_extra();

    void zero_host();
    void zero_device();
    void zero_device_extra();
  public:
    Base(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT);
    virtual ~Base();

    void createTexObject(cudaTextureObject_t *tex);
    void destroyTexObject(cudaTextureObject_t tex);

    void createTexObjectExtra(cudaTextureObject_t *tex);
    void destroyTexObjectExtra(cudaTextureObject_t tex);

    size_t Bytes_total_host() const { return totalLength_host_bytes; }
    size_t Bytes_total_device() const { return totalLength_device_bytes; }

    Float* H_elem() const { return h_elem; }
    Float* D_elem() const { return d_elem; }

    Float* D_elem_extra() const { return d_elem_extra; }

    int Precision() const{
      if( typeid(Float) == typeid(float) )
        return sizeof(float);
      else if( typeid(Float) == typeid(double) )
        return sizeof(double);
      else
        return 0;
    }
    void printInfo();
    void copyToDevice();
  };

  //====================================================================//

  template<typename Float> class Mom_matrix : public Base<Float> {
  public:
    Mom_matrix(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT);
    ~Mom_matrix(){;}

  };


  //====================================================================//

  template<typename Float> class Mom_matrices : public Base<Float> {
  public:
    Mom_matrices(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT);
    ~Mom_matrices(){;}

  };

  //====================================================================//

  template<typename Float> class Mom_matrices_x2 : public Base<Float> {
  public:
    Mom_matrices_x2(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT);
    ~Mom_matrices_x2(){;}

  };

  //====================================================================//

  template<typename Float> class LapH_prop : public Base<Float> {
  private:
    bool isRefHost;
  public:
    LapH_prop(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT);
    LapH_prop(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, Float* h_elem_ref); // overloaded constructor
    ~LapH_prop(){;}
    void readPropH(const std::string& fileName);
    void copyToDeviceRowByRow(int it);
    void copyToDeviceExtraDiag();
  };

  //===================================================================//

  template<typename Float> class Contract_rho_rho : public Base<Float> {
  private:
    void copyConstantsRho();
    FILE *pf;
    void dumpData();
  public:
    Contract_rho_rho(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile);
    ~Contract_rho_rho(){fclose(pf);}
    void contractRho(Mom_matrices<Float> &, LapH_prop<Float> &);
  };
  
  // ==================================================================//
  template<typename Float> class Contract_rho_pipi : public Base<Float> {
  private:
    void copyConstantsRhoPi();
    FILE *pf;
    void dumpData();
  public:
    Contract_rho_pipi(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile);
    ~Contract_rho_pipi(){fclose(pf);}
    void contractRhoPi(int idir, Mom_matrix<Float> &, Mom_matrix<Float> &, Mom_matrices<Float> &, LapH_prop<Float> &);
  };
  // ==================================================================//
  template<typename Float> class Contract_pipi_pipi : public Base<Float> {
  private:
    void copyConstantsPiPi();
    FILE *pf;
    void dumpData(ISOSPIN Iso);
  public:
    Contract_pipi_pipi(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile);
    ~Contract_pipi_pipi(){fclose(pf);}
    void contractPiPi(Mom_matrix<Float> &, Mom_matrix<Float> &,Mom_matrix<Float> &, Mom_matrix<Float> &, LapH_prop<Float> &, ISOSPIN Iso);
  };
  //===================================================================//
  template<typename Float> class Contract_loop_unity : public Base<Float> {
  private:
    FILE *pf;
    void dumpData();
  public:
    Contract_loop_unity(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile);
    ~Contract_loop_unity(){fclose(pf);}
    void contractLoopUnity(Mom_matrix<Float> &, LapH_prop<Float> &);
  };

  //===================================================================//

  template<typename Float> class Contract_sigma_sigma : public Base<Float> {
  private:
    void copyConstantsSigma();
    FILE *pf;
    void dumpData();
  public:
    Contract_sigma_sigma(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile);
    ~Contract_sigma_sigma(){fclose(pf);}
    void contractSigma(Mom_matrices_x2<Float> &, LapH_prop<Float> &);
  };
  
  //===================================================================//

  template<typename Float> class Contract_sigma_pipi : public Base<Float> {
  private:
    void copyConstantsSigmaPi();
    FILE *pf;
    void dumpData_fish_hor();
    void dumpData_triangle();
  public:
    Contract_sigma_pipi(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile);
    ~Contract_sigma_pipi(){fclose(pf);}
    void contractSigmaPi(Mom_matrix<Float> &, Mom_matrix<Float> &, Mom_matrices_x2<Float> &, LapH_prop<Float> &);
  };
  
  //===================================================================//

  /*
  template<typename Float> class Contract_sigma_sigma : public base<Float> {
  private:
    void copyConstantsSigmaSigma();
    FILE *pf;
    void dumpData();
  public:
    Contract_sigma_sigma(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile);
    ~Contract_sigma_sigma(){fclose(pf);}
    void contractSigmaSigma_Con(Mom_matrices<Float> &, LapH_prop<Float> &);
  };
  */
  // ==================================================================//
  //  template<typename Float> class LapH_eigenvectors : public Base<Float> {
  // public:
    
  // };
  // ================================================================== //
}


#endif


