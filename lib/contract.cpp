#include <contract.h>
#include <catchOpts.h>

#include <utils.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <stdint.h>

#define CC Base<Float>

using namespace contract;
using contract::i_dir;

bool G_isContractOn = false;
int G_NeV;
int G_Nt;
int G_dev;

extern bool isConstantRhoOn;
extern bool isConstantRhoPiPiOn;
extern bool isConstantPiPiPiPiOn;
extern bool isConstantSigmaOn;

static void initDevice(){
  if(!G_isContractOn)
    ABORT("Error: contract library is not on\n");

  int deviceCount;
  cudaDeviceProp deviceProp;

  cudaGetDeviceCount(&deviceCount);
  if(deviceCount == 0)
    ABORT("Error: No GPU devices found\n");
  else
    printf("Found %d devices\n",deviceCount);

  CHECK_CUDA_ERROR();

  for(int i = 0 ; i < deviceCount ; i++){
    cudaGetDeviceProperties(&deviceProp, i);
    CHECK_CUDA_ERROR();
    printf("Found device %d: %s with %f Mb of memory\n", i, deviceProp.name, (float)deviceProp.totalGlobalMem/(1024.*1024.));
  }

  if(G_dev < 0 || G_dev >= deviceCount){
    printf("Error: Set device id %d is not allowed\n",G_dev);
    ABORT("Aborting...\n");
  }

  cudaSetDevice(G_dev);
  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();
  printf("Using device %d\n",G_dev);

}

void contract::initContract(contractInfo info){
  if(info.NeV != 100)
    ABORT("This package works only with 100 eigenVectors, if you want more ask the developer\n");

  if(G_isContractOn)
    ABORT("Error: initContract can be called once only\n");

  G_NeV = info.NeV;
  G_Nt = info.Nt;
  G_dev = info.dev;
  G_isContractOn = true;
  initDevice();
}

void contract::destroyContract(){
  if(!G_isContractOn)
    ABORT("Error: cannot destroy contract library because is not on\n");
  
  cudaDeviceReset(); // release all constant memory also
  CHECK_CUDA_ERROR();
  if(isConstantRhoOn)
    isConstantRhoOn=false;
  if(isConstantRhoPiPiOn)
    isConstantRhoPiPiOn=false;
  if(isConstantPiPiPiPiOn)
    isConstantPiPiPiPiOn=false;
  if(isConstantSigmaOn)
    isConstantSigmaOn=false;

  G_isContractOn = false;
}


void* contract::readPropH(const std::string& fileName, PRECISION prec){
  if(!G_isContractOn)
    ABORT("Error: Need to initialize contract library before read propagator\n");

  long int totalLength = G_Nt * G_Nt * 4 * 4 * G_NeV * G_NeV;

  void* p_Prop;

  if(prec == SINGLE)
    p_Prop =  malloc(totalLength*2*sizeof(float));
  else
    p_Prop =  malloc(totalLength*2*sizeof(double));

  if(p_Prop == NULL)
    ABORT("Error: Not enough memory to allocate\n");

  FILE* f = fopen(fileName.c_str(), "rb");
  if(f == NULL)
    ABORT("Error: Cannot find file for reading propagatorH\n");

  size_t checkBytes;
  char cfgname[100];
  checkBytes = fread(cfgname, 1, 100, f); cfgname[99] = 0;
  std::cout << "Got configuration ID from file: " << cfgname << std::endl;

  uint32_t nt, nvec;
  checkBytes = fread(&nt, 4, 1, f);
  checkBytes = fread(&nvec, 4, 1, f);

  if( (int)nvec != G_NeV )
    ABORT("Error: NeV given is not equal with what I get from file\n");
  if( (int)nt != G_Nt )
    ABORT("Error: Nt given is not equal with what I get from file\n");

  void *buffer = malloc(totalLength*2*sizeof(double));
  if(buffer == NULL) ABORT("Error allocating memory for buffer reading propH\n");

  checkBytes = fread(buffer, 2*sizeof(double), totalLength, f);
  if((int)checkBytes != totalLength) 
    ABORT("Error: Did not read size expected\n");
 
  if(prec == SINGLE){
    for(int t1 = 0 ; t1 < G_Nt ; t1++)
      for(int t2 = 0 ; t2 < G_Nt ; t2++)
	for(int s1 = 0 ; s1 < 4 ; s1++)
	  for(int s2 = 0 ; s2 < 4 ; s2++)
	    for(int v1 = 0 ; v1 < G_NeV ; v1++)
	      for(int v2 = 0 ; v2 < G_NeV ; v2++)
		for(int ir = 0 ; ir < 2 ; ir++){
		  ((float*)p_Prop)[(t1*G_Nt*4*4*G_NeV*G_NeV + t2*4*4*G_NeV*G_NeV + s1*4*G_NeV*G_NeV + s2*G_NeV*G_NeV + v1*G_NeV + v2)*2 + ir] = 
		    (float)  ((double*)buffer)[(s1*G_NeV*G_Nt*4*G_NeV*G_Nt + v1*G_Nt*4*G_NeV*G_Nt + t1*4*G_NeV*G_Nt + s2*G_NeV*G_Nt + v2*G_Nt + t2)*2 + ir];
		}
  }
  else{
    for(int t1 = 0 ; t1 < G_Nt ; t1++)
      for(int t2 = 0 ; t2 < G_Nt ; t2++)
	for(int s1 = 0 ; s1 < 4 ; s1++)
	  for(int s2 = 0 ; s2 < 4 ; s2++)
	    for(int v1 = 0 ; v1 < G_NeV ; v1++)
	      for(int v2 = 0 ; v2 < G_NeV ; v2++)
		for(int ir = 0 ; ir < 2 ; ir++){
		  ((double*)p_Prop)[(t1*G_Nt*4*4*G_NeV*G_NeV + t2*4*4*G_NeV*G_NeV + s1*4*G_NeV*G_NeV + s2*G_NeV*G_NeV + v1*G_NeV + v2)*2 + ir] = 
		    ((double*)buffer)[(s1*G_NeV*G_Nt*4*G_NeV*G_Nt + v1*G_Nt*4*G_NeV*G_Nt + t1*4*G_NeV*G_Nt + s2*G_NeV*G_Nt + v2*G_Nt + t2)*2 + ir];
		}
  }

  free(buffer);
  fclose(f);
  return p_Prop;
}


// ======================================= Base Class ========================================== //

template<typename Float>
Base<Float>::Base(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT):
  h_elem(NULL), d_elem(NULL), d_elem_extra(NULL), isAllocHost(false), isAllocDevice(false), isTextureOn(false), isTextureExtraOn(false){
  if(!G_isContractOn) 
    ABORT("You need to initialize the library first\n");

  switch(classT){
  case BASE_CLASS:
    totalLength_host = 0;
    totalLength_device = 0;
    break;
  case LAPH_PROP_CLASS:
    totalLength_host = G_Nt * G_Nt * 4 * 4 * G_NeV * G_NeV;
    totalLength_device = G_Nt * 4 * 4 * G_NeV * G_NeV; // device holds only one row of times because in either case we do not have enough memory
    break;
  case MOM_MATRICES_CLASS: 
    totalLength_host = G_Nt * 5 * G_NeV * G_NeV; // 5 means we have e^{ip}, \nabla e^{ip}, commutator{e^{ip},\nabla(i=1,2,3)} => 5 cases 
    totalLength_device = totalLength_host;
    break;
  case MOM_MATRICES_CLASS_x2: 
    totalLength_host = G_Nt * 2 * G_NeV * G_NeV; // 2 means e^{ip}, \nabla e^{ip}
    totalLength_device = totalLength_host;
    break;
  case MOM_MATRIX_CLASS:
    totalLength_host = G_Nt * G_NeV * G_NeV;
    totalLength_device = totalLength_host;
    break;
  case CONTRACT_RHO_RHO_CLASS:
    totalLength_host = G_Nt * G_Nt * 16 * 3; // 16 combinations of operatos and 3 directions for the spatial directions
    totalLength_device = totalLength_host; // I do not think I need device memory for this but in any case
    break;
  case CONTRACT_RHO_PION_PION_CLASS:
    totalLength_host = G_Nt * G_Nt * 4; // 4 operators for rho use only one direction each time
    totalLength_device = totalLength_host;
    break;
  case CONTRACT_PION_PION_PION_PION_CLASS:
    totalLength_host = G_Nt * G_Nt * 5;   // in the case of I0 we have 5 diagrams, in the case of I1 we have 4 diagrams thus I choose 5 to have enough space
    totalLength_device = totalLength_host;
    break;
  case CONTRACT_LOOP_UNITY:
    totalLength_host = G_Nt;
    totalLength_device = totalLength_host;
    break;
  case CONTRACT_SIGMA_SIGMA_CLASS:
    totalLength_host = G_Nt * G_Nt * 4;
    totalLength_device = totalLength_host;
    break;
  case CONTRACT_SIGMA_PION_PION_CLASS:
    totalLength_host = G_Nt + G_Nt * G_Nt * 2;   // the first G_Nt is for the horizontal fish, next G_Nt * G_Nt * 2 is for the triangle
    totalLength_device = totalLength_host;
    break;
  default:
    ABORT("Error: The class you ask is not implemented yet\n");
    break;
  }

  totalLength_host_bytes = totalLength_host * 2 *sizeof(Float);
  totalLength_device_bytes = totalLength_device * 2 *sizeof(Float); // 2 is fo complex

  if(classT == BASE_CLASS)
    return;

  switch(alloc_flag){
  case HOST:
    create_host();
    break;
  case DEVICE:
    create_device();
    break;
  case BOTH:
    create_host();
    create_device();
    break;
  case BOTH_EXTRA:
    create_host();
    create_device();
    create_device_extra();
    break;
  case DEVICE_EXTRA:
    create_device();
    create_device_extra();
    break;
  case NONE:
    break;
  }

}

template<typename Float>
Base<Float>::~Base(){
  if(isAllocHost) destroy_host();
  if(isAllocDevice) destroy_device();
  if(isAllocDeviceExtra) destroy_device_extra();
}

template<typename Float>
void Base<Float>::create_host(){
  h_elem = (Float*) malloc(totalLength_host_bytes);
  if(h_elem == NULL)ABORT("Error: Allocating memory for host\n");
  isAllocHost=true;
  zero_host();
}

template<typename Float>
void Base<Float>::create_device(){
  cudaMalloc((void**)&d_elem, totalLength_device_bytes);
  CHECK_CUDA_ERROR();
  isAllocDevice=true;
  zero_device();
}

template<typename Float>
void Base<Float>::create_device_extra(){
  cudaMalloc((void**)&d_elem_extra, totalLength_device_bytes);
  CHECK_CUDA_ERROR();
  isAllocDeviceExtra=true;
  zero_device_extra();
}

template<typename Float>
void Base<Float>::destroy_host(){
  if(isAllocHost)
    free(h_elem);
  isAllocHost=false;
}

template<typename Float>
void Base<Float>::destroy_device(){
  if(isAllocDevice)
    cudaFree(d_elem);
  isAllocDevice=false;
}

template<typename Float>
void Base<Float>::destroy_device_extra(){
  if(isAllocDeviceExtra)
    cudaFree(d_elem_extra);
  isAllocDeviceExtra=false;
}

template<typename Float>
void Base<Float>::zero_host(){
  if(isAllocHost)
    memset(h_elem,0,totalLength_host_bytes);
  else
    ABORT("Error: Try initialize memory not allocated in not permited\n");
}

template<typename Float>
void Base<Float>::zero_device(){
  if(isAllocDevice)
    cudaMemset(d_elem,0,totalLength_device_bytes);
  else
    ABORT("Error: Try initialize memory not allocated in not permited\n");
}

template<typename Float>
void Base<Float>::zero_device_extra(){
  if(isAllocDeviceExtra)
    cudaMemset(d_elem_extra,0,totalLength_device_bytes);
  else
    ABORT("Error: Try initialize memory not allocated in not permited\n");
}

template<typename Float>
void Base<Float>::createTexObject(cudaTextureObject_t *tex){

  if(!isAllocDevice)
    ABORT("Error: Create texture object for memory not allocated is not allowed\n");
  if(isTextureOn)
    ABORT("Error: Textures object is already on\n");
  
  cudaChannelFormatDesc desc;
  memset(&desc, 0, sizeof(cudaChannelFormatDesc));
  int precision = Precision();
  if(precision == sizeof(float)) desc.f = cudaChannelFormatKindFloat;
  else desc.f = cudaChannelFormatKindSigned;

  if(precision == sizeof(float)){
    desc.x = 8*precision;
    desc.y = 8*precision;
    desc.z = 0;
    desc.w = 0;
  }
  else if(precision == sizeof(double)){
    desc.x = 8*precision/2;
    desc.y = 8*precision/2;
    desc.z = 8*precision/2;
    desc.w = 8*precision/2;
  }

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = d_elem;
  resDesc.res.linear.desc = desc;
  resDesc.res.linear.sizeInBytes = totalLength_device_bytes;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);
  CHECK_CUDA_ERROR();
}

template<typename Float>
void Base<Float>::destroyTexObject(cudaTextureObject_t tex){
  cudaDestroyTextureObject(tex);
  isTextureOn=false;
}



template<typename Float>
void Base<Float>::createTexObjectExtra(cudaTextureObject_t *tex){

  if(!isAllocDeviceExtra)
    ABORT("Error: Create texture object for memory not allocated is not allowed\n");
  if(isTextureExtraOn)
    ABORT("Error: Textures object is already on\n");
  
  cudaChannelFormatDesc desc;
  memset(&desc, 0, sizeof(cudaChannelFormatDesc));
  int precision = Precision();
  if(precision == sizeof(float)) desc.f = cudaChannelFormatKindFloat;
  else desc.f = cudaChannelFormatKindSigned;

  if(precision == sizeof(float)){
    desc.x = 8*precision;
    desc.y = 8*precision;
    desc.z = 0;
    desc.w = 0;
  }
  else if(precision == sizeof(double)){
    desc.x = 8*precision/2;
    desc.y = 8*precision/2;
    desc.z = 8*precision/2;
    desc.w = 8*precision/2;
  }

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = d_elem_extra;
  resDesc.res.linear.desc = desc;
  resDesc.res.linear.sizeInBytes = totalLength_device_bytes;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(tex, &resDesc, &texDesc, NULL);
  CHECK_CUDA_ERROR();
}

template<typename Float>
void Base<Float>::destroyTexObjectExtra(cudaTextureObject_t tex){
  cudaDestroyTextureObject(tex);
  isTextureExtraOn=false;
}


template<typename Float>
void Base<Float>::copyToDevice(){

  cudaMemcpy(d_elem, h_elem, totalLength_device_bytes,cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR();
}

// =========================================================================== //
// ================================= Mom_matrices =========================== //
template<typename Float>
Mom_matrices<Float>::Mom_matrices(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT): 
  Base<Float>(alloc_flag, classT)
{
  ;
}


// ========================================================================= //

// ================================= Mom_matrices_x2 =========================== //
template<typename Float>
Mom_matrices_x2<Float>::Mom_matrices_x2(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT): 
  Base<Float>(alloc_flag, classT)
{
  ;
}


// ========================================================================= //

// ================================= Mom_matrix =========================== //
template<typename Float>
Mom_matrix<Float>::Mom_matrix(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT): 
  Base<Float>(alloc_flag, classT)
{
  ;
}


// ========================================================================= //

// ============================== LapH_prop ================================ //
template<typename Float>
LapH_prop<Float>::LapH_prop(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT): 
  Base<Float>(alloc_flag, classT) , isRefHost(false)
{
  ;
}

template<typename Float>
LapH_prop<Float>::LapH_prop(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, Float* h_elem_ref):
  Base<Float>(alloc_flag, classT)
{
  if( (alloc_flag == HOST) || (alloc_flag == BOTH_EXTRA)  )
    ABORT("The overloaded constructor for LapH_prop does not accept this alloc_flag\n");
  CC::h_elem = h_elem_ref; // points at the same point of the input pointer
  isRefHost = true;
}


template<typename Float>
void LapH_prop<Float>::readPropH(const std::string& fileName){
  if(!CC::isAllocHost) ABORT("Error: Try read element without allocate memory\n");

  FILE* f = fopen(fileName.c_str(), "rb");
  if(f == NULL)
    ABORT("Error: Cannot find file for reading propagatorH\n");

  size_t checkBytes;
  char cfgname[100];
  checkBytes = fread(cfgname, 1, 100, f); cfgname[99] = 0;
  std::cout << "Got configuration ID from file: " << cfgname << std::endl;
  //  latname = std::string(cfgname);

  uint32_t nt, nvec;
  checkBytes = fread(&nt, 4, 1, f);
  checkBytes = fread(&nvec, 4, 1, f);

  //  printf("(%d,%d) , (%d,%d)\n",nvec,G_NeV,nt,G_Nt);

  if( (int)nvec != G_NeV )
    ABORT("Error: NeV given is not equal with what I get from file\n");
  if( (int)nt != G_Nt )
    ABORT("Error: Nt given is not equal with what I get from file\n");

  void *buffer = malloc(CC::totalLength_host*2*sizeof(double));
  if(buffer == NULL) ABORT("Error allocating memory for buffer reading propH\n");

  //  m.nt = nt; m.nvec = nvec; m.vals.resize(nt*nvec*4*nt*nvec*4);
  checkBytes = fread(buffer, 2*sizeof(double), CC::totalLength_host, f);
  if((int)checkBytes != CC::totalLength_host) 
    ABORT("Error: Did not read size expected\n");

  // <t1,v1,s1|Mtilde|t2,v2,s2> = Mtilde(t1,t2)^{v1,v2}_{s1,s2} = m(t2, v2, s2, t1, v1, s1)
  //  t1+nt*(v1+nvec*(s1+4*(t2+nt*(v2+nvec*s2))))

  for(int t1 = 0 ; t1 < G_Nt ; t1++)
    for(int t2 = 0 ; t2 < G_Nt ; t2++)
      for(int s1 = 0 ; s1 < 4 ; s1++)
	for(int s2 = 0 ; s2 < 4 ; s2++)
	  for(int v1 = 0 ; v1 < G_NeV ; v1++)
	    for(int v2 = 0 ; v2 < G_NeV ; v2++)
	      for(int ir = 0 ; ir < 2 ; ir++){
		//		CC::h_elem[(t1*G_Nt*4*4*G_NeV*G_NeV + t2*4*4*G_NeV*G_NeV + s1*4*G_NeV*G_NeV + s2*G_NeV*G_NeV + v1*G_NeV + v2)*2 + ir] = 
		//  (Float) ((double*)buffer)[(s2*G_NeV*G_Nt*4*G_NeV*G_Nt + v2*G_Nt*4*G_NeV*G_Nt + t2*4*G_NeV*G_Nt + s1*G_NeV*G_Nt + v1*G_Nt + t1)*2 + ir];
		CC::h_elem[(t1*G_Nt*4*4*G_NeV*G_NeV + t2*4*4*G_NeV*G_NeV + s1*4*G_NeV*G_NeV + s2*G_NeV*G_NeV + v1*G_NeV + v2)*2 + ir] = 
		  (Float) ((double*)buffer)[(s1*G_NeV*G_Nt*4*G_NeV*G_Nt + v1*G_Nt*4*G_NeV*G_Nt + t1*4*G_NeV*G_Nt + s2*G_NeV*G_Nt + v2*G_Nt + t2)*2 + ir];
	      }

  free(buffer);
  fclose(f);
}

template<typename Float>
void LapH_prop<Float>::copyToDeviceRowByRow(int it){
  if(!CC::isAllocHost && !isRefHost)
    ABORT("Error: Cannot copy from host to device if host is not already allocated or has a refernence pointer\n");  
  cudaMemcpy(CC::d_elem, CC::h_elem + it*CC::totalLength_device*2, CC::totalLength_device_bytes,cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR();
}

template<typename Float>
void LapH_prop<Float>::copyToDeviceExtraDiag(){
  if(!CC::isAllocHost && !isRefHost)
    ABORT("Error: Cannot copy from host to device if host is not already allocated or has a refernence pointer\n");
  if(!CC::isAllocDeviceExtra)
    ABORT("Error: Cannot copy to device extra is not allocated\n");

  Float *tmp = (Float*) malloc(CC::totalLength_device_bytes);
  if(tmp==NULL) ABORT("Error: Cannot allocate resources on host\n");

  for(int it = 0 ; it < G_Nt ; it++)
    for(int s1 = 0 ; s1 < 4 ; s1++)
      for(int s2 = 0 ; s2 < 4 ; s2++)
	for(int v1 = 0 ; v1 < G_NeV ; v1++)
	  for(int v2 = 0 ; v2 < G_NeV ; v2++)
	    for(int ir = 0 ; ir < 2 ; ir++)
	      tmp[(it*4*4*G_NeV*G_NeV + s1*4*G_NeV*G_NeV + s2*G_NeV*G_NeV + v1*G_NeV + v2)*2+ir] = CC::h_elem[(it*G_Nt*4*4*G_NeV*G_NeV + it*4*4*G_NeV*G_NeV + s1*4*G_NeV*G_NeV + s2*G_NeV*G_NeV + v1*G_NeV + v2)*2+ir]; // get only the diagonal elements in time
  
  cudaMemcpy(CC::d_elem_extra, tmp, CC::totalLength_device_bytes, cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR();
  free(tmp);
}
// ========================================================================= //
// ========================== Contract_rho_rho ============================= //

template<typename Float>
Contract_rho_rho<Float>::Contract_rho_rho(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile): 
  Base<Float>(alloc_flag, classT)
{
  pf = fopen(outfile.c_str(), "a");
  if(pf == NULL)
    ABORT("Error cannot open file for writting results\n");
}

template<typename Float>
void Contract_rho_rho<Float>::copyConstantsRho(){
  run_CopyConstantsRho();
}

template<typename Float>
void Contract_rho_rho<Float>::contractRho(Mom_matrices<Float> &momMatrices, LapH_prop<Float> &propH){

  cudaTextureObject_t momTex;
  cudaTextureObject_t propTex;

  momMatrices.copyToDevice();        // mom matrices copy data to device

  copyConstantsRho(); // first we need to copy the constants to device
  momMatrices.createTexObject(&momTex);
  propH.createTexObject(&propTex);

  PRECISION prec = sizeof(Float) == sizeof(float) ? SINGLE : DOUBLE;

  for(int it = 0 ; it < G_Nt ; it++){
    propH.copyToDeviceRowByRow(it);  // copy prop to device row by row
    run_ContractRho(propTex, momTex, it, G_Nt,(void*) (CC::h_elem+it*G_Nt * 16 * 3 * 2), prec);
  }

  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();

  dumpData();

  momMatrices.destroyTexObject(momTex);
  propH.destroyTexObject(propTex);
}

template<typename Float>
void Contract_rho_rho<Float>::dumpData(){

  Float *corr = (Float*) calloc(G_Nt*16*3*2,sizeof(Float));

  for(int dt = 0 ; dt < G_Nt ; dt++)
      for(int iop = 0 ; iop < 16 ; iop++)
	for(int igi = 0 ; igi < 3 ; igi++){
	  for(int it = 0 ; it < G_Nt ; it++){
	    int tf = (it+dt)%G_Nt;
	    corr[(dt*16*3+iop*3+igi)*2+0] += CC::h_elem[(tf*G_Nt*16*3 + it*16*3 + iop*3 + igi)*2+0];
	    corr[(dt*16*3+iop*3+igi)*2+1] += CC::h_elem[(tf*G_Nt*16*3 + it*16*3 + iop*3 + igi)*2+1];
	  }
	  corr[(dt*16*3+iop*3+igi)*2+0] /= G_Nt;
	  corr[(dt*16*3+iop*3+igi)*2+1] /= G_Nt;
	}

  for(int iop = 0 ; iop < 16 ; iop++){
    for(int it = 0 ; it < G_Nt ; it++){
      fprintf(pf,"%d %d \t %+e %+e \t %+e %+e \t %+e %+e\n",iop, it, corr[(it*16*3+iop*3+0)*2+0], corr[(it*16*3+iop*3+0)*2+1], corr[(it*16*3+iop*3+1)*2+0], corr[(it*16*3+iop*3+1)*2+1], corr[(it*16*3+iop*3+2)*2+0], corr[(it*16*3+iop*3+2)*2+1]);
    }
  }
  fprintf(pf,"\n");
  fflush(pf);

  free(corr);

}
// ======================================================================== //

//========================== Contract_rho_pipi ============================ //
template<typename Float>
Contract_rho_pipi<Float>::Contract_rho_pipi(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile): 
  Base<Float>(alloc_flag, classT)
{
  pf = fopen(outfile.c_str(), "a");
  if(pf == NULL)
    ABORT("Error cannot open file for writting results\n");

}

template<typename Float>
void Contract_rho_pipi<Float>::copyConstantsRhoPi(){
  run_CopyConstantsRhoPi();
}


template<typename Float>
void Contract_rho_pipi<Float>::contractRhoPi(int idir, Mom_matrix<Float> &momP1, Mom_matrix<Float> &momP2, Mom_matrices<Float> &momP1P2, LapH_prop<Float> &propH){

  cudaTextureObject_t texMomP1;
  cudaTextureObject_t texMomP2;
  cudaTextureObject_t texMomP1P2;
  cudaTextureObject_t texProp;
  cudaTextureObject_t texPropDiag;

  copyConstantsRhoPi();

  momP1.copyToDevice();
  momP2.copyToDevice();
  momP1P2.copyToDevice();
  propH.copyToDeviceExtraDiag();

  momP1.createTexObject(&texMomP1);
  momP2.createTexObject(&texMomP2);
  momP1P2.createTexObject(&texMomP1P2);
  propH.createTexObject(&texProp);
  propH.createTexObjectExtra(&texPropDiag);

  PRECISION prec = sizeof(Float) == sizeof(float) ? SINGLE : DOUBLE;
  
  for(int it = 0 ; it < G_Nt ; it++){
    propH.copyToDeviceRowByRow(it);  // copy prop to device row by row
    run_ContractRhoPi(texProp, texPropDiag,texMomP1, texMomP2, texMomP1P2, it, G_Nt, idir, (void*) (CC::h_elem+it*G_Nt * 4 * 2), prec);
  }

  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();

  dumpData();

  momP1.destroyTexObject(texMomP1);
  momP2.destroyTexObject(texMomP2);
  momP1P2.destroyTexObject(texMomP1P2);
  propH.destroyTexObject(texProp);
  propH.destroyTexObjectExtra(texPropDiag);
}

template<typename Float>
void Contract_rho_pipi<Float>::dumpData(){

  Float *corr = (Float*) calloc(G_Nt*4*2,sizeof(Float));

  for(int dt = 0 ; dt < G_Nt ; dt++)
    for(int iop = 0 ; iop < 4 ; iop++){
      for(int it = 0 ; it < G_Nt ; it++){
	int tf = (it+dt)%G_Nt;
	corr[(dt*4+iop)*2+0] += CC::h_elem[(tf*G_Nt*4 + it*4 + iop)*2+0];
	corr[(dt*4+iop)*2+1] += CC::h_elem[(tf*G_Nt*4 + it*4 + iop)*2+1];
      }
      corr[(dt*4+iop)*2+0] /= G_Nt;
      corr[(dt*4+iop)*2+1] /= G_Nt;
    }

  for(int iop = 0 ; iop < 4 ; iop++)
    for(int it = 0 ; it < G_Nt ; it++)
      fprintf(pf,"%d %d %d \t %+e %+e\n",iop,i_dir,it,2*corr[(it*4+iop)*2+0], 2*corr[(it*4+iop)*2+1]);
    
  fprintf(pf,"\n");
  fflush(pf);
  free(corr);

}
// ======================================================================== //

//========================== Contract_pipi_pipi ============================ //
template<typename Float>
Contract_pipi_pipi<Float>::Contract_pipi_pipi(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile): Base<Float>(alloc_flag, classT)
{
  pf = fopen(outfile.c_str(), "a");
  if(pf == NULL)
    ABORT("Error cannot open file for writting results\n");
}

template<typename Float>
void Contract_pipi_pipi<Float>::copyConstantsPiPi(){
  run_CopyConstantsPiPi();
}



template<typename Float>
void Contract_pipi_pipi<Float>::contractPiPi(Mom_matrix<Float> &momP1, Mom_matrix<Float> &momP2, Mom_matrix<Float> &momP3, Mom_matrix<Float> &momP4, LapH_prop<Float> &propH, ISOSPIN Iso){

  cudaTextureObject_t texMomP1;
  cudaTextureObject_t texMomP2;
  cudaTextureObject_t texMomP3;
  cudaTextureObject_t texMomP4;
  cudaTextureObject_t texProp;
  cudaTextureObject_t texPropDiag;

  copyConstantsPiPi();

  momP1.copyToDevice();
  momP2.copyToDevice();
  momP3.copyToDevice();
  momP4.copyToDevice();
  propH.copyToDeviceExtraDiag();

  momP1.createTexObject(&texMomP1);
  momP2.createTexObject(&texMomP2);
  momP3.createTexObject(&texMomP3);
  momP4.createTexObject(&texMomP4);

  propH.createTexObject(&texProp);
  propH.createTexObjectExtra(&texPropDiag);

  PRECISION prec = sizeof(Float) == sizeof(float) ? SINGLE : DOUBLE;

  for(int it = 0 ; it < G_Nt ; it++){
    propH.copyToDeviceRowByRow(it);  // copy prop to device row by row
    if(Iso == I1)run_ContractPiPi(texProp, texPropDiag,texMomP1, texMomP2, texMomP3, texMomP4, it, G_Nt, (void*) (CC::h_elem+it*G_Nt*5*2), prec);
    if(Iso == I0)run_ContractPiPi_I0(texProp, texPropDiag,texMomP1, texMomP2, texMomP3, texMomP4, it, G_Nt, (void*) (CC::h_elem+it*G_Nt*5*2), prec);
  }

  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();

  dumpData(Iso);

  momP1.destroyTexObject(texMomP1);
  momP2.destroyTexObject(texMomP2);
  momP3.destroyTexObject(texMomP3);
  momP4.destroyTexObject(texMomP4);

  propH.destroyTexObject(texProp);
  propH.destroyTexObjectExtra(texPropDiag);
}

template<typename Float>
void Contract_pipi_pipi<Float>::dumpData(ISOSPIN Iso){

  Float *corr = (Float*) calloc(G_Nt*5*2,sizeof(Float));

  for(int dt = 0 ; dt < G_Nt ; dt++){
    for(int it = 0 ; it < G_Nt ; it++){
      int tf = (it+dt)%G_Nt;
      for(int i = 0 ; i < 5 ; i++){
	corr[dt*5*2+i*2+0] += CC::h_elem[(tf*G_Nt*5 + it*5 + i)*2+0];
	corr[dt*5*2+i*2+1] += CC::h_elem[(tf*G_Nt*5 + it*5 + i)*2+1];
      }
    }
    for(int i = 0 ; i < 5 ; i++){
      corr[dt*5*2+i*2+0] /= G_Nt;
      corr[dt*5*2+i*2+1] /= G_Nt;
    }
  }


  for(int it = 0 ; it < G_Nt ; it++)
    fprintf(pf,"Square: %d \t %+e %+e\n",it,corr[it*5*2+0*2+0], corr[it*5*2+0*2+1]);
  fprintf(pf,"\n");

  for(int it = 0 ; it < G_Nt ; it++)
    fprintf(pf,"DoubleTriangle: %d \t %+e %+e\n",it,corr[it*5*2+1*2+0], corr[it*5*2+1*2+1]);
  fprintf(pf,"\n");

  for(int it = 0 ; it < G_Nt ; it++)
    fprintf(pf,"Star: %d \t %+e %+e\n",it,corr[it*5*2+2*2+0], corr[it*5*2+2*2+1]);
  fprintf(pf,"\n");

  for(int it = 0 ; it < G_Nt ; it++)
    fprintf(pf,"Fish: %d \t %+e %+e\n",it,corr[it*5*2+3*2+0], corr[it*5*2+3*2+1]);
  fprintf(pf,"\n");

  if(Iso == I0){
    for(int it = 0 ; it < G_Nt ; it++)
      fprintf(pf,"DoubleTriangle_Hor: %d \t %+e %+e\n",it,corr[it*5*2+4*2+0], corr[it*5*2+4*2+1]);
    fprintf(pf,"\n");
  }

  fflush(pf);
  free(corr);
}

//=====================================================================// 

//==================== Contract loop Unity ============================//
template<typename Float>
Contract_loop_unity<Float>::Contract_loop_unity(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile): Base<Float>(alloc_flag, classT)
{
  pf = fopen(outfile.c_str(), "a");
  if(pf == NULL)
    ABORT("Error cannot open file for writting results\n");
}

template<typename Float>
void Contract_loop_unity<Float>::contractLoopUnity(Mom_matrix<Float> &mom, LapH_prop<Float> &propH){
  cudaTextureObject_t texMom;
  cudaTextureObject_t texPropDiag;
  mom.copyToDevice();
  mom.createTexObject(&texMom);
  propH.copyToDeviceExtraDiag();
  propH.createTexObjectExtra(&texPropDiag);

  PRECISION prec = sizeof(Float) == sizeof(float) ? SINGLE : DOUBLE;
  run_ContractLoopUnity(texPropDiag, texMom, G_Nt, (void*) (CC::h_elem) , prec);

  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();
  dumpData();

  mom.destroyTexObject(texMom);
  propH.destroyTexObjectExtra(texPropDiag);
}

template<typename Float>
void Contract_loop_unity<Float>::dumpData(){

  for(int it = 0 ; it < G_Nt ; it++)
    fprintf(pf,"loop: %d \t %+e %+e\n",it,CC::h_elem[it*2+0], CC::h_elem[it*2+1]);
  fprintf(pf,"\n");
  fflush(pf);
}

// ========================== Contract_sigma_sigma ============================= //
template<typename Float>
Contract_sigma_sigma<Float>::Contract_sigma_sigma(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile): 
  Base<Float>(alloc_flag, classT)
{
  pf = fopen(outfile.c_str(), "a");
  if(pf == NULL)
    ABORT("Error cannot open file for writting results\n");
}

template<typename Float>
void Contract_sigma_sigma<Float>::copyConstantsSigma(){
  run_CopyConstantsSigma();
}

template<typename Float>
void Contract_sigma_sigma<Float>::contractSigma(Mom_matrices_x2<Float> &momMatrices_x2, LapH_prop<Float> &propH){

  cudaTextureObject_t momTex;
  cudaTextureObject_t propTex;

  momMatrices_x2.copyToDevice();        // mom matrices copy data to device

  copyConstantsSigma(); // first we need to copy the constants to device
  momMatrices_x2.createTexObject(&momTex);
  propH.createTexObject(&propTex);

  PRECISION prec = sizeof(Float) == sizeof(float) ? SINGLE : DOUBLE;

  for(int it = 0 ; it < G_Nt ; it++){
    propH.copyToDeviceRowByRow(it);  // copy prop to device row by row
    run_ContractSigma(propTex, momTex, it, G_Nt,(void*) (CC::h_elem+it*G_Nt * 4 * 2), prec);
  }

  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();

  dumpData();

  momMatrices_x2.destroyTexObject(momTex);
  propH.destroyTexObject(propTex);
}

template<typename Float>
void Contract_sigma_sigma<Float>::dumpData(){

  Float *corr = (Float*) calloc(G_Nt*4*2,sizeof(Float));

  for(int dt = 0 ; dt < G_Nt ; dt++)
    for(int iop = 0 ; iop < 4 ; iop++){
      for(int it = 0 ; it < G_Nt ; it++){
	int tf = (it+dt)%G_Nt;
	corr[(dt*4+iop)*2+0] += CC::h_elem[(tf*G_Nt*4 + it*4 + iop)*2+0];
	corr[(dt*4+iop)*2+1] += CC::h_elem[(tf*G_Nt*4 + it*4 + iop)*2+1];
      }
      corr[(dt*4+iop)*2+0] /= G_Nt;
      corr[(dt*4+iop)*2+1] /= G_Nt;
    }

  for(int iop = 0 ; iop < 4 ; iop++){
    for(int it = 0 ; it < G_Nt ; it++){
      fprintf(pf,"twop: %d %d \t %+e %+e\n",iop, it, -corr[(it*4+iop)*2+0], -corr[(it*4+iop)*2+1]);
    }
  }
  fprintf(pf,"\n");

  fflush(pf);
  free(corr);

}
// ======================================================================== //

// ========================== Contract_sigma_pipi ============================= //
template<typename Float>
Contract_sigma_pipi<Float>::Contract_sigma_pipi(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT, std::string outfile): 
  Base<Float>(alloc_flag, classT)
{
  pf = fopen(outfile.c_str(), "a");
  if(pf == NULL)
    ABORT("Error cannot open file for writting results\n");
}

template<typename Float>
void Contract_sigma_pipi<Float>::copyConstantsSigmaPi(){
  run_CopyConstantsSigmaPi();
}


template<typename Float>
void Contract_sigma_pipi<Float>::contractSigmaPi(Mom_matrix<Float> &momP1, Mom_matrix<Float> &momP2, Mom_matrices_x2<Float> &momP1P2, LapH_prop<Float> &propH){

  cudaTextureObject_t texMomP1;
  cudaTextureObject_t texMomP2;
  cudaTextureObject_t texMomP1P2;
  cudaTextureObject_t texProp;
  cudaTextureObject_t texPropDiag;

  copyConstantsSigmaPi();

  momP1.copyToDevice();
  momP2.copyToDevice();
  momP1P2.copyToDevice();
  propH.copyToDeviceExtraDiag();

  momP1.createTexObject(&texMomP1);
  momP2.createTexObject(&texMomP2);
  momP1P2.createTexObject(&texMomP1P2);
  propH.createTexObject(&texProp);
  propH.createTexObjectExtra(&texPropDiag);

  PRECISION prec = sizeof(Float) == sizeof(float) ? SINGLE : DOUBLE;
  
  // run here the fish hor

  run_ContractSigmaPi_fish_hor(texPropDiag,texMomP1, texMomP2, G_Nt, (void*) (CC::h_elem), prec);
  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();
  dumpData_fish_hor();

  for(int it = 0 ; it < G_Nt ; it++){
    propH.copyToDeviceRowByRow(it);  // copy prop to device row by row
    run_ContractSigmaPi_triangle(texProp, texPropDiag,texMomP1, texMomP2, texMomP1P2, it, G_Nt, (void*) (CC::h_elem+G_Nt*2+it*G_Nt * 2 * 2), prec);
  }

  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR();

  dumpData_triangle();

  momP1.destroyTexObject(texMomP1);
  momP2.destroyTexObject(texMomP2);
  momP1P2.destroyTexObject(texMomP1P2);
  propH.destroyTexObject(texProp);
  propH.destroyTexObjectExtra(texPropDiag);
}



template<typename Float>
void Contract_sigma_pipi<Float>::dumpData_fish_hor(){
  for(int it = 0 ; it < G_Nt ; it++)
    fprintf(pf,"fish_hor: %d \t %+e %+e\n",it,CC::h_elem[it*2+0], CC::h_elem[it*2+1]);
  fprintf(pf,"\n");
  fflush(pf);
}

template<typename Float>
void Contract_sigma_pipi<Float>::dumpData_triangle(){

  Float *corr = (Float*) calloc(G_Nt*2*2,sizeof(Float));

  for(int dt = 0 ; dt < G_Nt ; dt++)
    for(int iop = 0 ; iop < 2 ; iop++){
      for(int it = 0 ; it < G_Nt ; it++){
	int tf = (it+dt)%G_Nt;
	corr[(dt*2+iop)*2+0] += CC::h_elem[(G_Nt + tf*G_Nt*2 + it*2 + iop)*2+0]; // here we need an offset of G_Nt*2 to reach the triangle correlator
	corr[(dt*2+iop)*2+1] += CC::h_elem[(G_Nt + tf*G_Nt*2 + it*2 + iop)*2+1];
      }
      corr[(dt*2+iop)*2+0] /= G_Nt;
      corr[(dt*2+iop)*2+1] /= G_Nt;
    }

  for(int iop = 0 ; iop < 2 ; iop++){
    for(int it = 0 ; it < G_Nt ; it++){
      fprintf(pf,"triangle: %d %d \t %+e %+e\n",iop, it, (-2./sqrt(6.))*corr[(it*2+iop)*2+0], (-2./sqrt(6.))*corr[(it*2+iop)*2+1]);
    }
  }
  fprintf(pf,"\n");

  fflush(pf);
  free(corr);

}
// ======================================================================== //


