#include <catchOpts.h>

//using namespace contract;

//using contract::latname;
  // default values
namespace contract{
 
  std::string latname = "";
  std::string eigename = ""; 
  std::string propname = "";
  std::string momname = "";
  std::string outname = "out.dat";

  PRECISION Prec = DOUBLE;

  int devID = 0;
  int Nthreads = 1;
  int Nx = 16;
  int Ny = 16;
  int Nz = 16;
  int Nt = 32;
  int Nvec = 100;
  int Nmom = 1;
  int momComb = 1;
  int i_dir = 0;       // idir is needed in the case of rhopi to choose gamma_i direction

  int *px1=NULL;
  int *px2=NULL;
  int *px3=NULL;
  int *px4=NULL;

  int *py1=NULL;
  int *py2=NULL;
  int *py3=NULL;
  int *py4=NULL;

  int *pz1=NULL;
  int *pz2=NULL;
  int *pz3=NULL;
  int *pz4=NULL;
 

  //==================================================//
  static void printOptions(){
    printf("Available options:\n");
    printf("     --prec <double/single>                       # Contraction precision (default: double)\n");
    printf("     --devID <0,1,...>                            # Set the device that will be used for contractions (default: 0)\n");
    printf("     --Nthreads <n>                               # Number of cpu OMP threads that will be used to calculate the momentum matrices (default: 1)\n");
    printf("     --Nx <n>                                     # Lattice extend in the x-direction (default: 16)\n");
    printf("     --Ny <n>                                     # Lattice extend in the y-direction (default: 16)\n");
    printf("     --Nz <n>                                     # Lattice extend in the z-direction (default: 16)\n");
    printf("     --Nt <n>                                     # Lattice extend in the t-direction (default: 32)\n");
    printf("     --Nvec <n>                                   # Number of Eigenvectors for the LapH method (default: 100, hardcoded cannot change)\n");
    printf("     --Nmom <1,2,4>                               # Number of Momenta (rho->rho #1, rho->pipi #2, pipi->pipi #4) (default: 1, for rho->rho only)\n");
    printf("     --momComb <n>                                # Number of Momentum combinations (default: 1)\n");
    printf("     --idir <0,1,2>                               # Needed to choose direction of gamma_i in compute rho->pipi (default: 0, -> gamma_1)\n");
    printf("     --latname <path>                             # Path to the configuration (default: )\n");
    printf("     --eigename <path>                            # Path to the laplacian eigenvectors (default: )\n");
    printf("     --propname <path>                            # Path to the propagator (default: )\n");
    printf("     --momname <path>                             # Path to the momentum list(ascii format)  (default: )\n");
    printf("     --outname <path>                             # Path to the output file(ascii format) (default: out.dat)\n");
    fflush(stdout);
  }
  //==================================================//
  static bool isExist(const std::string& name){
    struct stat buffer;   
    return (stat (name.c_str(), &buffer) == 0); 
  }
  
  //======================================================//
  static void printSetValues(){
    printf("Got latname: %s\n",latname.c_str());
    printf("Got eigename: %s\n",eigename.c_str());
    printf("Got propname: %s\n",propname.c_str());
    printf("Got momname: %s\n",momname.c_str());
    printf("Got output name: %s\n", outname.c_str());

    if(Prec == SINGLE)
      printf("Contractions will be calculated in single precision\n");
    else
      printf("Contractions will be calculated in double precision\n");
    printf("Device ID to be used is %d\n",devID);
    printf("Number of OMP threads is %d\n",Nthreads);
    printf("Lattice size is (Nx,Ny,Nz,Nt) -> (%d,%d,%d,%d)\n",Nx,Ny,Nz,Nt);
    printf("Laplacian eigenvectors are %d\n",Nvec);
    printf("idir is used only in the case of rho->pipi and its value is %d\n",i_dir);
    printf("Number of momenta is %d\n",Nmom);
    printf("Number of momentum combinations is %d\n",momComb);
    printf("Printing all momentum combinations\n");
    switch(Nmom){
    case 1:
      for(int i = 0 ; i < momComb ; i++)
	printf("%+d %+d %+d\n",px1[i],py1[i],pz1[i]);
      break;
    case 2:
      for(int i = 0 ; i < momComb ; i++)
	printf("%+d %+d %+d \t %+d %+d %+d \t %+d %+d %+d\n",px1[i],py1[i],pz1[i], px2[i],py2[i],pz2[i], px3[i],py3[i],pz3[i]);
      break;
    case 4:
      for(int i = 0 ; i < momComb ; i++)
	printf("%+d %+d %+d \t %+d %+d %+d \t %+d %+d %+d \t %+d %+d %+d\n",px1[i],py1[i],pz1[i], px2[i],py2[i],pz2[i], px3[i],py3[i],pz3[i], px4[i],py4[i],pz4[i]);
      break;
    }
    fflush(stdout);
  }
  //=====================================================//
  static void readMomenta(const std::string& name){
    int nl = 0;
    int nc = 0;
    std::string line;
    std::fstream fp;
    fp.open(name.c_str(), std::fstream::in);

    while(!fp.eof()) {
      getline (fp, line);
      nl++;
    }
    if(line != ""){
      WARNING("No new line at the end of the file, number of lines may counted wrongly\n");
    }
    else
      nl--; //remove the new line at the end of the file

    fp.clear();
    fp.seekp(0);
    getline(fp,line);

    std::stringstream is(line);
    int dummy;
    while(is >> dummy)
      nc++;

    int nl_expect = momComb;
    int nc_expect = Nmom*3; // three directions on the lattice

    if(nl != nl_expect)
      ABORT("Number of lines in the file are not in agreement with the number of momentum combinations you passed\n");

    if(nc != nc_expect)
      ABORT("Number of columns in the file are not in agreement with the number of momenta you passed\n");

    switch(Nmom){
    case 1:
      px1=(int*) malloc(momComb*sizeof(int));
      py1=(int*) malloc(momComb*sizeof(int));
      pz1=(int*) malloc(momComb*sizeof(int));
      break;
    case 2:
      px1=(int*) malloc(momComb*sizeof(int));
      py1=(int*) malloc(momComb*sizeof(int));
      pz1=(int*) malloc(momComb*sizeof(int));

      px2=(int*) malloc(momComb*sizeof(int));
      py2=(int*) malloc(momComb*sizeof(int));
      pz2=(int*) malloc(momComb*sizeof(int));

      px3=(int*) malloc(momComb*sizeof(int));
      py3=(int*) malloc(momComb*sizeof(int));
      pz3=(int*) malloc(momComb*sizeof(int));
      break;
    case 4:
      px1=(int*) malloc(momComb*sizeof(int));
      py1=(int*) malloc(momComb*sizeof(int));
      pz1=(int*) malloc(momComb*sizeof(int));

      px2=(int*) malloc(momComb*sizeof(int));
      py2=(int*) malloc(momComb*sizeof(int));
      pz2=(int*) malloc(momComb*sizeof(int));

      px3=(int*) malloc(momComb*sizeof(int));
      py3=(int*) malloc(momComb*sizeof(int));
      pz3=(int*) malloc(momComb*sizeof(int));

      px4=(int*) malloc(momComb*sizeof(int));
      py4=(int*) malloc(momComb*sizeof(int));
      pz4=(int*) malloc(momComb*sizeof(int));
      break;
    default:
      ABORT("Only 1,2,4 momenta are allowed\n");
      break;
    }

    fp.clear();
    fp.seekp(0);


    // reading momenta from the file
    switch(Nmom){
    case 1:
      for(int i = 0 ; i < nl_expect ; i++){
	getline(fp,line);
	std::stringstream is(line);
	is >> px1[i];
	is >> py1[i];
	is >> pz1[i];
      }
      break;
    case 2:
      for(int i = 0 ; i < nl_expect ; i++){
	getline(fp,line);
	std::stringstream is(line);
	is >> px1[i];
	is >> py1[i];
	is >> pz1[i];
	is >> px2[i];
	is >> py2[i];
	is >> pz2[i];
	px3[i] = px1[i] + px2[i];
	py3[i] = py1[i] + py2[i];
	pz3[i] = pz1[i] + pz2[i];
      }
      break;
    case 4:
      for(int i = 0 ; i < nl_expect ; i++){
	getline(fp,line);
	std::stringstream is(line);
	is >> px1[i];
	is >> py1[i];
	is >> pz1[i];
	is >> px2[i];
	is >> py2[i];
	is >> pz2[i];

	is >> px3[i];
	is >> py3[i];
	is >> pz3[i];
	is >> px4[i];
	is >> py4[i];
	is >> pz4[i];
      }
      break;
    }

  }
  //====================================================//
void catchOpts(int argc, char **argv){
    if(argc%2 != 1){
      printOptions();
      ABORT("An option is missing a value or you called help\n");
    }
    for(int i = 1 ; i < argc ; i++){
      //+++++++++++++
      if( strcmp(argv[i], "--prec") ==0){
	if( strcmp(argv[i+1], "double") == 0)
	  Prec = DOUBLE;
	else if( strcmp(argv[i+1], "single") == 0)
	  Prec = SINGLE;
	else{
	  printOptions();
	  ABORT("Unkown precision\n");
	}
	i++;
	continue;
      }      
      //++++++++++++
      if( strcmp(argv[i], "--devID") ==0){
	devID = atoi(argv[i+1]);
	i++;
	continue;
      }
      //++++++++++++
      if( strcmp(argv[i], "--Nthreads") ==0){
	Nthreads = atoi(argv[i+1]);
	if(Nthreads < 1)
	  ABORT("OMP threads must be at least 1\n");
	i++;
	continue;
      }
      //++++++++++++
      if( strcmp(argv[i], "--Nx") ==0){
	Nx = atoi(argv[i+1]);
	i++;
	continue;
      }
      //++++++++++++
      if( strcmp(argv[i], "--Ny") ==0){
	Ny = atoi(argv[i+1]);
	i++;
	continue;
      }
      //++++++++++++
      if( strcmp(argv[i], "--Nz") ==0){
	Nz = atoi(argv[i+1]);
	i++;
	continue;
      }
      //++++++++++++
      if( strcmp(argv[i], "--Nt") ==0){
	Nt = atoi(argv[i+1]);
	i++;
	continue;
      }
      //++++++++++++
      if( strcmp(argv[i], "--Nvec") ==0){
	Nvec = atoi(argv[i+1]);
	i++;
	continue;
      }
      //++++++++++++
      if( strcmp(argv[i], "--Nmom") ==0){
	Nmom = atoi(argv[i+1]);
	if(Nmom != 1 && Nmom != 2 && Nmom != 4){
	  printOptions();
	  ABORT("Nmom can only be 1,2 or 4\n");
	}
	i++;
	continue;
      }
      //++++++++++++
      if( strcmp(argv[i], "--momComb") ==0){
	momComb = atoi(argv[i+1]);
	if(momComb < 1){
	  printOptions();
	  ABORT("At least one momentum combination is needed\n");
	}
	i++;
	continue;
      }
      //++++++++++++
      if( strcmp(argv[i], "--idir") ==0){
	i_dir = atoi(argv[i+1]);
	if(i_dir != 0 && i_dir != 1 && i_dir != 2){
	  printOptions();
	  ABORT("idir can only be 0,1 or 2\n");
	}
	i++;
	continue;
      }
      //++++++++++
      if( strcmp(argv[i], "--latname") ==0){
	latname = argv[i+1];
	i++;
	continue;
      }
      //+++++++++
      if( strcmp(argv[i], "--eigename") ==0){
	eigename = argv[i+1];
	i++;
	continue;
      }
      //+++++++++
      if( strcmp(argv[i], "--propname") ==0){
	propname = argv[i+1];
	i++;
	continue;
      }
      //+++++++++
      if( strcmp(argv[i], "--momname") ==0){
	momname = argv[i+1];
	i++;
	continue;
      }
      //+++++++++
      if( strcmp(argv[i], "--outname") ==0){
	outname = argv[i+1];
	i++;
	continue;
      }
      //+++++++++
      if( strcmp(argv[i], "--help") ==0){
	printOptions();
	ABORT("\n");
      }
      //+++++++++
      printf("Unrecognized option %s\n",argv[i]);
      printOptions();
      ABORT("\n");
    }// close loop reading options
    if(!isExist(latname))
      ABORT("Check existence of latname\n");
    if(!isExist(eigename))
      ABORT("Check existence of eigename\n");
    if(!isExist(propname))
      ABORT("Check existence of propname\n");
    if(!isExist(momname))
      ABORT("Check existence of momname\n");
    if(isExist(outname))
      ABORT("Output file already exist, either remove it or choose another path\n");
      
    readMomenta(momname);
    printSetValues();
  }

void destroyOpts(){

    free(px1);
    free(px2);
    free(px3);
    free(px4);

    free(py1);
    free(py2);
    free(py3);
    free(py4);

    free(pz1);
    free(pz2);
    free(pz3);
    free(pz4);
    
    px1 = NULL;
    px2 = NULL;
    px3 = NULL;
    px4 = NULL;
    
    py1 = NULL;
    py2 = NULL;
    py3 = NULL;
    py4 = NULL;
    
    pz1 = NULL;
    pz2 = NULL;
    pz3 = NULL;
    pz4 = NULL;
    
}

}
