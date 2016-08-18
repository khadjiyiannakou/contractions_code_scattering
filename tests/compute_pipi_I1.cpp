#include <stdlib.h>
#include <vector>
#include <matrix.h>
#include <layout_minsurface.h>
#include <qcd_fields.h>
#include <smearing.h>
#include <boundary_flip.h>
#include <laplacean_eigensolver.h>
#include <lapmult.h>
#include <inject.h>

#include <interface.h>
#include <catchOpts.h>


using namespace qcd;

using contract::latname;
using contract::eigename;
using contract::propname;
using contract::momname;
using contract::outname;
using contract::Prec;
using contract::devID;
using contract::Nthreads;
using contract::Nx;
using contract::Ny;
using contract::Nz;
using contract::Nt;
using contract::Nvec;
using contract::Nmom;
using contract::momComb;
using contract::px1;
using contract::px2;
using contract::px3;
using contract::px4;
using contract::py1;
using contract::py2;
using contract::py3;
using contract::py4;
using contract::pz1;
using contract::pz2;
using contract::pz3;
using contract::pz4;

int main(int argc, char** argv){
  contract::catchOpts(argc,argv);

  if(Nmom != 4)
    ABORT("Error: compute_pipi needs only Nmom to be 4\n");

  omp_set_num_threads(Nthreads);

  init_machine(argc,argv);
  contract::contractInfo info;

  info.NeV=Nvec;
  info.Nt = Nt;
  info.dev=devID;
  info.PropH_fileName = propname;
  

  int nx,ny,nz,nt;

  double *tmpMomP1 = (double*) malloc(info.Nt*info.NeV*info.NeV*2*sizeof(double));
  double *tmpMomP2 = (double*) malloc(info.Nt*info.NeV*info.NeV*2*sizeof(double));
  double *tmpMomP3 = (double*) malloc(info.Nt*info.NeV*info.NeV*2*sizeof(double));
  double *tmpMomP4 = (double*) malloc(info.Nt*info.NeV*info.NeV*2*sizeof(double));

  /*
  int p1x, p1y, p1z;
  int p2x, p2y, p2z;

  int p3x, p3y, p3z;
  int p4x, p4y, p4z;
  */

  nx = Nx;
  ny = Ny;
  nz = Nz;
  nt = Nt;
  int nvec = Nvec;

  /*  
  p1x=1;p1y=0;p1z=0;
  p2x=-1;p2y=0;p2z=0;
  p3x=0;p3y=1;p3z=0;
  p4x=0;p4y=-1;p4z=0;
  


  p1x=0;p1y=0;p1z=0;
  p2x=0;p2y=0;p2z=0;
  p3x=0;p3y=0;p3z=0;
  p4x=0;p4y=0;p4z=-2;
  */

  /////////////////////////////// memory allocation /////////////
  std::vector<matrix>** matp;
  matp = new std::vector<matrix>*[4];
  for(int i=0; i<4; ++i)
    matp[i] = new std::vector<matrix>(nt);

  for(int t=0; t<nt; ++t) 
    for(int i=0; i<4; ++i)
      (*matp[i])[t].resize(nvec, nvec);

  //////////////////////// read data and smearing //////////////
  int this_node=get_node_rank();

  layout_minsurface_eo desc(nx,ny,nz,nt);

  su3_field links(&desc);


  read_kentucky_lattice(latname, links); // read the configuration
  nhyp_smear_simple(links, links);       // apply one step of hyp smearing

  int bc[4] = {1,1,1,-1};
  apply_boundary(links, bc);             // apply boundary conditions

  laplacean_eigensolver lapeig(links);
  lapeig.allocate_eigensystem(info.NeV);
  read_laplace_eigensystem(eigename, lapeig);

  contract::initContract(info);
  void *ptr_prop = readPropH(propname, Prec); // read the propagator

  //////////////////////////////////////////////////////////////
  for(int ip = 0 ; ip < momComb ; ip++){
    printf("Starting momentum %d\n",ip);

    for(int t=0; t<nt; ++t) 
      for(int c=0; c<nvec; ++c)
	{ 
	  colorvector tmp0(&lapeig.desc),tmp1(&lapeig.desc),tmp2(&lapeig.desc),tmp3(&lapeig.desc);
	  tmp0 = tmp1 = tmp2 = tmp3 = * lapeig.eigsys(t).evec[c];
#pragma omp parallel for default(none) shared(lapeig, tmp0,tmp1,tmp2,tmp3, nx, ny, nz, nt, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, px4, py4, pz4,ip, this_node) schedule(static)
	  for(unsigned int i=0; i<lapeig.desc.sites_on_node; ++i)
	    {
	      position p = lapeig.desc.get_position(i, this_node);
	      double_complex phase[4];
	      phase[0] = exp(double_complex(0,-1)*2*M_PI*(p[0]*px1[ip]/((double)nx) + p[1]*py1[ip]/((double)ny) + p[2]*pz1[ip]/((double)nz)));
	      phase[1] = exp(double_complex(0,-1)*2*M_PI*(p[0]*px2[ip]/((double)nx) + p[1]*py2[ip]/((double)ny) + p[2]*pz2[ip]/((double)nz)));
	      phase[2] = exp(double_complex(0,1)*2*M_PI*(p[0]*px3[ip]/((double)nx) + p[1]*py3[ip]/((double)ny) + p[2]*pz3[ip]/((double)nz)));
	      phase[3] = exp(double_complex(0,1)*2*M_PI*(p[0]*px4[ip]/((double)nx) + p[1]*py4[ip]/((double)ny) + p[2]*pz4[ip]/((double)nz)));
	      //for(int j=0; j<4; ++j) (*tmp[j]).data[i] *= phase[j];
	      tmp0.data[i] *= phase[0]; tmp1.data[i] *= phase[1]; tmp2.data[i] *= phase[2]; tmp3.data[i] *= phase[3];
	    }
	  
	  for(int r=0; r<nvec; ++r) (*matp[0])[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp0); 
	  for(int r=0; r<nvec; ++r) (*matp[1])[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp1); 
	  for(int r=0; r<nvec; ++r) (*matp[2])[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp2); 
	  for(int r=0; r<nvec; ++r) (*matp[3])[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp3); 
	}



    for(int it = 0 ; it < nt; it++)
      for(int r = 0 ; r < nvec ; r++)
	for(int c = 0 ; c < nvec ; c++){
	  tmpMomP1[(it*nvec*nvec + r*nvec + c)*2+0] = (*matp[0])[it](r,c).real;
	  tmpMomP1[(it*nvec*nvec + r*nvec + c)*2+1] = (*matp[0])[it](r,c).imag;
	  
	  tmpMomP2[(it*nvec*nvec + r*nvec + c)*2+0] = (*matp[1])[it](r,c).real;
	  tmpMomP2[(it*nvec*nvec + r*nvec + c)*2+1] = (*matp[1])[it](r,c).imag;

	  tmpMomP3[(it*nvec*nvec + r*nvec + c)*2+0] = (*matp[2])[it](r,c).real;
	  tmpMomP3[(it*nvec*nvec + r*nvec + c)*2+1] = (*matp[2])[it](r,c).imag;
	  
	  tmpMomP4[(it*nvec*nvec + r*nvec + c)*2+0] = (*matp[3])[it](r,c).real;
	  tmpMomP4[(it*nvec*nvec + r*nvec + c)*2+1] = (*matp[3])[it](r,c).imag;
	}
    printf("Finish momentum %d\n",ip);
    printf("Starting contractions\n");
    calculatePiPi(info,ptr_prop, tmpMomP1, tmpMomP2, tmpMomP3, tmpMomP4, Prec, outname, contract::I1);
  }

  contract::destroyContract();
  contract::destroyOpts();
  shutdown_machine();
  free(tmpMomP1);
  free(tmpMomP2);
  free(tmpMomP3);
  free(tmpMomP4);
  free(ptr_prop);
  return 0;
}
