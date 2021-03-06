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

  if(Nmom != 1)
    ABORT("Error: compute_sigma needs only Nmom to be 1\n");

  omp_set_num_threads(Nthreads);

  init_machine(argc,argv);
  contract::contractInfo info;

  info.NeV=Nvec;
  info.Nt = Nt;
  info.dev=devID;
  info.PropH_fileName = propname;
  
  int nx,ny,nz,nt;

  double *tmpMom = (double*) malloc(4*info.Nt*info.NeV*info.NeV*2*sizeof(double));

  nx = Nx;
  ny = Ny;
  nz = Nz;
  nt = Nt;
  int nvec = Nvec;

  std::vector<matrix>** matp;
  std::vector<matrix>** matps;

  matp = new std::vector<matrix>*[2];
  matps = new std::vector<matrix>*[2];

  for(int i = 0 ; i < 2 ; i++){
    matp[i] = new std::vector<matrix>(nt);
    matps[i] = new std::vector<matrix>(nt);
  }


  for(int t = 0; t < nt; t++){
    for(int i = 0 ; i < 2 ; i++){
      (*matp[i])[t].resize(nvec, nvec);
      (*matps[i])[t].resize(nvec, nvec);
    }
  }

  int this_node=get_node_rank();
  ///////////////////////////////////////////////

  layout_minsurface_eo desc(nx,ny,nz,nt);

  su3_field links(&desc);


  read_kentucky_lattice(latname.c_str(), links); // read the configuration
  nhyp_smear_simple(links, links);       // apply one step of hyp smearing

  int bc[4] = {1,1,1,-1};
  apply_boundary(links, bc);             // apply boundary conditions

  laplacean_eigensolver lapeig(links);
  lapeig.allocate_eigensystem(info.NeV);
  read_laplace_eigensystem(eigename.c_str(), lapeig);

  contract::initContract(info);
  void *ptr_prop = readPropH(propname, Prec); // read the propagator
  /////////////////////////////////////////////////////////////////////////////////

  for(int ip = 0 ; ip < momComb ; ip++){

    int px, py, pz;
    px=px1[ip];
    py=py1[ip];
    pz=pz1[ip];

    printf("Starting momentum matrices (%+d, %+d, %+d)\n",px,py,pz);
    // plain momentum matrix
    for(int t=0; t<nt; ++t) 
      for(int c=0; c<nvec; ++c)
	{
	  colorvector tmp(&lapeig.desc), tmp2(&lapeig.desc);
	  tmp = tmp2 = *lapeig.eigsys(t).evec[c];
#pragma omp parallel for default(none) shared(lapeig, tmp, tmp2, nx, ny, nz, nt, px, py, pz, this_node) schedule(static)
	  for(unsigned int i=0; i<lapeig.desc.sites_on_node; ++i)
	    {
	      position p = lapeig.desc.get_position(i, this_node);
	      double_complex phaseP = exp(double_complex(0,+1)*2*M_PI*(p[0]*px/((double)nx) + p[1]*py/((double)ny) + p[2]*pz/((double)nz)));
	      double_complex phaseM = exp(double_complex(0,-1)*2*M_PI*(p[0]*px/((double)nx) + p[1]*py/((double)ny) + p[2]*pz/((double)nz)));
	      tmp.data[i] *= phaseP;
	      tmp2.data[i] *= phaseM;
	    }
	
	  for(int r=0; r<nvec; ++r) (*matp[0])[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp); 
	  for(int r=0; r<nvec; ++r) (*matp[1])[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp2); 
	}
  
    // smeared momentum matrix
    {
      vec_eigen_pair<colorvector> *dl[3];
      for(int j=0; j<3; ++j) dl[j] = new vec_eigen_pair<colorvector>(nvec, lapeig.desc);
      su3_field linksproj(&lapeig.desc);
      lapmult lm(linksproj);
      for(int t=0; t<nt; ++t) 
	{
	  extract(links, linksproj, position(0, 0, 0, t));

	  for(int c=0; c<nvec; ++c) for(int r=0; r<nvec; ++r) { (*matps[0])[t](r,c) = 0; (*matps[1])[t](r,c) = 0 ; }
	  for(int c=0; c<nvec; ++c) for(int j=0; j<3; ++j) lm.covderiv(j, *lapeig.eigsys(t).evec[c], *dl[j]->evec[c]);

	  for(int c=0; c<nvec; ++c)
	    {
	      for(int j=0; j<3; ++j)
		{
		  colorvector tmp(&lapeig.desc), tmp2(&lapeig.desc);
		  tmp = tmp2 = *dl[j]->evec[c];
#pragma omp parallel for default(none) shared(lapeig, tmp, tmp2, nx, ny, nz, nt, px, py, pz, this_node) schedule(static)
		  for(unsigned int i=0; i<lapeig.desc.sites_on_node; ++i)
		    {
		      position p = lapeig.desc.get_position(i, this_node);
		      double_complex phaseP = exp(double_complex(0,+1)*2*M_PI*(p[0]*px/((double)nx) + p[1]*py/((double)ny) + p[2]*pz/((double)nz)));
		      double_complex phaseM = exp(double_complex(0,-1)*2*M_PI*(p[0]*px/((double)nx) + p[1]*py/((double)ny) + p[2]*pz/((double)nz)));
		      tmp.data[i] *= phaseP;
		      tmp2.data[i] *= phaseM;
		    }

		  for(int r=0; r<nvec; ++r) (*matps[0])[t](r,c) += cscalar_product(*dl[j]->evec[r], tmp); 
		  for(int r=0; r<nvec; ++r) (*matps[1])[t](r,c) += cscalar_product(*dl[j]->evec[r], tmp2);
		}
	    }
	}
      for(int j=0; j<3; ++j) delete dl[j];
    }


    for(int i = 0 ; i < 2 ; i++){
      for(int it = 0 ; it < nt; it++)
	for(int r = 0 ; r < nvec ; r++)
	  for(int c = 0 ; c < nvec ; c++){
	    tmpMom[(i*nt*nvec*nvec + it*nvec*nvec + r*nvec + c)*2+0] = (*matp[i])[it](r,c).real;
	    tmpMom[(i*nt*nvec*nvec + it*nvec*nvec + r*nvec + c)*2+1] = (*matp[i])[it](r,c).imag;
	  }
    }

    for(int i = 0 ; i < 2 ; i++){
      for(int it = 0 ; it < nt; it++)
	for(int r = 0 ; r < nvec ; r++)
	  for(int c = 0 ; c < nvec ; c++){
	    tmpMom[((i+2)*nt*nvec*nvec + it*nvec*nvec + r*nvec + c)*2+0] = (*matps[i])[it](r,c).real;
	    tmpMom[((i+2)*nt*nvec*nvec + it*nvec*nvec + r*nvec + c)*2+1] = (*matps[i])[it](r,c).imag;
	  }
    }



    printf("Finish momentum matrices (%+d, %+d, %+d)\n",px,py,pz);
    printf("Starting contractions\n");
    calculateSigma(info,ptr_prop,tmpMom,Prec,outname);

  }

  contract::destroyContract();
  contract::destroyOpts();
  shutdown_machine();
  free(tmpMom);
  free(ptr_prop);

  for(int i = 0 ; i < 2 ; i++){
    delete matp[i];
    delete matps[i];
  }

  delete [] matp;
  delete [] matps;

  return 0;
}


