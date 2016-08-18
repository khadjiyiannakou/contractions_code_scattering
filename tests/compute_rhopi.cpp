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
using contract::i_dir;
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

  if(Nmom != 2)
    ABORT("Error: compute_rhoPI needs only Nmom to be 2\n");
    
  omp_set_num_threads(Nthreads);

  init_machine(argc,argv);
  contract::contractInfo info;

  info.NeV=Nvec;
  info.Nt = Nt;
  info.dev=devID;
  info.PropH_fileName = propname;
  
  int nx,ny,nz,nt;

  double *tmpMomP1P2 = (double*) malloc(5*info.Nt*info.NeV*info.NeV*2*sizeof(double));
  double *tmpMomP1 = (double*) malloc(info.Nt*info.NeV*info.NeV*2*sizeof(double));
  double *tmpMomP2 = (double*) malloc(info.Nt*info.NeV*info.NeV*2*sizeof(double));



  nx = Nx;
  ny = Ny;
  nz = Nz;
  nt = Nt;
  int nvec = Nvec;

  
  ///////////////////////// memory allocation //////////////////
  std::vector<matrix>** matp;
  //matp has three kinds of forms e^(-ip1), e^(-ip2), e^i(p1+p2)       
  matp = new std::vector<matrix>*[3];
  //matps has only one form \delta_j e^i(p1+p2) \delta_j
  std::vector<matrix> matps(nt);
  std::vector<matrix>** matpc;
  //matpc has three components {e^i(p1+p2),\Delta_i}
  matpc = new std::vector<matrix>*[3];
  for(int i=0; i<3; ++i)
    {
      matp[i] = new std::vector<matrix>(nt);
      matpc[i] = new std::vector<matrix>(nt);
    }
  for(int t=0; t<nt; ++t)
    {
      for(int i=0; i<3; ++i) (*matp[i])[t].resize(nvec, nvec);
      matps[t].resize(nvec, nvec);
      for(int i=0; i<3; ++i) (*matpc[i])[t].resize(nvec, nvec);
    }

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

  /////////////////////////////////////////////////////////////
  // plain matrix momenta for p1,p2,p1p2
  for(int ip = 0 ; ip < momComb ; ip++){
    printf("Starting momentum %d\n",ip);
    for(int t=0; t<nt; ++t) 
      for(int c=0; c<nvec; ++c)
	{
	  colorvector tmp0(&lapeig.desc),tmp1(&lapeig.desc),tmp2(&lapeig.desc);
	  tmp0 = tmp1 = tmp2  = * lapeig.eigsys(t).evec[c];
#pragma omp parallel for default(none) shared(lapeig, tmp0,tmp1,tmp2, nx, ny, nz, nt, px1, py1, pz1, px2, py2, pz2, px3, py3, pz3, ip, this_node) schedule(static)
	  for(unsigned int i=0; i<lapeig.desc.sites_on_node; ++i)
	    {
	      position p = lapeig.desc.get_position(i, this_node);
	      double_complex phase[4];
	      phase[0] = exp(double_complex(0,-1)*2*M_PI*(p[0]*px1[ip]/((double)nx) + p[1]*py1[ip]/((double)ny) + p[2]*pz1[ip]/((double)nz)));
	      phase[1] = exp(double_complex(0,-1)*2*M_PI*(p[0]*px2[ip]/((double)nx) + p[1]*py2[ip]/((double)ny) + p[2]*pz2[ip]/((double)nz)));
	      phase[2] = exp(double_complex(0,1)*2*M_PI*(p[0]*px3[ip]/((double)nx) + p[1]*py3[ip]/((double)ny) + p[2]*pz3[ip]/((double)nz)));
	      tmp0.data[i] *= phase[0]; tmp1.data[i] *= phase[1]; tmp2.data[i] *= phase[2];
	    }

	  for(int r=0; r<nvec; ++r) (*matp[0])[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp0);
	  for(int r=0; r<nvec; ++r) (*matp[1])[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp1);
	  for(int r=0; r<nvec; ++r) (*matp[2])[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp2);
	}


    ////////////////////////////////////////////////////////////
    // smeared momentum matrices
    {
      vec_eigen_pair<colorvector> *dl[3];
      for(int j=0; j<3; ++j) dl[j] = new vec_eigen_pair<colorvector>(nvec, lapeig.desc);
      su3_field linksproj(&lapeig.desc);
      lapmult lm(linksproj);
      for(int t=0; t<nt; ++t)
	{
	  extract(links, linksproj, position(0, 0, 0, t));

	  for(int c=0; c<nvec; ++c) for(int r=0; r<nvec; ++r) matps[t](r,c) = 0;
	  for(int c=0; c<nvec; ++c) for(int j=0; j<3; ++j) lm.covderiv(j, *lapeig.eigsys(t).evec[c], *dl[j]->evec[c]);

	  for(int c=0; c<nvec; ++c)
	    {
	      for(int j=0; j<3; ++j)
		{
		  colorvector tmp(&lapeig.desc);
		  tmp = *dl[j]->evec[c];
#pragma omp parallel for default(none) shared(lapeig, tmp, nx, ny, nz, nt, px3, py3, pz3, ip, this_node) schedule(static)
		  for(unsigned int i=0; i<lapeig.desc.sites_on_node; ++i)
		    {
		      position p = lapeig.desc.get_position(i, this_node);
		      double_complex phase = exp(double_complex(0,1)*2*M_PI*(p[0]*px3[ip]/((double)nx) + p[1]*py3[ip]/((double)ny) + p[2]*pz3[ip]/((double)nz)));
		      tmp.data[i] *= phase;
		    }

		  for(int r=0; r<nvec; ++r) matps[t](r,c) += cscalar_product(*dl[j]->evec[r], tmp);
		}
	    }
	}
      for(int j=0; j<3; ++j) delete dl[j];
    }
    ///////////////////////////////////////////////////////////
    // smeared momentum matrices with commutators
    {
      vec_eigen_pair<colorvector> *dl[3];
      for(int j=0; j<3; ++j) dl[j] = new vec_eigen_pair<colorvector>(nvec, lapeig.desc);
      su3_field linksproj(&lapeig.desc);
      lapmult lm(linksproj);
      for(int t=0; t<nt; ++t)
	{
	  extract(links, linksproj, position(0, 0, 0, t));

	  for(int i=0; i<3; ++i) for(int c=0; c<nvec; ++c) for(int r=0; r<nvec; ++r) (*matpc[i])[t](r,c) = 0;
	  for(int c=0; c<nvec; ++c) for(int j=0; j<3; ++j) lm.covderiv(j, *lapeig.eigsys(t).evec[c], *dl[j]->evec[c]);

	  for(int c=0; c<nvec; ++c)
	    {
	      for(int j =0; j<3; ++j)
		{
		  colorvector tmp(&lapeig.desc);
		  colorvector tmp1(&lapeig.desc);
		  tmp  = *dl[j]->evec[c];
		  tmp1= *lapeig.eigsys(t).evec[c];
#pragma omp parallel for default(none) shared(lapeig, tmp, tmp1, nx, ny, nz, nt, px3, py3, pz3, ip, this_node) schedule(static)
		  for(unsigned int i=0; i<lapeig.desc.sites_on_node; ++i)
		    {
		      position p = lapeig.desc.get_position(i, this_node);
		      double_complex phase = exp(double_complex(0,1)*2*M_PI*(p[0]*px3[ip]/((double)nx) + p[1]*py3[ip]/((double)ny) + p[2]*pz3[ip]/((double)nz)));
		      tmp.data[i] *= phase;
		      tmp1.data[i] *= phase;
		    }

		  for(int r=0; r<nvec; ++r) (*matpc[j])[t](r,c) = cscalar_product(*lapeig.eigsys(t).evec[r], tmp)-cscalar_product(*dl[j]->evec[r], tmp1);
		}
	    }

	}
      for(int j=0; j<3; ++j) delete dl[j];
    }
    ///////////////////////////////////////////////////////////

    for(int it = 0 ; it < nt; it++)
      for(int r = 0 ; r < nvec ; r++)
	for(int c = 0 ; c < nvec ; c++){
	  tmpMomP1P2[(0*nt*nvec*nvec + it*nvec*nvec + r*nvec + c)*2+0] = (*matp[2])[it](r,c).real;
	  tmpMomP1P2[(0*nt*nvec*nvec + it*nvec*nvec + r*nvec + c)*2+1] = (*matp[2])[it](r,c).imag;

	  tmpMomP1P2[(1*nt*nvec*nvec + it*nvec*nvec + r*nvec + c)*2+0] = matps[it](r,c).real;
	  tmpMomP1P2[(1*nt*nvec*nvec + it*nvec*nvec + r*nvec + c)*2+1] = matps[it](r,c).imag;
	  for(int i = 0 ; i < 3 ; i++){
	    tmpMomP1P2[((2+i)*nt*nvec*nvec + it*nvec*nvec + r*nvec + c)*2+0] = (*matpc[i])[it](r,c).real;
	    tmpMomP1P2[((2+i)*nt*nvec*nvec + it*nvec*nvec + r*nvec + c)*2+1] = (*matpc[i])[it](r,c).imag;
	  }

	}

    for(int it = 0 ; it < nt; it++)
      for(int r = 0 ; r < nvec ; r++)
	for(int c = 0 ; c < nvec ; c++){
	  tmpMomP1[(it*nvec*nvec + r*nvec + c)*2+0] = (*matp[0])[it](r,c).real;
	  tmpMomP1[(it*nvec*nvec + r*nvec + c)*2+1] = (*matp[0])[it](r,c).imag;

	  tmpMomP2[(it*nvec*nvec + r*nvec + c)*2+0] = (*matp[1])[it](r,c).real;
	  tmpMomP2[(it*nvec*nvec + r*nvec + c)*2+1] = (*matp[1])[it](r,c).imag;

	}
    printf("Finish momentum %d\n",ip);
    printf("Starting contractions\n");
    calculateRhoPi(i_dir,info,ptr_prop,tmpMomP1, tmpMomP2, tmpMomP1P2,Prec, outname);
  }

  contract::destroyContract();
  contract::destroyOpts();
  shutdown_machine();
  free(tmpMomP1P2);
  free(tmpMomP1);
  free(tmpMomP2);
  free(ptr_prop);
  return 0;
}
