########### make.inc included by the Makefile

CUDA_INSTALL_PATH = /usr/local/cuda-7.5
MPI_HOME = /usr/lib/openmpi/lib
MY_HOME = /home/kyriakos/Desktop/contractions/KH_package

CPU_ARCH = x86_64
GPU_ARCH = sm_35
OS = linux

HOST_DEBUG = no
DEVICE_DEBUG = no

FECC = gcc
FECXX = g++

CC  = $(FECC)
CXX = $(FECXX)
NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc 

INC = -I$(CUDA_INSTALL_PATH)/include
INC += -I$(MY_HOME)/..
INC += -I$(MY_HOME)/../eigen
INC += -I$(MY_HOME)/../gwu-qcd/include

ifeq ($(strip $(CPU_ARCH)), x86_64)
  ifeq ($(strip $(OS)), osx)
    LIB = -L$(CUDA_INSTALL_PATH)/lib -lcudart -lcufft -lcublas
    NVCCOPT = -m64
  else
    LIB = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcufft -lcublas
    NVCCOPT = 
  endif
else
  LIB = -L$(CUDA_INSTALL_PATH)/lib -lcudart -m32 -lcufft -lcublas
  COPT = -malign-double -m32
  NVCCOPT = -m32
endif

COMP_CAP = $(GPU_ARCH:sm_%=%0)

COPT += -D__COMPUTE_CAPABILITY__=$(COMP_CAP)
NVCCOPT += -D__COMPUTE_CAPABILITY__=$(COMP_CAP)

TESLA_ARCH = $(shell [ $(COMP_CAP) -lt 200 ] && echo true)
ifneq ($(TESLA_ARCH),true)
  NVCCOPT += 
else
  NVCCOPT += -w
endif

ifeq ($(strip $(HOST_DEBUG)), yes)
  NVCCOPT += -g -DHOST_DEBUG
  COPT += -g -fno-inline -DHOST_DEBUG
endif

ifeq ($(strip $(DEVICE_DEBUG)), yes)
  NVCCOPT += -G
endif

CFLAGS = -Wall -O3 -std=c99 $(COPT) $(INC) -fopenmp
CXXFLAGS = -Wall -O3 $(COPT) $(INC) -fopenmp
NVCCFLAGS = -O3 $(NVCCOPT) -arch=$(GPU_ARCH) $(INC)
LDFLAGS = -fPIC $(LIB)

LDFLAGS += -L$(MY_HOME)/../gwu-qcd/lib -L$(MY_HOME)/../ -lLapH -lqcd -lmpi_cxx -lmpi -lgomp 