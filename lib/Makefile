include ../make.inc

CONTRACT = libcontract.a

# Here include the object files
CONTRACT_OBJS = contract_kernels_rho.o contract_kernels_rhopi.o contract_kernels_pipi.o contract_kernels_loop_unity.o contract_kernels_sigma.o contract_kernels_sigmapi.o interface.o catchOpts.o 

# HEADERS FILES, FOUND IN INCLUDE
CONTRACT_HDRS = contract.h constants.h constants_rho.h constants_rhopi.h constants_pipi.h constants_sigma.h

# inline headers
INL_HDRS = device_opts_inline.h calculate_rho_core.h calculate_rhopi_core.h calculate_pipi_square_core.h calculate_loop_unity_core.h calculate_pipi_doubleTriangle_core.h calculate_pipi_doubleTriangleHor_core.h calculate_pipi_starfish_core.h calculate_sigma_one_fish_core.h calculate_sigmapi_hor_fish_core.h calculate_sigmapi_triangle_core.h

INC += -I../include -I./

HDRS = $(CONTRACT_HDRS:%=../include/%)
HDRS += $(INL_HDRS)

all: $(CONTRACT)

$(CONTRACT): $(CONTRACT_OBJS) ../make.inc
	ar cru $@ $(CONTRACT_OBJS)

clean:
	-rm -f *.o $(CONTRACT)

interface.o: interface.cpp contract.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@


%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) --ptxas-options=-v --maxrregcount=48  $< -c -o $@
#--maxrregcount=40
.PHONY: all clean
