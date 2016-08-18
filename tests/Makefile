include ../make.inc

CONTRACT = ../lib/libcontract.a

# include the names of the executables you want to create
TESTS = compute_rho compute_rhopi compute_pipi_I1 compute_pipi_I0 compute_sigma compute_sigmapi
INC += -I../include -I./ -I../lib

all: $(TESTS)

############################################
compute_rho: compute_rho.o $(CONTRACT)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

compute_rhopi: compute_rhopi.o $(CONTRACT)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

compute_pipi_I1: compute_pipi_I1.o $(CONTRACT)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

compute_pipi_I0: compute_pipi_I0.o $(CONTRACT)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

compute_sigma: compute_sigma.o $(CONTRACT)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

compute_sigmapi: compute_sigmapi.o $(CONTRACT)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)
############################################
%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(INC) $(CXXFLAGS) $< -c -o $@

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

clean:
	-rm -f *.o $(TESTS)

.PHONY: all clean
