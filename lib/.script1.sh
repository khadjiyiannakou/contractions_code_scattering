/usr/local/cuda-5.5/bin/nvcc  -O3 -ccbin g++ -m64 -gencode arch=compute_35,code=\"sm_35,compute_35\" -D__COMPUTE_CAPABILITY__=350 -arch=sm_35 -I/usr/local/cuda-5.5/include -I../include -I./ --ptxas-options=-v --maxrregcount=40 contract_kernels.cu -c -o contract_kernels.o