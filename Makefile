# Makefile for MCGPULite @ github.com/z0gSh1u/MCGPULite
.SUFFIXES: .cu .o

PROG = MCGPULite_v1.3.x

SHELL = /bin/sh
RM = /bin/rm -vf

# You should modify these according to your machine.
NVCC = /usr/local/cuda-10.1/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda-10.1/include/
CUDA_LIB = /usr/local/cuda-10.1/lib64/
CUDA_SDK_INCLUDE = /usr/local/cuda-10.0/samples/common/inc/
CUDA_SDK_LIB = /usr/local/cuda-10.0/samples/common/lib/linux/x86_64/
OPENMPI_INCLUDE = /home/zhuoxu/app/openmpi-4.1.1/install/include/
OPENMPI_LIB = /home/zhuoxu/app/openmpi-4.1.1/install/lib/
ZLIB_INCLUDE = /home/zhuoxu/app/zlib-1.2.11/install/include/
ZLIB_LIB = /home/zhuoxu/app/zlib-1.2.11/install/lib/

CFLAGS = -O3 -use_fast_math -m64 -DUSING_CUDA -DUSING_MPI \
	-I./ -I$(CUDA_INCLUDE) -I$(CUDA_SDK_INCLUDE) -I$(OPENMPI_INCLUDE) -I$(ZLIB_INCLUDE) \
	-L$(CUDA_SDK_LIB) -L$(CUDA_LIB) -lcudart \
	-lm \
	-L$(ZLIB_LIB) -lz \
	-L$(OPENMPI_LIB) -lmpi \
	--ptxas-options=-v

SRCS = MCGPULite_v1.3.cu

default: $(PROG)
$(PROG):
	$(NVCC) $(CFLAGS) $(SRCS) -o $(PROG)

clean:
	$(RM) $(PROG)
