# TODO: Make this Makefile.in pretty

TARGETS = libdplane.so
OBJS = dplane.o inference.o utili.o
CC=g++ -g -O3  -std=c++14
DPDK=1

#GREEDY=1
SLOSLACK=1

CPPFLAGS=-DMAX_CPUS=48 -Wall -Wno-deprecated-declarations

ifeq ($(GREEDY), 1)
CPPFLAGS += -DGREEDY
endif

ifeq ($(SLOSLACK), 1)
CPPFLAGS += -DSLOSLACK
endif

# Add arch-specific optimization
ifeq ($(shell uname -m),x86_64)
LIBS += -m64
endif

# mtcp library and header 
MTCP_FLD    = /home/qi/dplane_rr/mtcp
MTCP_INC    =-I${MTCP_FLD}/include -I${MTCP_FLD}/src/include -I${MTCP_FLD}/src/libccp
MTCP_LIB    =-L${MTCP_FLD}/lib -L${MTCP_FLD}/src/libccp
MTCP_TARGET = ${MTCP_LIB}/libmtcp.a -lccp

UTIL_FLD = /home/qi/dplane_rr/util
UTIL_INC = -I${UTIL_FLD}/include
UTIL_OBJ = ${UTIL_FLD}/http_parsing.o ${UTIL_FLD}/tdate_parse.o ${UTIL_FLD}/netlib.o

IO_INC = -I/home/qi/dplane_rr/io_engine/include
DPLANE_INC = -I/home/qi/dplane_rr/apps/dl_serv/dplane/include
#TF_INC= -I/home/qi/libtf_gpu/include
DPDK_INC= -I${RTE_SDK}/${RTE_TARGET}/include

CUDA_INC = -I/usr/local/cuda/include
TRT_INC = -I/home/qi/TensorRT-7.2.3.4/include
#TRT_INC = -I/home/qi/TensorRT-8.0.0.3/include
OPENCV_INC = -I/usr/local/include/opencv4/

INC=-I./include ${DPLANE_INC} ${TRT_INC} ${OPENCV_INC} ${MTCP_INC} ${DPDK_INC} ${UTIL_INC} ${IO_INC} #${TF_INC}

#TF_LIBS += -ltensorflow  -labsl
TRT_LIBS += -lnvinfer
CUDA_LIBS += -lcuda -lcudart -lcudnn
OPENCV_LIBS += -lopencv_core -lopencv_imgproc -lopencv_highgui
TRT_LIBS_DIR=/home/qi/TensorRT-7.2.3.4/lib
#TRT_LIBS_DIR=/home/qi/TensorRT-8.0.0.3/lib

#DPDK_MACHINE_LINKER_FLAGS=${RTE_SDK}/${RTE_TARGET}/lib/ldflags.txt
#DPDK_MACHINE_LDFLAGS=$(shell cat ${DPDK_MACHINE_LINKER_FLAGS}) 

all: $(TARGETS)

inference.o: inference.cpp
	${CC} ${CPPFLAGS} ${INC} -fPIC -c $<

utili.o: utili.cpp
	$(CC) ${CPPFLAGS} ${INC} -fPIC -c $<

dplane.o: dplane.cpp inference.o
	${CC} ${CPPFLAGS} ${INC} -fPIC -c $<


$(TARGETS): $(OBJS)
	rm -f $@
	$(CC) -shared -fPIC ${OBJS} ${MTCP_FLD}/lib/libmtcp.a  -o $@ 
	#-Wl,--whole-archive ${MTCP_FLD}/lib/libmtcp.a -Wl,-lccp -Wl,--no-whole-archive -L${TRT_LIB_DIR}/lib -L/usr/local/cuda/lib64 -lasan ${TRT_LIBS} ${CUDA_LIBS} ${OPENCV_LIBS} -L${RTE_SDK}/${RTE_TARGET}/lib ${DPDK_MACHINE_LDFLAGS} -o $@  
#   -L/home/qi/libtf_gpu/lib 

clean:
	rm -f *~ *.o ${TARGETS} log_* *.so *.a

