# TODO: Make this Makefile.in pretty

TARGETS = dl_serv 
CC=g++ -g -O3  -std=c++14 
DPDK=1
PS=0
NETMAP=0 ONVM=0
CPPFLAGS=-DMAX_CPUS=48 -Wno-deprecated-declarations 
#-fsanitize=address -fno-omit-frame-pointer -fsanitize=leak -use-after-free

# Add arch-specific optimization
ifeq ($(shell uname -m),x86_64)
LIBS += -m64
endif

# mtcp library and header 
MTCP_FLD    =../../mtcp
#MTCP_INC    =-I${MTCP_FLD}/include -I${MTCP_FLD}/src/include -I${MTCP_FLD}/src/libccp
MTCP_INC    =-I${MTCP_FLD}/include -I${MTCP_FLD}/src/include 
MTCP_LIB    =-L${MTCP_FLD}/lib 
MTCP_TARGET = ${MTCP_LIB}/libmtcp.a 


UTIL_FLD = ../../util
UTIL_INC = -I${UTIL_FLD}/include
UTIL_OBJ = ${UTIL_FLD}/http_parsing.o ${UTIL_FLD}/tdate_parse.o ${UTIL_FLD}/netlib.o 
# util library and header
INC = -I./dplane/include/ -I/usr/local/cuda/include -I/home/qi/dplane_rr/apps/dl_serv/dplane/include ${UTIL_INC} ${MTCP_INC} -I${UTIL_FLD}/include -I/home/qi/TensorRT-7.2.3.4/include -I${RTE_SDK}/${RTE_TARGET}/include

#-I/home/qi/libtf_gpu/include 
LIBS = ${MTCP_LIB} -L${MTCP_FLD}/src/libccp #-L/home/qi/libtf_gpu/lib 
# dpdk-specific variables
ifeq ($(DPDK),1)
DPDK_MACHINE_LINKER_FLAGS=${RTE_SDK}/${RTE_TARGET}/lib/ldflags.txt
#DPDK_MACHINE_LINKER_FLAGS=./ldflags.txt
DPDK_MACHINE_LDFLAGS=$(shell cat ${DPDK_MACHINE_LINKER_FLAGS})
#DPDK_LIBS = -L/usr/local/lib/x86_64-linux-gnu

endif


CUDA_LIBS = /usr/local/cuda/lib64
CUDA_STUB_LIBS=/usr/local/cuda/lib64/stubs
TRT_LIBS = /home/qi/TensorRT-7.2.3.4/lib
#TRT_LIBS = /home/qi/TensorRT-8.0.0.3/lib
DPLANE_LIBS = /home/qi/dplane_rr/apps/dl_serv/dplane


#DPDK_MACHINE_LINKER_FLAGS=$${RTE_SDK}/$${RTE_TARGET}/lib/ldflags.txt
#DPDK_MACHINE_LDFLAGS=$(shell cat ${DPDK_MACHINE_LINKER_FLAGS})
#$(info link_flag=$(DPDK_MACHINE_LDFLAGS))
LIBS += -g -O3 -pthread -lrt -march=native -Wl,-rpath=${DPLANE_LIBS} -L${RTE_SDK}/${RTE_TARGET}/lib -L${DPLANE_LIBS} -ldplane ${MTCP_FLD}/lib/libmtcp.a -L/home/qi/dplane_rr/mtcp/src/libccp -Wl,-rpath=${TRT_LIBS} -L${TRT_LIBS} -Wl,-rpath=${CUDA_LIBS} -L${CUDA_LIBS} -Wl,-rpath=${CUDA_STUB_LIBS} -L${CUDA_STUB_LIBS}\
		-lopencv_core -lopencv_imgproc -lopencv_highgui -Wl,-lnvinfer -Wl,-lcublas -Wl,-lcudnn -Wl,-lcudart -Wl,-lcuda -lnuma -lmtcp -lpthread -lrt -ldl -lgmp  -lm -labsl ${DPDK_MACHINE_LDFLAGS} 

#-L${RTE_SDK}/${RTE_TARGET}/lib 
#-ltensorflow 
#-L${TF_LIBS} 
#endif

ifeq ($V,) # no echo
	export MSG=@echo
	export HIDE=@
else
	export MSG=@\#
	export HIDE=
endif

all: dl_serv 

#$(info ${LIBS})

dl_serv.o: dl_serv.cpp
	$(MSG) "   CC $<"
	$(HIDE) ${CC} -c $< ${CPPFLAGS} ${INC} -Wl,--no-undefined 

dl_serv: dl_serv.o 
	$(MSG) "   LD $< ${UTIL_OBJ} ${LIBS} "
	$(HIDE) ${CC} $< ${LIBS} ${UTIL_OBJ} -Wl,--no-undefined -o $@ 

clean:
	$(MSG) "   CLEAN $(TARGETS)"
	$(HIDE) rm -f *~ *.o ${TARGETS} log_*

distclean: clean
	rm -rf Makefile
