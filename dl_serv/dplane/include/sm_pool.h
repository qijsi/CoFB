#pragma once
#include <iostream>
#include <cuda.h>
#include <thread>
#include <vector>
#include <unistd.h>
#include <assert.h>
#include <cuda_runtime_api.h>

#define CONTEXT_POOL_SIZE  20
#define MIN_SM 2
#define THREAD_NUM 8


class sm_pool {
    public:
    int create_pool();
    static CUcontext get_context(int i);

	static CUcontext contextPool[CONTEXT_POOL_SIZE];
    int num_sm;
    int avail_sm;
    cudaDeviceProp prop;
    static bool initialization;
};

