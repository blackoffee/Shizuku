#pragma once

#include <stdio.h>
#include "common.h"
#include "cuda_runtime.h"
#include <GLEW/glew.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        //if (abort) exit(code);
    }
}
