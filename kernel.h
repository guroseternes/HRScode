#ifndef KERNEL_H_
#define KERNEL_H_
#include <stdio.h>
#include "gpu_ptr.h"
#include <cuda.h>
#include <cfloat>
#include "kernel_arg_structs.h"
//#include "global.h"

//Declare variables

//Block sizes
// For flux function
const int BLOCKDIM = 16;
const int SM_BLOCKDIM_Y = BLOCKDIM + 1;
const int TILEDIM = BLOCKDIM-2;
const int INNERTILEDIM = TILEDIM-2;

const int BLOCKDIM_RK = 16;

const int BLOCKDIM_BC = 256;

const int TIMETHREADS = 64;

// Constants for the method
const float THETA = 1.5; //minmod limiter parameter
const float GAMMA = 1.4; //gas constant

// Pointer to store the r-values for the CFL condition
//extern float* L_host;
//extern float* L_device;
extern int nElements;

// Pointer to store the dt-value f
extern float* dt_device;

// Kernel arguments
extern collBCKernelArgs* BCArgs[3];
extern FluxKernelArgs* fluxArgs[3];
extern RKKernelArgs* RKArgs[3];
extern DtKernelArgs* dtArgs;

// gridBlocks and threadBlocks
extern dim3 gridBC;
extern dim3 blockBC;

extern dim3 gridBlockFlux;
extern dim3 threadBlockFlux;

extern dim3 gridBlockRK;
extern dim3 threadBlockRK;

// Declare functions 
extern void init_allocate();

void callRKKernel(dim3 grid, dim3 block, int step, RKKernelArgs* RKarg);

void callDtKernel(int nThreads, DtKernelArgs* DtArg);

void callCollectiveSetBCWall(dim3 grid, dim3 block, const collBCKernelArgs* h_ctx);

void callFluxKernel(dim3 grid, dim3 block, int step, FluxKernelArgs* FluxArgs);

void callCollectiveSetBCOpen(dim3 grid, dim3 block, const collBCKernelArgs* arg);

#endif

