#include "cuda.h"
#include "gpu_ptr.h"
#include "kernel_arg_structs.h"
#include <sys/time.h>

// Print GPU properties
void print_properties(){
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
         printf("Device count: %d\n", deviceCount);

        cudaDeviceProp p;
        cudaSetDevice(0);
        cudaGetDeviceProperties (&p, 0);
        printf("Compute capability: %d.%d\n", p.major, p.minor);
        printf("Name: %s\n" , p.name);
	printf("Compute concurrency %i\n", p.concurrentKernels);
        printf("\n\n");
}

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

inline void set_bc_args(collBCKernelArgs* args, gpu_raw_ptr U0, gpu_raw_ptr U1, gpu_raw_ptr U2, gpu_raw_ptr U3, unsigned int NX, unsigned int NY,int border){

	args->U0 = U0;
        args->U1 = U1;
        args->U2 = U2;
        args->U3 = U3;
	
	args->NX = NX;
	args->NY = NY;
	args->global_border = border;
}
	

inline void set_rk_args(RKKernelArgs* args, float* dt, gpu_raw_ptr U0, gpu_raw_ptr U1, gpu_raw_ptr U2, gpu_raw_ptr U3, gpu_raw_ptr R0, gpu_raw_ptr R1, gpu_raw_ptr R2, gpu_raw_ptr R3, gpu_raw_ptr Q0, gpu_raw_ptr Q1, gpu_raw_ptr Q2, gpu_raw_ptr Q3, unsigned int nx,unsigned int ny, int border){

	args->dt = dt;

	args->U0 = U0;
	args->U1 = U1;
	args->U2 = U2;
	args->U3 = U3;

	args->R0 = R0;
	args->R1 = R1;
	args->R2 = R2;
	args->R3 = R3;

	args->Q0 = Q0;
	args->Q1 = Q1;
	args->Q2 = Q2;
	args->Q3 = Q3;

 	args->nx = nx;
	args->ny = ny;
	args->global_border = border;
}

inline void set_dt_args(DtKernelArgs* args, float* L, float* dt, unsigned int nElements, float dx, float dy, float scale){
	args->L = L;
	args->dt = dt;
	args->nElements = nElements;
	args->dx = dx;
	args->dy = dy;
	args->scale = scale;
}


inline void set_flux_args(FluxKernelArgs* args, float* L, gpu_raw_ptr U0, gpu_raw_ptr U1, gpu_raw_ptr U2, gpu_raw_ptr U3, gpu_raw_ptr R0, gpu_raw_ptr R1, gpu_raw_ptr R2, gpu_raw_ptr R3, unsigned int nx, unsigned int ny, int border, float dx, float dy, float theta, float gamma, int innerDimX, int innerDimY){
	args->L = L; 	

	args->U0 = U0;
	args->U1 = U1;
	args->U2 = U2;
	args->U3 = U3;

	args->R0 = R0;
	args->R1 = R1;
	args->R2 = R2;
	args->R3 = R3;
 	
	args->nx = nx;
	args->ny = ny;
	args->global_border = border;
	args->dx = dx;
	args->dy = dy;

	args->gamma = gamma;
	args->theta = theta;

	args->innerDimX = innerDimX;
	args->innerDimY = innerDimY;

}

void setLandDt(int nElements, float* L_host, float* L_device, float* dt_device){

	L_host = new float[nElements];
	for (int i = 0; i < nElements; i++)
		L_host[i] = FLT_MAX;

	cudaMalloc((void**)&L_device, sizeof(float)*(nElements));
	cudaMemcpy(L_device,L_host, sizeof(float)*(nElements), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dt_device, sizeof(float));
}


void computeGridBlock(dim3& gridBlock, dim3& threadBlock, int NX, int NY, int tiledimX, int tiledimY, int blockdimX, int blockdimY){

        int gridDimx =  (NX + tiledimX - 1)/tiledimX;
        int gridDimy =  (NY + tiledimY - 1)/tiledimY;
                                                                                                                       
        threadBlock.x = blockdimX;
        threadBlock.y = blockdimY;
        gridBlock.x = gridDimx;
        gridBlock.y = gridDimy;
}
