#include "cpu_ptr.h"
#include "kernel.h"
#include "ICsquare.h" 
#include "util_cpu.h"
#include "configurations.h"
#include "cudaProfiler.h"
#include "cuda_profiler_api.h"
//#include "global.h"

// Global variables, will be moved to h-file when ready
// Grid parameters
unsigned int nx = 400;
unsigned int ny = 400;
int border = 2;

//Time parameters
float timeLength =0.3;
float currentTime = 0;
float dt;
float cfl_number = 0.475;
float theta = 1.3;
float gamma = 1.4;
int step = 0;
int maxStep = 1000;

int main(int argc,char **argv){

// Print GPU properties
print_properties();

// Files to print the result after the last time step
FILE *rho_file;
FILE *E_file;
rho_file = fopen("rho_final.txt", "w");
E_file = fopen("E_final.txt", "w");

// Construct initial condition for problem
ICsquare Config(0.5,0.5,gamma);

// Set initial values for Configuration 1
Config.set_rho(rhoConfig19);
Config.set_pressure(pressureConfig19);
Config.set_u(uConfig19);
Config.set_v(vConfig19);

// Determining global border based on left over tiles (a little hack)
int globalPadding;
int gridDimx = (nx + 2*border + INNERTILEDIM)/INNERTILEDIM;
globalPadding = INNERTILEDIM*gridDimx -(nx+2*border);

// Change border to add padding
border = border + globalPadding/2;

// Initiate the matrices for the unknowns in the Euler equations
cpu_ptr_2D rho(nx, ny, border,1);
cpu_ptr_2D E(nx, ny, border,1);
cpu_ptr_2D rho_u(nx, ny, border,1);
cpu_ptr_2D rho_v(nx, ny, border,1);
cpu_ptr_2D zeros(nx, ny, border,1);

// Set initial condition
Config.setIC(rho, rho_u, rho_v, E);


double timeStart = get_wall_time();


// Transfer to the GPU
gpu_ptr_2D rho_device(nx, ny, border, rho.get_ptr());
gpu_ptr_2D E_device(nx, ny, border, E.get_ptr());
gpu_ptr_2D rho_u_device(nx, ny, border, rho_u.get_ptr());
gpu_ptr_2D rho_v_device(nx, ny, border, rho_v.get_ptr()); 

gpu_ptr_2D R0(nx, ny, border);
R0.set(0,0,0,nx,ny,border); 
gpu_ptr_2D R1(nx, ny, border);
R1.set(0,0,0,nx,ny,border); 
gpu_ptr_2D R2(nx, ny, border);
R2.set(0,0,0,nx,ny,border); 
gpu_ptr_2D R3(nx, ny, border);
R3.set(0,0,0,nx,ny,border); 

/*
gpu_ptr_2D R0(nx, ny, border, zeros.get_ptr());
gpu_ptr_2D R1(nx, ny, border, zeros.get_ptr());
gpu_ptr_2D R2(nx, ny, border, zeros.get_ptr());
gpu_ptr_2D R3(nx, ny, border, zeros.get_ptr());
*/

gpu_ptr_2D Q0(nx, ny, border);
Q0.set(0,0,0,nx,ny,border); 
gpu_ptr_2D Q1(nx, ny, border);
Q1.set(0,0,0,nx,ny,border); 
gpu_ptr_2D Q2(nx, ny, border);
Q2.set(0,0,0,nx,ny,border); 
gpu_ptr_2D Q3(nx, ny, border);
Q3.set(0,0,0,nx,ny,border); 
/*
gpu_ptr_2D Q0(nx, ny, border, zeros.get_ptr());
gpu_ptr_2D Q1(nx, ny, border, zeros.get_ptr());
gpu_ptr_2D Q2(nx, ny, border, zeros.get_ptr());
gpu_ptr_2D Q3(nx, ny, border, zeros.get_ptr());
*/

// cuda error check
//printf("1 %s\n", cudaGetErrorString(cudaGetLastError()));


// Test 
cpu_ptr_2D rho_dummy(nx, ny, border);
cpu_ptr_2D E_dummy(nx, ny, border);


//cudaProfilerStart();
//cuProfilerStart();

// Set block and grid sizes
dim3 gridBC = dim3(1, 1, 1);
dim3 blockBC = dim3(BLOCKDIM_BC,1,1);

dim3 gridBlockFlux;
dim3 threadBlockFlux;

dim3 gridBlockRK;
dim3 threadBlockRK;

computeGridBlock(gridBlockFlux, threadBlockFlux, nx + 2*border, ny + 2*border, INNERTILEDIM, BLOCKDIM);

computeGridBlock(gridBlockRK, threadBlockRK, nx + 2*border, ny + 2*border, BLOCKDIM_RK, BLOCKDIM_RK);

int nElements = gridBlockFlux.x*gridBlockFlux.y;
//printf("xDim %i\t yDim %i\t", gridBlockFlux.x, threadBlockFlux.y); 

// Set the dt and L on the device
float* dt_device;
float* dt_host;
float* L_host;
float* L_device;

//setLandDt(nElements, L_host, L_device, dt_device); 
printf("2 %s\n", cudaGetErrorString(cudaGetLastError()));
	L_host = new float[nElements];
	dt_host = new float[1];
	for (int i = 0; i < nElements; i++)
		L_host[i] = FLT_MAX;

	cudaMalloc((void**)&L_device, sizeof(float)*(nElements));
	cudaMemcpy(L_device,L_host, sizeof(float)*(nElements), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dt_device, sizeof(float));

printf("3 %s\n", cudaGetErrorString(cudaGetLastError()));

init_allocate();

// Set BC arguments
set_bc_args(BCArgs[0], rho_device.getRawPtr(), rho_u_device.getRawPtr(), rho_v_device.getRawPtr(), E_device.getRawPtr(), nx+2*border, ny+2*border, border);
set_bc_args(BCArgs[1], Q0.getRawPtr(), Q1.getRawPtr(), Q2.getRawPtr(), Q3.getRawPtr(), nx+2*border, ny+2*border, border);
set_bc_args(BCArgs[2], rho_device.getRawPtr(), rho_u_device.getRawPtr(), rho_v_device.getRawPtr(), E_device.getRawPtr(), nx+2*border, ny+2*border, border);

// Set FLUX arguments
set_flux_args(fluxArgs[0], L_device, rho_device.getRawPtr(), rho_u_device.getRawPtr(), rho_v_device.getRawPtr(), E_device.getRawPtr(), R0.getRawPtr(),R1.getRawPtr(), R2.getRawPtr(), R3.getRawPtr(), nx, ny, border, rho.get_dx(), rho.get_dy(), theta, gamma);
set_flux_args(fluxArgs[1], L_device, Q0.getRawPtr(), Q1.getRawPtr(), Q2.getRawPtr(), Q3.getRawPtr(), R0.getRawPtr(),R1.getRawPtr(), R2.getRawPtr(), R3.getRawPtr(), nx, ny, border, rho.get_dx(), rho.get_dy(), theta, gamma);

// Set TIME argument
set_dt_args(dtArgs, L_device, dt_device, nElements, rho.get_dx(), rho.get_dy(), cfl_number);

// Set Rk arguments
set_rk_args(RKArgs[0], dt_device, rho_device.getRawPtr(), rho_u_device.getRawPtr(), rho_v_device.getRawPtr(), E_device.getRawPtr(), R0.getRawPtr(), R1.getRawPtr(), R2.getRawPtr(), R3.getRawPtr(), Q0.getRawPtr(), Q1.getRawPtr(), Q2.getRawPtr(), Q3.getRawPtr(), nx, ny, border); 
set_rk_args(RKArgs[1], dt_device, Q0.getRawPtr(), Q1.getRawPtr(), Q2.getRawPtr(), Q3.getRawPtr(), R0.getRawPtr(), R1.getRawPtr(), R2.getRawPtr(), R3.getRawPtr(), rho_device.getRawPtr(), rho_u_device.getRawPtr(), rho_v_device.getRawPtr(), E_device.getRawPtr(), nx, ny, border); 

// Update boudries
callCollectiveSetBCOpen(gridBC, blockBC, BCArgs[0]);

while (currentTime < timeLength && step < maxStep){	
	
	//RK1	
	//Compute flux
	callFluxKernel(gridBlockFlux, threadBlockFlux, 0, fluxArgs[0]);	
	
	// Compute timestep (based on CFL condition)
	callDtKernel(TIMETHREADS, dtArgs);
	
	cudaMemcpy(dt_host, dt_device, sizeof(float), cudaMemcpyDeviceToHost);
	// Perform RK1 step
	callRKKernel(gridBlockRK, threadBlockRK, 0, RKArgs[0]);
	
	//Update boudries
	callCollectiveSetBCOpen(gridBC, blockBC, BCArgs[1]);		

	//RK2
	// Compute flux
	callFluxKernel(gridBlockFlux, threadBlockFlux, 1, fluxArgs[1]);

	//Perform RK2 step
	callRKKernel(gridBlockRK, threadBlockRK, 1, RKArgs[1]);	

	callCollectiveSetBCOpen(gridBC, blockBC, BCArgs[2]);

	step++;	
	currentTime += dt_host[0];	
//	printf("Step: %i, current time: %.6f dt:%.6f\n" , step,currentTime, dt_host[0]);

}

//cuProfilerStop();
//cudaProfilerStop();

printf("Elapsed time %.5f", get_wall_time() - timeStart);

rho_device.download(rho_dummy.get_ptr());
rho_dummy.printToFile(rho_file, true, true);
//R3.download(rho_dummy.get_ptr());
//E_dummy.printToFile(E_file, true, true);

printf("Step: %i, current time: %.6f dt:%.6f" , step,currentTime, dt_host[0]); 
// Print test

printf("9 %s\n", cudaGetErrorString(cudaGetLastError()));

/*
cudaMemcpy(L_host, L_device, sizeof(float)*(nElements), cudaMemcpyDeviceToHost);
for (int i =0; i < nElements; i++)
	printf(" %.7f ", L_host[i]); 
*/
printf("%s\n", cudaGetErrorString(cudaGetLastError()));

return(0);
}


