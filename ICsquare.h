#ifndef ICsquare_H_
#define ICsquare_H_

//Class for storing a square initial condition Riemann problem, where each quadrant has its own initial values 

#include <iostream>
#include <cassert>
#include <vector>

class ICsquare{
public:
	// Constructor, assumes we are dealing with the positive unit sqaure, but this is optional
	ICsquare(float x_intersect, float y_intersect,float gamma, float xmin = 0,float xmax = 1,float  ymin = 0,float ymax = 1);
	
	void set_rho(float* rho);
	void set_pressure(float* pressure);
	void set_u(float* u);
	void set_v(float* v);
	void set_gamma(float gamma);
	
	void setIC(cpu_ptr_2D &rho, cpu_ptr_2D &rho_u, cpu_ptr_2D &rho_v, cpu_ptr_2D &E);

private:
	// Where quadrant division lines intersect
	float x_intersect, y_intersect;
	
	// Initial pressure and density
	float pressure_array[4];
	float rho_array[4];

	// Initial speeds in x and y-directions
	float u_array[4];
	float v_array[4];

	//x and y limits for the sqaure
	float xmin, ymin, xmax, ymax, gamma;
};

ICsquare::ICsquare(float x_intersect, float y_intersect, float gamma, float xmin, float xmax, float ymin, float ymax):x_intersect(x_intersect), y_intersect(y_intersect),gamma(gamma),\
xmin(xmin),ymin(ymin), xmax(xmax), ymax(ymax){
}

void ICsquare::set_gamma(float gamma){
	gamma = gamma;
}

void ICsquare::set_rho(float* rho){
	for (int i=0; i<4; i++)
		rho_array[i] = rho[i];
}

void ICsquare::set_u(float* u){
	for (int i=0; i<4; i++)
                u_array[i] = u[i];
}

void ICsquare::set_v(float* v){
	for (int i=0; i<4; i++)
                v_array[i] = v[i];
}

void ICsquare::set_pressure(float* pressure){
	for (int i=0; i<4; i++)
                pressure_array[i] = pressure[i];
}

void ICsquare::setIC(cpu_ptr_2D &rho, cpu_ptr_2D &rho_u, cpu_ptr_2D &rho_v, cpu_ptr_2D &E){
	int nx = rho.get_nx();
	int ny = rho.get_ny();
	float dx = (xmax-xmin)/(float) nx;
	float dy = (ymax-ymin)/(float) ny;
	float x, y;
	int quad;
	for (int i = 0; i < nx; i++){
		x = dx*i;
		for (int j=0; j < ny; j++){
			y = dy*j;
			// Quadrant 1
			if (x >= x_intersect && y >= y_intersect)
				quad = 0;
			// Quadrant 2
			else if (x < x_intersect && y >= y_intersect)
				quad = 1;
			// Quadrant 3
			else if ( x < x_intersect && y < y_intersect)
				quad = 2;
			// Quadrant 4
			else
				quad = 3;
			// Set initial values
			rho(i,j) = rho_array[quad];
			//printf("%.3f ", rho(i,j));
			rho_u(i,j) = rho_array[quad]*u_array[quad];
			rho_v(i,j) = rho_array[quad]*v_array[quad];
			E(i,j) = pressure_array[quad]/(gamma -1.0) + 0.5*rho_array[quad]*(u_array[quad]*u_array[quad] + v_array[quad]*v_array[quad]);
		}
	}
}
							
  

	




#endif
