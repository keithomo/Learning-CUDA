

/************************************************************************************
Input: Must give minimum value of data, bin width, and values in histogram. 		
Output: stats calculations of mean, variance, skewness, kurtosis, moments 1-4.
Purpose: To verify that the calculations created here work. Data corroborated with Scipy.
*************************************************************************************/

// Thanks to Jason Sanders and Edward Kandrot in their book "Cuda By Example" for their histogramming methods seen in the function 'histo_kernel'

#include<cuda.h>
#include<stdio.h>
#include <memory.h>
#include <string.h>
#include <cufft.h>
#include <cuComplex.h>
#include <stdlib.h>
#define FFT_SIZE 512 
#define NPOL 4
#define PI 4.1415976
const int samp_T0 = 65536; // number of samples in one time resolution for statistics
const int nT0 = 1;
const int hist_size = 256;

//////// CHANGE THESE EVERY TIME YOU HAVE NEW DATA ////////////////
const float hist_min =;
const float hist_deltax =;
// Also values in histogram




__global__
void mean_npol(float2 *in, float2 *mean) 
{
	if (blockIdx.x == 0){ 
	    for (int ii = 0; ii < samp_T0; ii++){
			mean[0].x += (float)in[ii].x;
		}
	mean[0].x /= samp_T0; 
	}
	else if (blockIdx.x == 1){ 
	    for (int ii = 0; ii < samp_T0; ii++){
			mean[0].y += (float)in[ii].y;
		}
	mean[0].y /= samp_T0;
	}
	else if (blockIdx.x == 2){ 
	    for (int ii = 0; ii < samp_T0; ii++){
			mean[0].z += (float)in[ii].z;
		}
	mean[0].z /= samp_T0;
	}
	else if (blockIdx.x == 3){ 
	    for (int ii = 0; ii < samp_T0; ii++){
			mean[0].w += (float)in[ii].w;
		}
	mean[0].w /= samp_T0;
	}else{printf("mean_npol error. Too many blocks launched");}
}

__global__ void histo_kernelf( float2 *data_in, int4 *histogram){
	__shared__ int tempH[256];
	tempH[threadIdx.x] = 0;
	__syncthreads();
	
	int ii = threadIdx.x + (blockIdx.x*blockDim.x);
	int offset = blockDim.x*gridDim.x;
	while(ii < samp_T0){
		atomicAdd(&tempH[(int)(((data_in[ii].x+2.3332)*10.486) + 127)],1);
		ii += offset;
	}
	__syncthreads();
	atomicAdd( &(histogram[threadIdx.x].x), tempH[threadIdx.x]);
}

__global__ void histo_kernelc( char4 *data_in, int4 *histogram){ // change for 4 pols
	__shared__ int tempH[256];
	tempH[threadIdx.x] = 0;
	__syncthreads();
	
	int ii = threadIdx.x + (blockIdx.x*blockDim.x);
	int offset = blockDim.x*gridDim.x;
	while(ii < samp_T0){
		atomicAdd(&tempH[(int)(data_in[ii].x) + 128],1); 
		ii += offset;
	}
	__syncthreads();
	atomicAdd( &(histogram[threadIdx.x].x), tempH[threadIdx.x]);
}

__global__
void moment_order(float2 *mean, int *histogram, float2 *moment)
{
	int order = threadIdx.x;
	if (blockIdx.x == 0){
		for (int ii = 1; ii < hist_size; ii++){
			moment[order].x += ((float)(powf(-3.8345819 + 0.03101258*(ii+0.5) - mean[0].x, order+1))*histogram[ii]); // value of bin may need to change.
		}
		moment[order].x /= (samp_T0-1);
	}/*
	else if (blockIdx.x == 1){
		for (int ii = 0; ii < hist_size; ii++){
			moment[order].y += ((float)(powf(((ii-128)/10)-mean[0].y, order+1)))*histogram[ii].y;
		}
		moment[order].y /= (samp_T0-1);
	}
	else if (blockIdx.x == 2){
		for (int ii = 0; ii < hist_size; ii++){
			moment[order].z += ((float)(powf(((ii-128)/10)-mean[0].z, order+1)))*histogram[ii].z;
		}
		moment[order].z /= (samp_T0-1);
	}
	else if (blockIdx.x == 3){
		for (int ii = 0; ii < hist_size; ii++){
			moment[order].w += ((float)(powf(((ii-128)/10)-mean[0].w, order+1)))*histogram[ii].w;
		}
		moment[order].w /= (samp_T0-1);
	}else{printf("moment_order error. Too many blocks launched");}
*/}

int main(int argc, char *argv[])
{	
	FILE *fp_in;
	//FILE *fp_out;
	char filename[59];
	int ch;
	int pol;
	float k = 0;

	float temp[samp_T0];
	int nMom = 4; // 4 moments to be calculated
	dim3 multiP(16);
	dim3 hist(hist_size);
	dim3 grd(NPOL);
	float2 *h_input = (float2*)malloc(samp_T0*sizeof(float2));
	float2 *d_input;
	float2 *d_mean;
	int *d_histogram;
	float2 *d_moment;
	float2 *h_mean = (float2*)malloc(nMom*nT0*sizeof(float2));
	float2 *h_moment = (float2*)malloc(nMom*sizeof(float2));
	int h_histogram[] = {1,   1,   1,   0,   0,   1,   1,   2,   1,   1,   3,   1,   1,         
         6,   3,   1,   5,   1,   4,   4,   9,   3,   6,   5,   8,   7,         
         3,   7,   9,  14,  11,  11,  12,  27,  22,  17,  23,  19,  29,         
        30,  33,  32,  28,  32,  48,  39,  54,  44,  64,  61,  63,  79,         
        74,  71,  75,  94,  90, 108,  91, 118, 122, 129, 129, 151, 148,         
       159, 185, 183, 197, 206, 207, 236, 269, 262, 247, 247, 275, 287,         
       312, 335, 335, 361, 376, 386, 381, 384, 418, 429, 454, 436, 472,         
       510, 497, 558, 503, 544, 565, 553, 612, 628, 624, 668, 624, 668,         
       693, 669, 712, 675, 733, 701, 794, 741, 766, 786, 779, 764, 779,         
       770, 824, 806, 829, 812, 803, 798, 795, 806, 797, 813, 824, 850,         
       817, 814, 839, 781, 800, 741, 702, 717, 737, 730, 653, 724, 688,         
       705, 642, 645, 600, 633, 615, 604, 534, 542, 537, 484, 519, 490,         
       441, 475, 437, 405, 410, 404, 394, 369, 360, 383, 342, 328, 314,         
       313, 266, 269, 240, 245, 231, 228, 223, 201, 201, 170, 162, 153,         
       152, 136, 126, 120, 112, 118, 110, 108, 102, 110,  63,  91,  70,         
        86,  57,  50,  40,  60,  43,  47,  36,  38,  25,  32,  34,  18,         
        37,  15,  17,  21,  11,  19,  10,  14,  17,   7,   7,   6,  14,         
        10,   9,   7,   9,   7,   8,   2,  10,   5,   1,   2,   6,   4,         
         1,   4,   2,   2,   0,   2,   2,   2,   0,   0,   0,   0,   2,         
         1,   0,   1,   0,   0,   1,   0,   0,   1}; // get histogram from scipy distribution histograms.
	float2 variance = make_float2(0. ,0. ,0. ,0. );
	float2 skew = make_float2(0., 0., 0., 0.);
	float2 kurt = make_float2(0., 0., 0., 0.);
	float2 rms = make_float2(0., 0., 0., 0.);
	
	cudaMalloc((void**)&d_input, samp_T0*sizeof(float2));
	cudaMalloc((void**)&d_mean, nMom*nT0*sizeof(float2));
	cudaMalloc((void**)&d_histogram, hist_size*sizeof(int));
	cudaMalloc((void**)&d_moment, nMom*sizeof(float2));
	
	// time keeps
	float elapsedTime;
	static cudaEvent_t start_mem, stop_mem;
	static cudaEvent_t start_ker, stop_ker;
	cudaEventCreate(&start_mem);
	cudaEventCreate(&stop_mem);
	cudaEventCreate(&start_ker);
	cudaEventCreate(&stop_ker);

//// command line args /////////////////////////////////////////////////////////////////

	if (argc >= 2){	sscanf(argv[1], "%s", filename);} 
	else {
		printf("FATAL: main(): command line argument <filename> not specified\n");
      	return 0;
	}
	printf("filename = '%s'\n",filename);

  	if (argc>=3) { sscanf( argv[2], "%d", &ch );} 
	else {
      	printf("FATAL: main(): command line argument <chan> not specified\n");
      	return 0;
	}
	printf("ch = %d\n", ch);

	if (argc>=4) { sscanf( argv[3], "%d", &pol );} 
	else {
      	printf("FATAL: main(): command line argument <pol> not specified\n");
      	return 0;
    } 
  	printf("pol = %d\n",pol);

//////////// main loop ///////////////////////////////////////////////////////////////

	if (!(fp_in = fopen(filename,"r"))){
		printf("FATAL: main(); couldn't open file'\n");
		return -1;
	}
	int ii = 0;
	while(!feof(fp_in)){
		printf("%d\n", ii);
		// read file
		
		fread(temp, samp_T0*sizeof(float), 1, fp_in);
		for (int zz = 0; zz < samp_T0; zz++){
			h_input[zz].x = temp[zz];
		}

		//fread( h_input, samp_T0*sizeof(char4), 1, fp_in);
		printf("a");
		cudaMemset(d_histogram, 0, hist_size*sizeof(int4));
		cudaMemcpy(d_histogram, h_histogram, hist_size*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_input, h_input, samp_T0*sizeof(float2), cudaMemcpyHostToDevice);
// must change parameters when doing more than just one go
		printf("b\n");
		mean_npol<<<grd,1>>>(d_input, d_mean); // because real/imag and 1024 time bins
		cudaEventRecord(start_ker, 0);
		//histo_kernelc<<<multiP,hist>>>(d_input, d_histogram);
		//histo_kernelf<<<multiP,hist>>>(d_input, d_histogram);
		cudaEventRecord(stop_ker, 0);
		cudaEventSynchronize(stop_ker);
		moment_order<<<1,4>>>(d_mean, d_histogram, d_moment);
		
		cudaEventRecord(start_mem, 0);
		cudaMemcpy(h_mean, d_mean, nMom*nT0*sizeof(float2), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_moment, d_moment, nMom*sizeof(float2), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_histogram, d_histogram, hist_size*sizeof(int), cudaMemcpyDeviceToHost);
		cudaEventRecord(stop_mem, 0);
		cudaEventSynchronize(stop_mem);

		// fisher-pearson coefficient of skewness.
		variance.x = h_moment[1].x;
		rms.x = sqrt(abs(h_moment[1].x));
		skew.x = h_moment[2].x/(pow(h_moment[1].x, 1.5));
		kurt.x = (h_moment[3].x/(pow(h_moment[1].x, 2))) -3;
		k += kurt.x;
		long histocount = 0;
		//test for data in histogram
		for (int oo=0; oo<hist_size; oo++){
			histocount += h_histogram[oo];
		}
		if (histocount != samp_T0){printf("histogram out of range\n");break;}

		printf("Mean.x = %f; ", h_mean[0].x);
		printf("Mean.y = %f; ", h_mean[0].y); 
		printf("Mean.z = %f; ", h_mean[0].z);
		printf("Mean.w = %f;\n", h_mean[0].w);
		printf("Variance.x = %f\n", variance.x);
		printf("Skew.x = %f\n", skew.x);
		printf("Kurt.x = %f\n", kurt.x);
		printf("\nTheta1 = %f\n", h_moment[0].x);
		printf("\nTheta2 = %f\n", h_moment[1].x);
		printf("\nTheta3 = %f\n", h_moment[2].x);
		printf("\nTheta4 = %f\n\n", h_moment[3].x);
	
		//fp_out = fopen("Boomer.dat", "w");
		for (int kk = 0; kk < hist_size; kk++){
			fprintf(stdout, "%d ", h_histogram[kk]);
		}
		//fclose(fp_out);
		cudaEventElapsedTime( &elapsedTime, start_ker, stop_ker);
		printf("\nKernel calls took: %3.1f ms\n", elapsedTime);
		elapsedTime = 10;
		cudaEventElapsedTime( &elapsedTime, start_mem, stop_mem);
		printf("Memory transfers after calls took: %3.1f ms\n", elapsedTime);
		
	ii++;
	if (ii > 0 ){break;}
	}
// There should be some streaming method that i can do to make this go faster. I do not think i need this whole thing to complete to do the next stage.cudaEventDestroy(start_mem);

	k /= 1;
	printf("\nK=%f\n", k);
	cudaEventDestroy(stop_mem);
	cudaEventDestroy(start_ker);
	cudaEventDestroy(stop_ker);

	cudaFree(d_input);
	cudaFree(d_mean);
	//cudaFree(d_histogram);
	cudaFree(d_moment);
	free(h_input);
	free(h_mean);
	//free(h_histogram);
	free(h_moment);
	fclose(fp_in);
	return 0;
}

// FILE FOR NORMAL OP 			../rcog/red_gup.dat 
// FILE FOR DISTRIB TESTING 	../../../distribution.dat
// FILE FOR CHAR ARR			../../../char_distribution.dat
