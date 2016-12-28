#include<cuda.h>
#include<stdio.h>
#include <memory.h>
#include <string.h>
#include <cufft.h>
#include <cuComplex.h>

#define FFT_SIZE 512 // cannot be more than 1024
#define CH_BYTES 33548288
const int samp_T0 = 65536; // how many 160 ns samples in period of .01048576 seconds
const int nT0 = 8*128; 
const int avg_size = samp_T0/FFT_SIZE;			// fft time samples (avg_size*128) per loop
const int samps_in = nT0*samp_T0; // one picture
const int samps_out = samps_in/avg_size;
const int half_BW = FFT_SIZE/2;
const int batch = samps_in/FFT_SIZE; 	// number of times FFT performed.

	/***************************************/
	/******** Function Definitions *********/
	/***************************************/

__global__
void convert_to_floatX(char4 *in, float2 *out)
{// ignore z and w portions because that is y polarization not x
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[idx].x = (float)in[idx].x;
    out[idx].y = (float)in[idx].y;
}

__global__
void convert_to_floatY(char4 *in, float2 *out)
{// ignore z and w portions because that is y polarization not x
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[idx].x = (float)in[idx].z;
    out[idx].y = (float)in[idx].w;
}

__global__ 
void power_arr(float2 *d_result, float *d_pwr_flo)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	// calculate X(s)X(s)*  = power
    d_pwr_flo[idx] = d_result[idx].x * d_result[idx].x + d_result[idx].y * (d_result[idx].y);
}

__global__
void avg_to_ms(float *d_pwr_flo, float *d_avg_pwr) // make sure grd is recalculated
{
	// This uses two different indexes because the two arrays are of different size.
	int idxo  = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idxi  = (blockIdx.x * blockDim.x * avg_size) + threadIdx.x;
	d_avg_pwr[idxo] = 0; // just in case 
	for (int ii = 0; ii < FFT_SIZE*avg_size; ii+= FFT_SIZE){
		d_avg_pwr[idxo] += d_pwr_flo[((idxi) + ii)]; // add consecutive elements in column
	}
	d_avg_pwr[idxo] = d_avg_pwr[idxo]/avg_size;
}

int simple_fft(float2 *df_out, float2 *d_result)
{
	cufftHandle plan;
	if (cufftPlan1d(&plan, FFT_SIZE, CUFFT_C2C, batch) != CUFFT_SUCCESS){
		printf("CUFFT error: Plan creation failed\n");
		return -1;
	}
	if (cufftExecC2C(plan, df_out, d_result, CUFFT_FORWARD) != CUFFT_SUCCESS){ 
		printf("CUFFT error: ExecC2C Forward Failed\n");
		return -1;
	}
	if (cudaDeviceSynchronize() != cudaSuccess){
		printf("Cuda error: Failed to synchronize\n");
		return -1;
	}
 	//Destroy CUFFT context
    cufftDestroy(plan);
	return 0;
}

	/***************************************/
	/**************** Main *****************/
	/***************************************/

int main(int argc, char *argv[])
{	
	// variables
	dim3 blk(FFT_SIZE); // number of threads per block // must be FFT_SIZE!
    dim3 grd(samps_in/FFT_SIZE); // number of blocks // batch
	dim3 grd_avg(samps_out/FFT_SIZE); 
	char filename[60];
	int ch;
	int pol;

	// memory
	FILE *fp_in; 
	FILE *rfpo;    	
    char4 *h_input = (char4*)malloc(samps_in*sizeof(char4));	
    char4 *d_input;
    float2 *df_out;
	float2 *d_result;	 
 	float *d_pwr_flo;
	float *d_avg_pwr;
	float *h_avg_pwr = (float*)malloc( (samps_out)*sizeof(float) );

	/*************************/
	/**** Allocate Memory ****/
	/*************************/

	cudaMalloc((void**)&d_input, samps_in*sizeof(char4));
	cudaMalloc((void**)&df_out, samps_in*sizeof(float2));
	cudaMalloc((void**)&d_result, samps_in*sizeof(float2));
	cudaMalloc((void**)&d_pwr_flo, samps_in*sizeof(float));
	cudaMalloc((void**)&d_avg_pwr, (samps_out)*sizeof(float));

	/*************************/
	/*** Command Line Arg ****/
	/*************************/

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
  	printf("pol = %d\nThe next command line arg can control 'fake' number of blocks processed\n",pol);

	/*************************/
	/* initialize /main loop */
	/*************************/
	// open infile
	if (!(fp_in = fopen(filename,"r"))){
		printf("FATAL: main(); couldn't open file'\n");
		return -1;
	}
	if (!(rfpo = fopen("known_real.dat","w"))) {
	    printf("FATAL: main(): couldn't fopen() output file\n");
	   	return -1;
	}
	int ii = 0;
	while(!feof(fp_in)){ // not exactly a block each time. Just a little bit more.
		printf("%d\n", ii);
		// read file
		fread( h_input, samps_in*sizeof(char4), 1, fp_in);
		
		// CUDA START // 
		cudaMemcpy(d_input, h_input, samps_in*sizeof(char4), cudaMemcpyHostToDevice);
		if (pol == 1){	
			convert_to_floatY<<<grd,blk>>>(d_input, df_out);   
		}
		else {
			convert_to_floatX<<<grd,blk>>>(d_input, df_out);   
		}
		simple_fft(df_out, d_result);
		power_arr<<<grd,blk>>>(d_result, d_pwr_flo);
		avg_to_ms<<<grd_avg,blk>>>(d_pwr_flo, d_avg_pwr);
		for (int jj = 0; jj < (samps_out/FFT_SIZE); jj++){ // flip the bandstop to bandpass.
			cudaMemcpy(h_avg_pwr + (FFT_SIZE*jj), d_avg_pwr + half_BW + (FFT_SIZE*jj), half_BW*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_avg_pwr + half_BW + (FFT_SIZE*jj), d_avg_pwr + (FFT_SIZE*jj), half_BW*sizeof(float), cudaMemcpyDeviceToHost);
		}
		// CUDA END //	

		// write array to binary
		fwrite(h_avg_pwr, samps_out*sizeof(float), 1, rfpo);
		ii++;
		//if (ii == 1){break;}
	}	// end loop: for (int ii = 0; ii < num_of_blocks; ii++)

	// close files
	fclose(rfpo);
	fclose(fp_in);

    // cleanup memory
    free(h_input);
    cudaFree(d_input);
	cudaFree(df_out);
    cudaFree(d_result);
	cudaFree(d_pwr_flo);
	cudaFree(d_avg_pwr);
	free(h_avg_pwr);
   	return 0;
}

/*

The infile to use is "/home/scratch/komogros/guppi_56465_J1713+0747_0006.0000.raw"*/
