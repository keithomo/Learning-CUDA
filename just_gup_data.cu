
#include<cuda.h>
#include<stdio.h>
#include <memory.h>
#include <string.h>
#include <cufft.h>
#include <complex.h>
#include <cuComplex.h>
#define CH_BYTES 33548288

int main(int argc, char *argv[]){
	
	FILE *fp_in;
	FILE *fp_out;
	const int nBlks = 128;
	int transfer_size = (CH_BYTES -(512*4));
	unsigned char *transfer_data = (unsigned char*)malloc( transfer_size*sizeof(char) );


	/*************************/
	/*** Command Line Arg ****/
	/*************************/

	char filename[60];
	int ch;
	if (argc >= 2){
		sscanf(argv[1], "%s", filename);
	} 
	else {
		printf("FATAL: main(): command line argument <filename> not specified\n");
      	return 0;
	}
	printf("filename = '%s'\n",filename);
  	if (argc>=3) {
      	sscanf( argv[2], "%d", &ch );
    } 
	else {
      	printf("FATAL: main(): command line argument <chan> not specified\n");
      	return 0;
	}
	printf("ch = %d\n", ch);

	/*************************/
	/**** read guppi data ****/
	/*************************/

	for (int ii = 0; ii < nBlks; ii++){
		printf("%d\n", ii);
		// read file
		if (ii == 0){
			// open infile
			if (!(fp_in = fopen(filename,"r"))){
				printf("FATAL: main(); couldn't open file'\n");
				return -1;
			}
			// open outfile
			if (!(fp_out = fopen("just_gup_data.dat","w"))){
				printf("FATAL: main(); couldn't open file'\n");
				return -1;
			}
			// read file
			fseek(fp_in, (6240+(CH_BYTES*(ch-1))+2048), SEEK_SET); // run to beginning of block 1 chan 30 
			fread( transfer_data, transfer_size*sizeof(char), 1, fp_in);
		}
		else{
			fseek(fp_in,((CH_BYTES*(32-ch))+6240+(CH_BYTES*(ch-1))+2048) , SEEK_CUR); // run to beginning of block 2-8 chan 30 
			fread( transfer_data, transfer_size*sizeof(char), 1, fp_in);
		}

	/*************************/
	/*** write guppi data ****/
	/*************************/

		fwrite(transfer_data, transfer_size*sizeof(char), 1, fp_out);
	}
	fclose(fp_in);
	fclose(fp_out);
	free(transfer_data);
	return 0;
}

/*

The infile to use is "/home/scratch/komogros/guppi_56465_J1713+0747_0006.0000.raw"*/



