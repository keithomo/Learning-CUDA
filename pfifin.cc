#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#define N 4*1024


int main ()
{
	int fd, fe;
	float *arr = (float*)malloc(N*sizeof(float));
	float stat[8] = {1, 1, 1, 1, 1, 1, 1, 1};

	// open fifo for read
	if((fd = open("hermes", O_RDONLY)) < 0) {printf( "Fifo did not open boss!\n"); return 0;}
	else{printf("RDONLY opened successfully!\n");}
	// open fifo for read
	if((fe = open("fehermes", O_RDONLY)) < 0) {printf( "eFifo did not open boss!\n"); return 0;}
	else{printf("eRDONLY opened successfully!\n");}

	// print spectrogram
	int nr = read(fd, arr, (N/2)*sizeof(float));
	printf("squak!");
	nr += read(fd, &arr[N/2], (N/2)*sizeof(float));
	printf("\n");
	for (int ii = 0; ii < 16; ii++){printf("%f ", arr[ii]);}
	printf("\n");
	for (int ii = (N - 16); ii < N; ii++){printf("%f ", arr[ii]);}
	printf("\n");
	printf("nr = %d\n", nr);

	// print statistics
	int snr = read(fe, stat, 8*sizeof(float));
	for (int ii = 0; ii < 8; ii++){printf("%f ", stat[ii]);}
	printf("\n");
	printf("snr = %d\n", snr);

	return 0;
}


// make 1 pipe




// I know pipe size limit is 64k
// Exactly 16 reads for 1 channel of RCOG spectrogram
// 1 read for all channels of RCOG stats exactly 16 times over



























/*// To test random code
// To compile this program. In the command line type:
// $ gcc -o testcode testcode.c -lm
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include<iostream>
#include <fstream>
#include<iomanip>
#define N 16
using namespace std;

int main(){
	// recieve some data from stdin then print to terminal
	
	int receive[N];
	for (int ii = 0; ii < N; ii++){
		receive[N] = 0;
	}

	fread(receive, N*sizeof(int),0, stdin);

	for (int ii = 0; ii < N; ii++){
		printf("%d ", receive[ii]);
	}






	return 0;
}

	




/*/// To test random code
// To compile this program. In the command line type:
// $ gcc -o testcode testcode.c -lm
/*
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include<iostream>
#include<iomanip>
#define N 16
using namespace std;

int main(){
	// make some data then send it out to stdout.
	int kittenz[N]; //= {"abcdefghijklmno"};

	for (int ii = 0; ii < N; ii++){
		kittenz[ii] = ii;
	}
	fwrite( kittenz, N*sizeof(int), 1,  stdout);



THIS IS YOUR OUT FOR YOUR IN



	return 0;
}
*/
