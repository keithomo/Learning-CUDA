#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>
#define N 4*1024

// dealing with 2 pipes!
int main (int argc, char **argv)
{
	int fd, fe;
	float *arr = (float*)malloc(N*sizeof(float));
	float stat[8] = {83138, 83138, 83138, 83138, 83138, 83138, 83138, 83138};
	for (int ii = 0; ii < N; ii++){arr[ii] = ii;}

	// open fifo for write
	if((fd = open("hermes", O_WRONLY)) < 0) {printf( "Fifo did not open boss!\n"); return 0;}
	else{printf("WRONLY opened successfully!\n");}	
	// open fifo for write
	if((fe = open("fehermes", O_WRONLY)) < 0) {printf( "eFifo did not open boss!\n"); return 0;}
	else{printf("eWRONLY opened successfully!\n");}

	// spectrogram
	int nw = write(fd, arr, N*sizeof(float));
	printf("nw = %d\n", nw);

	// statistics
	int snw = write(fe, stat, 8*sizeof(float));
	printf("snw = %d\n", snw);

	return 0;
}

































/*   ======================= Program to find how many bytes a pipe can contain ========================
#include <signal.h>
#include <unistd.h>
#include <limits.h>
#include <stdio.h>
int count;
void alrm_action(int);

int main()
{
	int p[2];
	int pipe_size;
	char c = 'x';
	static struct sigaction act;

	// set up the signal handler
	act.sa_handler = alrm_action;
	sigfillset(&(act.sa_mask));

	// create pipe
	if (pipe(p) == -1){
		perror("pipe call");
		return 1;
	}

	// determine size of pipe
	pipe_size = fpathconf(p[0], _PC_PIPE_BUF);
	printf("Maximum size of write to pipe: %d bytes\n", pipe_size);

	// set the signal handler
	sigaction(SIGALRM, &act, NULL);

	while(1)
	{
		//set alarm
		alarm(20);
		// write down pipe
		write(p[1], &c, 1);

	// reset alarm
		alarm(0);
	
		if ((++count % 1024) == 0){ printf("%d characters in pipe\n", count);}
	}
	return 0;
}

void alrm_action(int signo)
{
	printf("write blocked after %d characters\n", count);
}
========================================================================================================*/
/* ====================================================================================================
SYSTEM CALL: pipe();

PROTOTYPE: int pipe( int fd[2] );
	RETURNS: 0 on success                                                       
		-1 on error: errno = EMFILE (no free descriptors)
							EMFILE (system file table is full)
							EFAULT (fd array is not valid)

NOTES: fd[0] is set up for reading, fd[1] is set up for writing
=======================================================================================================*/
